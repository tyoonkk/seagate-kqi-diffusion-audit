# augmenters_torch_ddpm.py
from __future__ import annotations

import math
import random
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sinusoidal_time_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """
    t: (batch,) int64 timesteps
    returns: (batch, dim)
    """
    device = t.device
    half = dim // 2
    # log-spaced frequencies
    freqs = torch.exp(torch.linspace(math.log(1e-4), math.log(1.0), steps=half, device=device))
    args = t[:, None].float() * freqs[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class EpsMLP(nn.Module):
    def __init__(self, input_dim: int, time_dim: int = 128, class_dim: int = 16,
                 hidden_dim: int = 256, n_layers: int = 4, dropout: float = 0.1):
        super().__init__()
        self.time_dim = time_dim
        self.class_embed = nn.Embedding(2, class_dim)

        layers = []
        in_dim = input_dim + time_dim + class_dim
        for _ in range(n_layers - 1):
            layers += [nn.Linear(in_dim, hidden_dim), nn.SiLU(), nn.Dropout(dropout)]
            in_dim = hidden_dim
        layers += [nn.Linear(in_dim, input_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        t_emb = sinusoidal_time_embedding(t, self.time_dim)
        y_emb = self.class_embed(y)
        h = torch.cat([x_t, t_emb, y_emb], dim=-1)
        return self.net(h)


@dataclass
class TorchDDPMConfig:
    T: int = 200                  # 1000도 가능하지만 SECOM은 200~400이 현실적
    beta_start: float = 1e-4
    beta_end: float = 2e-2
    time_dim: int = 128
    class_dim: int = 16
    hidden_dim: int = 256
    n_layers: int = 4
    dropout: float = 0.1

    epochs: int = 200
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-5
    grad_clip: float = 1.0

    use_amp: bool = True          # GPU면 mixed precision
    device: str = "cuda"          # "cuda" or "cpu"
    num_workers: int = 0
    pin_memory: bool = True
    gpu_resident_batches: bool = True
    gpu_resident_max_mb: int = 2048


class TorchTabDDPMConditionalAugmenter:
    """
    연속형 전용 Conditional DDPM (SECOM에 적합)
    - 입력 X는 이미 "diffusion용 스케일링(standard/quantile)"이 된 상태를 권장
    - sample()은 같은 스케일 공간에서 synthetic을 반환
    """
    def __init__(self, seed: int = 0, cfg: Optional[TorchDDPMConfig] = None):
        self.seed = seed
        self.cfg = cfg or TorchDDPMConfig()
        self.model: Optional[EpsMLP] = None

        self.betas = None
        self.alphas = None
        self.alpha_bars = None
        self.feature_noise_scale_np: Optional[np.ndarray] = None
        self.feature_noise_scale_t: Optional[torch.Tensor] = None
        self.sampler_correction_every: int = 0
        self.sampler_correction_alpha: float = 0.0
        self.spike_idx_np: Optional[np.ndarray] = None
        self.spike_low_np: Optional[np.ndarray] = None
        self.spike_high_np: Optional[np.ndarray] = None
        self.spike_median_np: Optional[np.ndarray] = None
        self.corr_group_idx_np: Optional[np.ndarray] = None
        self.corr_group_mean_np: Optional[np.ndarray] = None
        self.corr_group_components_np: Optional[np.ndarray] = None
        self.spike_idx_t: Optional[torch.Tensor] = None
        self.spike_low_t: Optional[torch.Tensor] = None
        self.spike_high_t: Optional[torch.Tensor] = None
        self.spike_median_t: Optional[torch.Tensor] = None
        self.corr_group_idx_t: Optional[torch.Tensor] = None
        self.corr_group_mean_t: Optional[torch.Tensor] = None
        self.corr_group_components_t: Optional[torch.Tensor] = None

    def set_tabdiff_controls(
        self,
        feature_noise_scale: Optional[np.ndarray] = None,
        *,
        sampler_correction_every: int = 0,
        sampler_correction_alpha: float = 0.0,
        spike_idx: Optional[np.ndarray] = None,
        spike_low: Optional[np.ndarray] = None,
        spike_high: Optional[np.ndarray] = None,
        spike_median: Optional[np.ndarray] = None,
        corr_group_idx: Optional[np.ndarray] = None,
        corr_group_mean: Optional[np.ndarray] = None,
        corr_group_components: Optional[np.ndarray] = None,
    ) -> None:
        self.feature_noise_scale_np = None if feature_noise_scale is None else np.asarray(feature_noise_scale, dtype=np.float32)
        self.sampler_correction_every = max(0, int(sampler_correction_every))
        self.sampler_correction_alpha = float(np.clip(sampler_correction_alpha, 0.0, 1.0))
        self.spike_idx_np = None if spike_idx is None else np.asarray(spike_idx, dtype=np.int64)
        self.spike_low_np = None if spike_low is None else np.asarray(spike_low, dtype=np.float32)
        self.spike_high_np = None if spike_high is None else np.asarray(spike_high, dtype=np.float32)
        self.spike_median_np = None if spike_median is None else np.asarray(spike_median, dtype=np.float32)
        self.corr_group_idx_np = None if corr_group_idx is None else np.asarray(corr_group_idx, dtype=np.int64)
        self.corr_group_mean_np = None if corr_group_mean is None else np.asarray(corr_group_mean, dtype=np.float32)
        self.corr_group_components_np = None if corr_group_components is None else np.asarray(corr_group_components, dtype=np.float32)

    def _move_controls_to_device(self, device: torch.device, input_dim: int) -> None:
        if self.feature_noise_scale_np is not None:
            scale = np.asarray(self.feature_noise_scale_np, dtype=np.float32)
            if scale.shape[0] != input_dim:
                raise ValueError("feature_noise_scale dimension does not match input_dim")
            self.feature_noise_scale_t = torch.from_numpy(scale).to(device=device, dtype=torch.float32).view(1, -1)
        else:
            self.feature_noise_scale_t = torch.ones(1, input_dim, device=device, dtype=torch.float32)

        self.spike_idx_t = None
        self.spike_low_t = None
        self.spike_high_t = None
        self.spike_median_t = None
        if self.spike_idx_np is not None and self.spike_idx_np.size > 0:
            self.spike_idx_t = torch.from_numpy(self.spike_idx_np).to(device=device, dtype=torch.long)
            self.spike_low_t = torch.from_numpy(self.spike_low_np).to(device=device, dtype=torch.float32)
            self.spike_high_t = torch.from_numpy(self.spike_high_np).to(device=device, dtype=torch.float32)
            self.spike_median_t = torch.from_numpy(self.spike_median_np).to(device=device, dtype=torch.float32)

        self.corr_group_idx_t = None
        self.corr_group_mean_t = None
        self.corr_group_components_t = None
        if self.corr_group_idx_np is not None and self.corr_group_idx_np.size > 0:
            self.corr_group_idx_t = torch.from_numpy(self.corr_group_idx_np).to(device=device, dtype=torch.long)
            self.corr_group_mean_t = torch.from_numpy(self.corr_group_mean_np).to(device=device, dtype=torch.float32)
            self.corr_group_components_t = torch.from_numpy(self.corr_group_components_np).to(device=device, dtype=torch.float32)

    def _apply_sampler_correction(self, x_t: torch.Tensor) -> torch.Tensor:
        alpha = float(self.sampler_correction_alpha)
        if alpha <= 0.0:
            return x_t

        if self.corr_group_idx_t is not None and self.corr_group_components_t is not None:
            x_group = x_t.index_select(1, self.corr_group_idx_t)
            centered = x_group - self.corr_group_mean_t.view(1, -1)
            coeffs = centered @ self.corr_group_components_t.transpose(0, 1)
            proj = coeffs @ self.corr_group_components_t + self.corr_group_mean_t.view(1, -1)
            x_group = (1.0 - alpha) * x_group + alpha * proj
            x_t = x_t.clone()
            x_t[:, self.corr_group_idx_t] = x_group

        if self.spike_idx_t is not None:
            x_spike = x_t.index_select(1, self.spike_idx_t)
            x_spike = torch.minimum(torch.maximum(x_spike, self.spike_low_t.view(1, -1)), self.spike_high_t.view(1, -1))
            x_spike = (1.0 - alpha) * x_spike + alpha * self.spike_median_t.view(1, -1)
            if x_t.is_leaf:
                x_t = x_t.clone()
            x_t[:, self.spike_idx_t] = x_spike
        return x_t

    def _init_schedule(self, device: torch.device):
        T = self.cfg.T
        betas = torch.linspace(self.cfg.beta_start, self.cfg.beta_end, T, device=device)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.betas, self.alphas, self.alpha_bars = betas, alphas, alpha_bars

    def fit(self, X_scaled: np.ndarray, y: np.ndarray) -> None:
        set_all_seeds(self.seed)

        device = torch.device(self.cfg.device if torch.cuda.is_available() else "cpu")
        X = torch.from_numpy(np.ascontiguousarray(X_scaled)).to(dtype=torch.float32)
        y_t = torch.from_numpy(np.ascontiguousarray(y.astype(np.int64, copy=False)))

        self.model = EpsMLP(
            input_dim=X.shape[1],
            time_dim=self.cfg.time_dim,
            class_dim=self.cfg.class_dim,
            hidden_dim=self.cfg.hidden_dim,
            n_layers=self.cfg.n_layers,
            dropout=self.cfg.dropout
        ).to(device)

        self._init_schedule(device)
        self._move_controls_to_device(device, input_dim=X.shape[1])

        opt = torch.optim.AdamW(self.model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        use_amp = bool(self.cfg.use_amp and device.type == "cuda")
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        else:
            scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        non_blocking = bool(self.cfg.pin_memory and device.type == "cuda")
        total_bytes = (X.numel() * X.element_size()) + (y_t.numel() * y_t.element_size())
        use_gpu_resident = (
            bool(self.cfg.gpu_resident_batches)
            and device.type == "cuda"
            and total_bytes <= int(self.cfg.gpu_resident_max_mb) * 1024 * 1024
        )
        if use_gpu_resident:
            X = X.to(device)
            y_t = y_t.to(device)
            dl = None
            print(f"[TorchDDPM] gpu-resident batches enabled ({total_bytes / (1024 * 1024):.1f} MB)")
        else:
            ds = TensorDataset(X, y_t)
            loader_kwargs = {
                "num_workers": max(0, int(self.cfg.num_workers)),
                "pin_memory": bool(self.cfg.pin_memory and device.type == "cuda"),
            }
            if loader_kwargs["num_workers"] > 0:
                loader_kwargs["persistent_workers"] = True
            dl = DataLoader(ds, batch_size=self.cfg.batch_size, shuffle=True, drop_last=False, **loader_kwargs)

        self.model.train()
        for epoch in range(1, self.cfg.epochs + 1):
            total = 0.0
            n = 0
            if use_gpu_resident:
                order = torch.randperm(X.shape[0], device=device)
                batch_iter = (
                    (
                        X.index_select(0, order[start : start + self.cfg.batch_size]),
                        y_t.index_select(0, order[start : start + self.cfg.batch_size]),
                    )
                    for start in range(0, X.shape[0], self.cfg.batch_size)
                )
            else:
                batch_iter = dl

            for x0, yb in batch_iter:
                if not use_gpu_resident:
                    x0 = x0.to(device, non_blocking=non_blocking)
                    yb = yb.to(device, non_blocking=non_blocking)

                bsz = x0.size(0)
                t = torch.randint(0, self.cfg.T, (bsz,), device=device, dtype=torch.long)
                eps = torch.randn_like(x0) * self.feature_noise_scale_t

                ab = self.alpha_bars[t].view(-1, 1)
                x_t = torch.sqrt(ab) * x0 + torch.sqrt(1.0 - ab) * eps

                opt.zero_grad(set_to_none=True)
                if use_amp and hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
                    amp_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.float16)
                else:
                    amp_ctx = nullcontext()
                with amp_ctx:
                    eps_pred = self.model(x_t, t, yb)
                    loss = F.mse_loss(eps_pred, eps)

                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                if self.cfg.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                scaler.step(opt)
                scaler.update()

                total += float(loss.item()) * bsz
                n += bsz

            # 필요하면 여기서 val loss 기반 early stopping 추가 가능
            if epoch % 20 == 0 or epoch == 1:
                print(f"[TorchDDPM] epoch={epoch:04d} loss={total / max(1, n):.6f}")

        self.model.eval()

    @torch.no_grad()
    def sample(self, n_samples: int, y_label: int = 1) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Call fit() before sample().")

        device = next(self.model.parameters()).device
        noise_scale = self.feature_noise_scale_t
        if noise_scale is None:
            noise_scale = torch.ones(1, self.model.net[-1].out_features, device=device, dtype=torch.float32)
        x_t = torch.randn(n_samples, self.model.net[-1].out_features, device=device) * noise_scale
        y = torch.full((n_samples,), int(y_label), dtype=torch.long, device=device)

        T = self.cfg.T
        for t_step in reversed(range(T)):
            t = torch.full((n_samples,), t_step, device=device, dtype=torch.long)
            eps_pred = self.model(x_t, t, y)

            beta_t = self.betas[t_step]
            alpha_t = self.alphas[t_step]
            ab_t = self.alpha_bars[t_step]

            if t_step > 0:
                z = torch.randn_like(x_t) * noise_scale
            else:
                z = torch.zeros_like(x_t)

            # DDPM ancestral update
            x_t = (1.0 / torch.sqrt(alpha_t)) * (x_t - (beta_t / torch.sqrt(1.0 - ab_t)) * eps_pred) + torch.sqrt(beta_t) * z
            if self.sampler_correction_every > 0 and t_step > 0 and (t_step % self.sampler_correction_every == 0):
                x_t = self._apply_sampler_correction(x_t)

        return x_t.detach().cpu().numpy()

    def save(self, path: str) -> None:
        if self.model is None:
            raise RuntimeError("Nothing to save; fit() first.")
        torch.save({
            "seed": self.seed,
            "cfg": self.cfg.__dict__,
            "model_state": self.model.state_dict(),
        }, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location="cpu")
        self.seed = ckpt["seed"]
        self.cfg = TorchDDPMConfig(**ckpt["cfg"])
        # model init needs input_dim; store it in state dict keys
        # easiest: infer from last layer weight
        last_w = ckpt["model_state"]["net.%d.weight" % (len(ckpt["model_state"]) // 2)] if False else None
        # Practical approach: user should call load_with_input_dim(...)
        raise NotImplementedError("For simplicity, re-init with known input_dim and then load_state_dict.")


def load_torch_ddpm(path: str, input_dim: int) -> TorchTabDDPMConditionalAugmenter:
    ckpt = torch.load(path, map_location="cpu")
    cfg = TorchDDPMConfig(**ckpt["cfg"])
    aug = TorchTabDDPMConditionalAugmenter(seed=ckpt["seed"], cfg=cfg)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    aug.model = EpsMLP(input_dim=input_dim, time_dim=cfg.time_dim, class_dim=cfg.class_dim,
                       hidden_dim=cfg.hidden_dim, n_layers=cfg.n_layers, dropout=cfg.dropout).to(device)
    aug._init_schedule(device)
    aug.model.load_state_dict(ckpt["model_state"])
    aug.model.eval()
    return aug
