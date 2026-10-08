"""표 데이터용 조건부 DDPM, 잔차 잡음 예측기판 (제출한 Seagate 논문 MDPI 수정 실험, 2026-10).

보관 생성기(.venv/augmenters_torch_ddpm.py)의 두 문제를 고친다.

1. 잡음 예측기 `EpsMLP` 는 입력에서 출력으로 가는 지름길이 없는 4~6층 MLP 이고 은닉 폭(256~384)이 특징 수(1,483)보다
   훨씬 작다. 잡음 예측에 필요한 항등에 가까운 사상을 표현하지 못해 학습 손실이 1.0(잡음을 0 으로 예측하는 값)에 머물렀다.
   → 출력 = c_t · x_t + F(x_t, t, y). c_t = sqrt(1 - ᾱ_t) 는 "특징이 서로 독립인 표준정규" 가정에서의 최적 선형 예측 계수다.
     F 의 마지막 층을 0 으로 시작하므로 학습 전에도 가우시안 독립 기준선과 같고, 학습은 특징 사이의 의존 구조를 더한다.
2. 잡음 일정(T=200, β 끝값 0.01 또는 0.02)이 끝까지 가지 않아 ᾱ_T 가 0.13~0.36 에 머물렀는데, 표본 추출은 순수 잡음에서
   시작했다. → 표준 일정 T=1000, 선형 β 1e-4~2e-2 (Ho et al., 2020; ᾱ_T ≈ 4e-5).

보관 클래스와 같은 사용법(fit(X_scaled, y), sample(n, y_label))을 유지해 파이프라인에 그대로 끼울 수 있다.
"""

from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sinusoidal_time_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    args = t.float()[:, None] * freqs[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


class ResBlock(nn.Module):
    def __init__(self, hidden: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        self.fc1 = nn.Linear(hidden, hidden * 2)
        self.fc2 = nn.Linear(hidden * 2, hidden)
        self.drop = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        z = self.norm(h) + cond
        z = self.fc2(self.drop(F.silu(self.fc1(z))))
        return h + z


class ResEpsNet(nn.Module):
    """잡음 예측기: ε̂ = c_t·x_t + F(x_t, t, y). F 의 출력층은 0 으로 시작한다."""

    def __init__(self, input_dim: int, hidden_dim: int = 1024, n_blocks: int = 4, time_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.time_dim = time_dim
        self.inp = nn.Linear(input_dim, hidden_dim)
        self.temb = nn.Sequential(nn.Linear(time_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.cemb = nn.Embedding(2, hidden_dim)
        self.blocks = nn.ModuleList([ResBlock(hidden_dim, dropout) for _ in range(n_blocks)])
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.out = nn.Linear(hidden_dim, input_dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, y: torch.Tensor, skip_coef: torch.Tensor) -> torch.Tensor:
        cond = self.temb(sinusoidal_time_embedding(t, self.time_dim)) + self.cemb(y)
        h = self.inp(x_t) + cond
        for b in self.blocks:
            h = b(h, cond)
        return skip_coef * x_t + self.out(F.silu(self.out_norm(h)))


@dataclass
class TorchDDPMv2Config:
    T: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 2e-2
    hidden_dim: int = 1024
    n_blocks: int = 4
    time_dim: int = 128
    dropout: float = 0.1
    train_steps: int = 20000
    batch_size: int = 256
    lr: float = 5e-4
    weight_decay: float = 1e-4
    warmup_steps: int = 500
    grad_clip: float = 1.0
    ema_decay: float = 0.999
    log_every: int = 500
    sample_batch: int = 4096
    device: str = "cuda"
    # v2.1: 과적합 방지와 속도 (기본값은 v2 와 같게 두고, 진단에서 켠다)
    standardize: bool = False      # 분위 공간에서 특징별 평균·표준편차로 다시 맞춘다 (학습 자료로만 계산)
    holdout_frac: float = 0.0      # 학습 자료 일부를 떼어 조기 종료에 쓴다 (공식 검증·테스트는 쓰지 않는다)
    min_holdout: int = 20
    eval_every: int = 250
    patience: int = 8              # 떼어 둔 자료 손실이 eval_every × patience 단계 동안 나아지지 않으면 멈춘다
    use_bf16: bool = False


class TorchTabDDPMv2Augmenter:
    """보관 `TorchTabDDPMConditionalAugmenter` 와 같은 사용법. 표본은 EMA 가중치로 뽑는다."""

    def __init__(self, seed: int = 0, cfg: Optional[TorchDDPMv2Config] = None):
        self.seed = seed
        self.cfg = cfg or TorchDDPMv2Config()
        self.model: Optional[ResEpsNet] = None
        self.ema: Optional[ResEpsNet] = None
        self.loss_log: List[dict] = []

    def _schedule(self, device: torch.device) -> None:
        c = self.cfg
        self.betas = torch.linspace(c.beta_start, c.beta_end, c.T, device=device, dtype=torch.float32)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        ab_prev = torch.cat([torch.ones(1, device=device), self.alpha_bars[:-1]])
        self.post_var = self.betas * (1.0 - ab_prev) / (1.0 - self.alpha_bars)

    def skip_coef(self, t: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(1.0 - self.alpha_bars[t]).view(-1, 1)

    def eps_pred(self, model: ResEpsNet, x_t: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return model(x_t, t, y, self.skip_coef(t))

    def _std(self, X: np.ndarray) -> np.ndarray:
        if getattr(self, "mu", None) is None:
            return np.asarray(X, dtype=np.float32)
        return ((np.asarray(X, dtype=np.float32) - self.mu) / self.sd).astype(np.float32)

    def fit(self, X_scaled: np.ndarray, y: np.ndarray) -> None:
        set_all_seeds(self.seed)
        c = self.cfg
        device = torch.device(c.device if torch.cuda.is_available() else "cpu")
        self.device = device
        X_np = np.ascontiguousarray(X_scaled, dtype=np.float32)
        y_np = np.ascontiguousarray(y.astype(np.int64))
        # 학습 자료 안에서 떼어 둘 부분 (클래스 비율 유지). 공식 검증·테스트 자료는 쓰지 않는다.
        rng = np.random.default_rng(self.seed + 104729)
        hold = np.zeros(len(y_np), dtype=bool)
        if c.holdout_frac > 0:
            for cls in np.unique(y_np):
                idx = np.flatnonzero(y_np == cls)
                k = int(round(len(idx) * c.holdout_frac))
                if k >= c.min_holdout and len(idx) - k >= c.min_holdout:
                    hold[rng.choice(idx, size=k, replace=False)] = True
        self.holdout_mask = hold
        # 표준화 값은 떼어 두지 않은 부분으로만 계산한다
        self.mu, self.sd = None, None
        if c.standardize:
            self.mu = X_np[~hold].mean(axis=0).astype(np.float32)
            sd = X_np[~hold].std(axis=0).astype(np.float32)
            self.sd = np.where(sd > 1e-3, sd, 1.0).astype(np.float32)
            X_np = self._std(X_np)
        X = torch.from_numpy(X_np[~hold]).to(device)
        Y = torch.from_numpy(y_np[~hold]).to(device)
        self._schedule(device)
        self.model = ResEpsNet(X.shape[1], c.hidden_dim, c.n_blocks, c.time_dim, c.dropout).to(device)
        self.ema = copy.deepcopy(self.model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        opt = torch.optim.AdamW(self.model.parameters(), lr=c.lr, weight_decay=c.weight_decay)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(1, c.warmup_steps)))
        g = torch.Generator(device=device)
        g.manual_seed(self.seed)
        # 떼어 둔 자료의 손실은 고정된 (t, ε) 로 잰다
        if hold.any():
            Xh = torch.from_numpy(X_np[hold]).to(device)
            Yh = torch.from_numpy(y_np[hold]).to(device)
            gh = torch.Generator(device=device)
            gh.manual_seed(self.seed + 1)
            reps = 4
            th = torch.randint(0, c.T, (reps, Xh.shape[0]), device=device, generator=gh)
            eh = torch.randn((reps,) + tuple(Xh.shape), device=device, generator=gh)
        self.best = {"step": 0, "holdout_loss": float("nan")}
        best_state, bad = None, 0
        n = X.shape[0]
        self.model.train()
        ema_params = list(self.ema.parameters())
        model_params = list(self.model.parameters())
        amp = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if (c.use_bf16 and device.type == "cuda") else None
        run = 0.0
        for step in range(1, c.train_steps + 1):
            idx = torch.randint(0, n, (min(c.batch_size, n),), device=device, generator=g)
            x0, yb = X[idx], Y[idx]
            t = torch.randint(0, c.T, (x0.shape[0],), device=device, generator=g)
            eps = torch.randn(x0.shape, device=device, generator=g)
            ab = self.alpha_bars[t].view(-1, 1)
            x_t = torch.sqrt(ab) * x0 + torch.sqrt(1.0 - ab) * eps
            if amp is not None:
                with amp:
                    pred = self.eps_pred(self.model, x_t, t, yb)
            else:
                pred = self.eps_pred(self.model, x_t, t, yb)
            loss = F.mse_loss(pred.float(), eps)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), c.grad_clip)
            opt.step()
            sched.step()
            with torch.no_grad():
                torch._foreach_mul_(ema_params, c.ema_decay)
                torch._foreach_add_(ema_params, model_params, alpha=1.0 - c.ema_decay)
            run += float(loss.detach())
            if step % c.log_every == 0:
                self.loss_log.append({"step": step, "train_loss": run / c.log_every})
                run = 0.0
            if hold.any() and step % c.eval_every == 0:
                with torch.no_grad():
                    tot = 0.0
                    for r in range(reps):
                        ab_h = self.alpha_bars[th[r]].view(-1, 1)
                        xt_h = torch.sqrt(ab_h) * Xh + torch.sqrt(1.0 - ab_h) * eh[r]
                        tot += float(F.mse_loss(self.eps_pred(self.ema, xt_h, th[r], Yh).float(), eh[r]))
                    hl = tot / reps
                self.loss_log.append({"step": step, "holdout_loss": hl})
                if not (hl >= self.best["holdout_loss"]):  # nan 이거나 더 작으면 갱신
                    self.best = {"step": step, "holdout_loss": hl}
                    best_state = {k: v.detach().clone() for k, v in self.ema.state_dict().items()}
                    bad = 0
                else:
                    bad += 1
                    if bad >= c.patience:
                        break
        self.steps_run = step
        if best_state is not None:
            self.ema.load_state_dict(best_state)
        self.model.eval()

    @torch.no_grad()
    def denoise_loss(self, X_scaled: np.ndarray, y: np.ndarray, seed: int, n_draws: int = 8, use_ema: bool = True) -> dict:
        """고정 난수로 같은 (t, ε) 를 뽑아 모델·0 예측·가우시안 독립 예측의 잡음 예측 MSE 를 잰다."""
        dev = self.device
        X = torch.from_numpy(np.ascontiguousarray(self._std(X_scaled), dtype=np.float32)).to(dev)
        Y = torch.from_numpy(np.ascontiguousarray(y.astype(np.int64))).to(dev)
        g = torch.Generator(device=dev)
        g.manual_seed(seed)
        m = self.ema if use_ema else self.model
        tot = {"model": 0.0, "zero": 0.0, "gauss_indep": 0.0}
        cnt = 0
        for _ in range(n_draws):
            t = torch.randint(0, self.cfg.T, (X.shape[0],), device=dev, generator=g)
            eps = torch.randn(X.shape, device=dev, generator=g)
            ab = self.alpha_bars[t].view(-1, 1)
            x_t = torch.sqrt(ab) * X + torch.sqrt(1.0 - ab) * eps
            tot["model"] += float(F.mse_loss(self.eps_pred(m, x_t, t, Y), eps, reduction="sum"))
            tot["zero"] += float((eps ** 2).sum())
            tot["gauss_indep"] += float(((self.skip_coef(t) * x_t - eps) ** 2).sum())
            cnt += eps.numel()
        return {k: v / cnt for k, v in tot.items()}

    @torch.no_grad()
    def sample(self, n: int, y_label: int = 1) -> np.ndarray:
        dev = self.device
        g = torch.Generator(device=dev)
        g.manual_seed(self.seed + 7919)
        out = []
        d = self.ema.inp.in_features
        for start in range(0, n, self.cfg.sample_batch):
            b = min(self.cfg.sample_batch, n - start)
            x = torch.randn((b, d), device=dev, generator=g)
            y = torch.full((b,), int(y_label), device=dev, dtype=torch.long)
            amp = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if (self.cfg.use_bf16 and dev.type == "cuda") else None
            for ti in range(self.cfg.T - 1, -1, -1):
                t = torch.full((b,), ti, device=dev, dtype=torch.long)
                if amp is not None:
                    with amp:
                        eps_hat = self.eps_pred(self.ema, x, t, y).float()
                else:
                    eps_hat = self.eps_pred(self.ema, x, t, y)
                a, ab, bt = self.alphas[ti], self.alpha_bars[ti], self.betas[ti]
                mean = (x - bt / torch.sqrt(1.0 - ab) * eps_hat) / torch.sqrt(a)
                if ti > 0:
                    x = mean + torch.sqrt(self.post_var[ti]) * torch.randn((b, d), device=dev, generator=g)
                else:
                    x = mean
            out.append(x.float().cpu().numpy())
        S = np.vstack(out)
        if getattr(self, "mu", None) is not None:
            S = S * self.sd + self.mu
        return S
