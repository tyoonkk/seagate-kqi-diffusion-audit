"""표 데이터용 혼합형 조건부 확산 생성기 (제출한 Seagate 논문 MDPI 수정 실험, 2026-10).

v2.1(augmenters_torch_ddpm_v2.py)은 모든 특징을 분위 정규화한 연속값으로 다뤘다. 진단 3차에서 v2.1 은 잡음 예측 손실이
자명한 기준선보다 훨씬 낮았지만(과제 10: 0.053 대 0.237), 0/1 이진 특징의 드문 값(예: 불량의 3% 만 1)을 거의 만들지 못했다.
실제 학습 불량에서 드문 이진 값만 지워도 떼어 둔 불량과의 판별 AUC 가 0.562 에서 0.997 로 오른다(과제 10).
TabDDPM(Kotelnikov et al., 2023)은 범주형 특징을 가우시안 대신 다항 확산(Hoogeboom et al., 2021)으로 다룬다. 이 판도 그렇게 한다.

- 연속 특징: v2.1 과 같다. ε̂ = √(1−ᾱ_t)·x_t + F_c(·), 분위 정규화 뒤 특징별 표준화, T=1000 선형 β 1e-4→0.02.
- 이진 특징: 각 열을 범주 2개짜리 변수로 보고 다항 확산을 쓴다. q(x_t=1|x_0) = ᾱ_t·x_0 + (1−ᾱ_t)/2.
  망은 x_0=1 의 로짓을 낸다. 로짓 = logit(p_j(y)) + F_b(·) 이고 F_b 의 출력층은 0 으로 시작한다(처음에는 학습 자료의 1 비율).
  p_θ(x_{t−1}|x_t) 는 예측한 x_0 분포를 q(x_{t−1}|x_t,x_0) 에 넣어 얻는다(Hoogeboom et al., 2021; TabDDPM 과 같음).
  손실은 KL(q(x_{t−1}|x_t,x_0) ‖ p_θ(x_{t−1}|x_t)) 이고 t=0 에서는 −log p_θ(x_0|x_1) 이 된다.
- 전체 손실 = 연속 특징 평균 MSE + T × 이진 특징 평균 KL. TabDDPM 구현처럼 균등 추출한 t 의 KL 을 1/p(t)=T 로 키워
  변분 하한 전체의 크기에 맞춘다(그러지 않으면 이진 손실이 연속 손실의 약 1/1000 이라 이진 특징이 거의 학습되지 않는다).
- 학습 자료 안에서 클래스별 20% 를 떼어 두고 그 손실로 조기 종료한다(공식 검증·테스트 자료는 쓰지 않는다). EMA 가중치로 추출한다.

사용: fit(X_cont_scaled, X_bin(0/1), y), sample(n, y_label) → (X_cont_scaled, X_bin)
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from augmenters_torch_ddpm_v2 import ResBlock, set_all_seeds, sinusoidal_time_embedding

EPS = 1e-12


class MixedNet(nn.Module):
    def __init__(self, d_cont: int, d_bin: int, hidden: int, n_blocks: int, time_dim: int, dropout: float, base_logit: torch.Tensor):
        super().__init__()
        self.d_cont, self.d_bin, self.time_dim = d_cont, d_bin, time_dim
        self.inp = nn.Linear(d_cont + d_bin, hidden)
        self.temb = nn.Sequential(nn.Linear(time_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.cemb = nn.Embedding(2, hidden)
        self.blocks = nn.ModuleList([ResBlock(hidden, dropout) for _ in range(n_blocks)])
        self.out_norm = nn.LayerNorm(hidden)
        self.out_c = nn.Linear(hidden, max(d_cont, 1))
        self.out_b = nn.Linear(hidden, max(d_bin, 1))
        for lin in (self.out_c, self.out_b):
            nn.init.zeros_(lin.weight)
            nn.init.zeros_(lin.bias)
        self.register_buffer("base_logit", base_logit)  # (2, d_bin): 클래스별 학습 자료의 1 비율 로짓

    def forward(self, xc_t, xb_t, t, y, skip_coef):
        parts = []
        if self.d_cont:
            parts.append(xc_t)
        if self.d_bin:
            parts.append(2.0 * xb_t - 1.0)
        cond = self.temb(sinusoidal_time_embedding(t, self.time_dim)) + self.cemb(y)
        h = self.inp(torch.cat(parts, dim=1)) + cond
        for b in self.blocks:
            h = b(h, cond)
        z = F.silu(self.out_norm(h))
        eps = skip_coef * xc_t + self.out_c(z)[:, : self.d_cont] if self.d_cont else None
        logit = None
        if self.d_bin:
            logit = self.base_logit[y] + self.out_b(z)[:, : self.d_bin]
        return eps, logit


@dataclass
class TorchDDPMv3Config:
    T: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 2e-2
    hidden_dim: int = 1024
    n_blocks: int = 4
    time_dim: int = 128
    dropout: float = 0.1
    train_steps: int = 30000
    batch_size: int = 256
    lr: float = 5e-4
    weight_decay: float = 1e-4
    warmup_steps: int = 500
    grad_clip: float = 1.0
    ema_decay: float = 0.999
    log_every: int = 500
    sample_batch: int = 4096
    device: str = "cuda"
    standardize: bool = True
    holdout_frac: float = 0.2
    min_holdout: int = 20
    eval_every: int = 250
    patience: int = 8
    use_bf16: bool = True


class TorchTabDDPMv3Augmenter:
    def __init__(self, seed: int = 0, cfg: Optional[TorchDDPMv3Config] = None):
        self.seed = seed
        self.cfg = cfg or TorchDDPMv3Config()
        self.loss_log: List[dict] = []

    # ── 일정 ──
    def _schedule(self, device) -> None:
        c = self.cfg
        self.betas = torch.linspace(c.beta_start, c.beta_end, c.T, device=device, dtype=torch.float32)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.ab_prev = torch.cat([torch.ones(1, device=device), self.alpha_bars[:-1]])  # t=0 이면 1 (깨끗한 자료)
        self.post_var = self.betas * (1.0 - self.ab_prev) / (1.0 - self.alpha_bars)

    def _std(self, X):
        if self.mu is None:
            return np.asarray(X, dtype=np.float32)
        return ((np.asarray(X, dtype=np.float32) - self.mu) / self.sd).astype(np.float32)

    def _pred(self, model, xc, xb, t, y, amp):
        skip = torch.sqrt(1.0 - self.alpha_bars[t]).view(-1, 1)
        if amp is not None:
            with amp:
                eps, logit = model(xc, xb, t, y, skip)
            return (eps.float() if eps is not None else None), (logit.float() if logit is not None else None)
        return model(xc, xb, t, y, skip)

    def _bin_post(self, xb_t, t, x0_prob1):
        """p(x_{t-1}=1 | x_t, x0 분포). x0_prob1 은 x_0=1 의 확률(참값이면 0/1)."""
        a = self.alphas[t].view(-1, 1)
        abp = self.ab_prev[t].view(-1, 1)
        A1 = a * xb_t + (1.0 - a) * 0.5          # q(x_t | x_{t-1}=1)
        A0 = a * (1.0 - xb_t) + (1.0 - a) * 0.5  # q(x_t | x_{t-1}=0)
        B1 = abp * x0_prob1 + (1.0 - abp) * 0.5
        B0 = abp * (1.0 - x0_prob1) + (1.0 - abp) * 0.5
        u1, u0 = A1 * B1, A0 * B0
        return u1 / (u1 + u0).clamp_min(EPS)

    def _bin_kl(self, xb_t, xb0, t, logit):
        q1 = self._bin_post(xb_t, t, xb0)
        p1 = self._bin_post(xb_t, t, torch.sigmoid(logit))
        q0, p0 = 1.0 - q1, 1.0 - p1
        kl = q1 * (torch.log(q1.clamp_min(EPS)) - torch.log(p1.clamp_min(EPS))) + q0 * (torch.log(q0.clamp_min(EPS)) - torch.log(p0.clamp_min(EPS)))
        return kl

    def _noisy(self, xc0, xb0, t, eps, ub):
        ab = self.alpha_bars[t].view(-1, 1)
        xc_t = torch.sqrt(ab) * xc0 + torch.sqrt(1.0 - ab) * eps if xc0 is not None else None
        xb_t = None
        if xb0 is not None:
            prob1 = ab * xb0 + (1.0 - ab) * 0.5
            xb_t = (ub < prob1).float()
        return xc_t, xb_t

    def _loss_parts(self, model, xc0, xb0, y, t, eps, ub, amp):
        xc_t, xb_t = self._noisy(xc0, xb0, t, eps, ub)
        eps_hat, logit = self._pred(model, xc_t if xc_t is not None else torch.zeros((len(y), 0), device=y.device),
                                    xb_t if xb_t is not None else torch.zeros((len(y), 0), device=y.device), t, y, amp)
        lc = F.mse_loss(eps_hat.float(), eps) if xc0 is not None else torch.zeros((), device=y.device)
        lb = self.cfg.T * self._bin_kl(xb_t, xb0, t, logit.float()).mean() if xb0 is not None else torch.zeros((), device=y.device)
        return lc, lb

    # ── 학습 ──
    def fit(self, X_cont: np.ndarray, X_bin: np.ndarray, y: np.ndarray) -> None:
        set_all_seeds(self.seed)
        c = self.cfg
        dev = torch.device(c.device if torch.cuda.is_available() else "cpu")
        self.device = dev
        Xc = np.ascontiguousarray(X_cont, dtype=np.float32)
        Xb = np.ascontiguousarray(X_bin, dtype=np.float32)
        yv = np.ascontiguousarray(np.asarray(y).astype(np.int64))
        self.d_cont, self.d_bin = Xc.shape[1], Xb.shape[1]
        rng = np.random.default_rng(self.seed + 104729)
        hold = np.zeros(len(yv), dtype=bool)
        if c.holdout_frac > 0:
            for cls in np.unique(yv):
                idx = np.flatnonzero(yv == cls)
                k = int(round(len(idx) * c.holdout_frac))
                if k >= c.min_holdout and len(idx) - k >= c.min_holdout:
                    hold[rng.choice(idx, size=k, replace=False)] = True
        self.holdout_mask = hold
        self.mu, self.sd = None, None
        if c.standardize and self.d_cont:
            self.mu = Xc[~hold].mean(axis=0).astype(np.float32)
            sd = Xc[~hold].std(axis=0).astype(np.float32)
            self.sd = np.where(sd > 1e-3, sd, 1.0).astype(np.float32)
            Xc = self._std(Xc)
        # 클래스별 이진 1 비율 (학습 부분만), 로짓으로 기준선을 만든다
        base = np.zeros((2, self.d_bin), dtype=np.float32)
        allr = Xb[~hold].mean(axis=0) if self.d_bin else np.zeros(0)
        for cls in (0, 1):
            m = (~hold) & (yv == cls)
            r = Xb[m].mean(axis=0) if (self.d_bin and m.any()) else allr
            r = np.clip(r, 1e-4, 1 - 1e-4)
            base[cls] = np.log(r / (1 - r))
        self.base_rate = 1.0 / (1.0 + np.exp(-base))
        self._schedule(dev)
        model = MixedNet(self.d_cont, self.d_bin, c.hidden_dim, c.n_blocks, c.time_dim, c.dropout, torch.from_numpy(base)).to(dev)
        self.model = model
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        tr = ~hold
        XC = torch.from_numpy(Xc[tr]).to(dev) if self.d_cont else None
        XB = torch.from_numpy(Xb[tr]).to(dev) if self.d_bin else None
        Y = torch.from_numpy(yv[tr]).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=c.lr, weight_decay=c.weight_decay)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(1, c.warmup_steps)))
        g = torch.Generator(device=dev)
        g.manual_seed(self.seed)
        amp = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if (c.use_bf16 and dev.type == "cuda") else None
        if hold.any():
            gh = torch.Generator(device=dev)
            gh.manual_seed(self.seed + 1)
            HC = torch.from_numpy(Xc[hold]).to(dev) if self.d_cont else None
            HB = torch.from_numpy(Xb[hold]).to(dev) if self.d_bin else None
            HY = torch.from_numpy(yv[hold]).to(dev)
            reps = 4
            nh = int(hold.sum())
            th = torch.randint(0, c.T, (reps, nh), device=dev, generator=gh)
            eh = torch.randn((reps, nh, max(self.d_cont, 1)), device=dev, generator=gh)[:, :, : self.d_cont]
            uh = torch.rand((reps, nh, max(self.d_bin, 1)), device=dev, generator=gh)[:, :, : self.d_bin]
        self.best = {"step": 0, "holdout_loss": float("nan")}
        best_state, bad = None, 0
        ema_p, mod_p = list(self.ema.parameters()), list(model.parameters())
        n = len(Y)
        run = 0.0
        model.train()
        for step in range(1, c.train_steps + 1):
            idx = torch.randint(0, n, (min(c.batch_size, n),), device=dev, generator=g)
            t = torch.randint(0, c.T, (len(idx),), device=dev, generator=g)
            eps = torch.randn((len(idx), max(self.d_cont, 1)), device=dev, generator=g)[:, : self.d_cont]
            ub = torch.rand((len(idx), max(self.d_bin, 1)), device=dev, generator=g)[:, : self.d_bin]
            lc, lb = self._loss_parts(model, XC[idx] if XC is not None else None, XB[idx] if XB is not None else None, Y[idx], t, eps, ub, amp)
            loss = lc + lb
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), c.grad_clip)
            opt.step()
            sched.step()
            with torch.no_grad():
                torch._foreach_mul_(ema_p, c.ema_decay)
                torch._foreach_add_(ema_p, mod_p, alpha=1.0 - c.ema_decay)
            run += float(loss.detach())
            if step % c.log_every == 0:
                self.loss_log.append({"step": step, "train_loss": run / c.log_every})
                run = 0.0
            if hold.any() and step % c.eval_every == 0:
                with torch.no_grad():
                    tot = 0.0
                    for r in range(reps):
                        a, b = self._loss_parts(self.ema, HC, HB, HY, th[r], eh[r], uh[r], amp)
                        tot += float(a + b)
                    hl = tot / reps
                self.loss_log.append({"step": step, "holdout_loss": hl})
                if not (hl >= self.best["holdout_loss"]):
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
        model.eval()

    # ── 진단: 떼어 둔 자료의 손실과 자명한 기준선 ──
    @torch.no_grad()
    def denoise_loss(self, X_cont, X_bin, y, seed: int = 1, n_draws: int = 8) -> dict:
        dev = self.device
        XC = torch.from_numpy(self._std(X_cont)).to(dev) if self.d_cont else None
        XB = torch.from_numpy(np.asarray(X_bin, dtype=np.float32)).to(dev) if self.d_bin else None
        Y = torch.from_numpy(np.asarray(y).astype(np.int64)).to(dev)
        g = torch.Generator(device=dev)
        g.manual_seed(seed)
        out = {"cont_model": 0.0, "cont_zero": 0.0, "cont_gauss_indep": 0.0, "bin_model": 0.0, "bin_copy": 0.0, "bin_marginal": 0.0}
        nc = nb = 0
        for _ in range(n_draws):
            t = torch.randint(0, self.cfg.T, (len(Y),), device=dev, generator=g)
            eps = torch.randn((len(Y), max(self.d_cont, 1)), device=dev, generator=g)[:, : self.d_cont]
            ub = torch.rand((len(Y), max(self.d_bin, 1)), device=dev, generator=g)[:, : self.d_bin]
            xc_t, xb_t = self._noisy(XC, XB, t, eps, ub)
            e_hat, logit = self.ema(xc_t if xc_t is not None else torch.zeros((len(Y), 0), device=dev),
                                    xb_t if xb_t is not None else torch.zeros((len(Y), 0), device=dev), t, Y,
                                    torch.sqrt(1.0 - self.alpha_bars[t]).view(-1, 1))
            if self.d_cont:
                skip = torch.sqrt(1.0 - self.alpha_bars[t]).view(-1, 1)
                out["cont_model"] += float(((e_hat.float() - eps) ** 2).sum())
                out["cont_zero"] += float((eps ** 2).sum())
                out["cont_gauss_indep"] += float(((skip * xc_t - eps) ** 2).sum())
                nc += eps.numel()
            if self.d_bin:
                base = self.ema.base_logit[Y]
                copy_logit = torch.logit(xb_t.clamp(0.01, 0.99))  # x_0 = x_t 라고 보는 예측 (조금 부드럽게)
                T = self.cfg.T
                out["bin_model"] += T * float(self._bin_kl(xb_t, XB, t, logit.float()).sum())
                out["bin_copy"] += T * float(self._bin_kl(xb_t, XB, t, copy_logit).sum())
                out["bin_marginal"] += T * float(self._bin_kl(xb_t, XB, t, base).sum())
                nb += xb_t.numel()
        for k in list(out):
            out[k] = out[k] / (nc if k.startswith("cont") else nb) if (nc if k.startswith("cont") else nb) else float("nan")
        out["total_model"] = (out["cont_model"] if self.d_cont else 0.0) + (out["bin_model"] if self.d_bin else 0.0)
        out["total_baseline"] = (out["cont_gauss_indep"] if self.d_cont else 0.0) + (min(out["bin_copy"], out["bin_marginal"]) if self.d_bin else 0.0)
        return out

    # ── 추출 ──
    @torch.no_grad()
    def sample(self, n: int, y_label: int = 1):
        dev = self.device
        g = torch.Generator(device=dev)
        g.manual_seed(self.seed + 7919)
        amp = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if (self.cfg.use_bf16 and dev.type == "cuda") else None
        outc, outb = [], []
        for start in range(0, n, self.cfg.sample_batch):
            b = min(self.cfg.sample_batch, n - start)
            xc = torch.randn((b, max(self.d_cont, 1)), device=dev, generator=g)[:, : self.d_cont]
            xb = (torch.rand((b, max(self.d_bin, 1)), device=dev, generator=g)[:, : self.d_bin] < 0.5).float()
            y = torch.full((b,), int(y_label), device=dev, dtype=torch.long)
            for ti in range(self.cfg.T - 1, -1, -1):
                t = torch.full((b,), ti, device=dev, dtype=torch.long)
                eps_hat, logit = self._pred(self.ema, xc, xb, t, y, amp)
                if self.d_cont:
                    a, ab, bt = self.alphas[ti], self.alpha_bars[ti], self.betas[ti]
                    mean = (xc - bt / torch.sqrt(1.0 - ab) * eps_hat) / torch.sqrt(a)
                    xc = mean + torch.sqrt(self.post_var[ti]) * torch.randn(xc.shape, device=dev, generator=g) if ti > 0 else mean
                if self.d_bin:
                    p1 = self._bin_post(xb, t.view(-1), torch.sigmoid(logit))
                    xb = (torch.rand(xb.shape, device=dev, generator=g) < p1).float()
            outc.append(xc.float().cpu().numpy())
            outb.append(xb.cpu().numpy().astype(np.int8))
        C = np.vstack(outc) if self.d_cont else np.zeros((n, 0), dtype=np.float32)
        if self.mu is not None:
            C = C * self.sd + self.mu
        return C, np.vstack(outb) if self.d_bin else np.zeros((n, 0), dtype=np.int8)
