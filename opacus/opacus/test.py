#!/usr/bin/env python3
import math, argparse, torch
import torch.nn as nn
import torch.optim as optim
from opacus.grad_sample import GradSampleModule

from opacus.optimizers.autoclipoptimizer import AutoSFixedDPOptimizer  # 替换成你的路径


# ---------- 小模型与数据 ----------
class TinyNet(nn.Module):
    def __init__(self, d_in=32, d_h=64, d_out=10):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, d_h), nn.ReLU(), nn.Linear(d_h, d_out))

    def forward(self, x):
        return self.net(x)


def make_batch(B=128, d_in=32, num_classes=10, device="cuda"):
    x = torch.randn(B, d_in, device=device)
    y = torch.randint(0, num_classes, (B,), device=device)
    return x, y


# ---------- A. 纯噪声校准 ----------
def check_pure_noise(dp_opt, model, B, loss_reduction):
    model.zero_grad(set_to_none=True)
    dp_opt.zero_grad()
    x, _ = make_batch(B, device=next(model.parameters()).device)
    y = model(x)
    (y.sum() * 0.0).backward()  # 产生 grad_sample 但总梯度为0

    dp_opt.clip_and_accumulate()
    dp_opt.add_noise()  # 不要先 scale_grad

    v_grad, v_sum = [], []
    for p in dp_opt.params:
        g = getattr(p, "grad", None)
        sg = getattr(p, "summed_grad", None)
        if sg is not None and sg.numel() > 0 and torch.isfinite(sg).all():
            v_sum.append(sg.detach().flatten())
        if g is not None and g.numel() > 0 and torch.isfinite(g).all():
            v_grad.append(g.detach().flatten())

    sigma = float(dp_opt.noise_multiplier)
    res = {}
    if v_sum:
        std_sum = torch.cat(v_sum).std(unbiased=False).item()
        res["summed_grad_std"] = std_sum
        res["target_sum"] = sigma
    if v_grad:
        std_grad = torch.cat(v_grad).std(unbiased=False).item()
        res["grad_std"] = std_grad
        if loss_reduction == "mean":
            EB = (
                getattr(dp_opt, "actual_batch_size", None)
                or getattr(dp_opt, "expected_batch_size", None)
                or B
            )
            res["target_mean"] = sigma / float(EB)
    return res


# ---------- B. 单步差分（last step） ----------
def check_last_step(dp_opt, model, B):
    model.zero_grad(set_to_none=True)
    x, y = make_batch(B, device=next(model.parameters()).device)
    loss = nn.CrossEntropyLoss(reduction="mean")(model(x), y)
    loss.backward()
    dp_opt.clip_and_accumulate()
    dp_opt.add_noise()
    return dp_opt.print_last_noise_std()  # 你的实现里已打印并返回数值


# ---------- C. 多步累计 ----------
def check_multi_steps(dp_opt, model, B, steps=10, loss_reduction="mean"):
    buf = []
    for _ in range(steps):
        model.zero_grad(set_to_none=True)
        dp_opt.zero_grad()
        x, _ = make_batch(B, device=next(model.parameters()).device)
        (model(x).sum() * 0.0).backward()  # 纯噪声
        dp_opt.clip_and_accumulate()
        dp_opt.add_noise()
        # 抓取这一步的噪声：用 grad 与 summed 的差分策略
        # 这里直接用 p.summed_grad 视作噪声（因为纯噪声场景下信号为0）
        for p in dp_opt.params:
            sg = getattr(p, "summed_grad", None)
            if sg is not None and sg.numel() > 0:
                buf.append(sg.detach().flatten())
    if not buf:
        return None
    allv = torch.cat(buf)
    est = allv.std(unbiased=False).item()
    sigma = float(dp_opt.noise_multiplier)
    if loss_reduction == "mean":
        EB = (
            getattr(dp_opt, "actual_batch_size", None)
            or getattr(dp_opt, "expected_batch_size", None)
            or B
        )
        target = sigma / float(EB)
    else:
        target = sigma
    return est, target


# ---------- D. SNR 粗测 ----------
def check_snr(dp_opt, model, B, loss_reduction="mean"):
    # 先来一次有信号的反传
    model.zero_grad(set_to_none=True)
    x, y = make_batch(B, device=next(model.parameters()).device)
    loss = nn.CrossEntropyLoss()(model(x), y)
    loss.backward()
    dp_opt.clip_and_accumulate()

    # 信号范数（加噪前）
    sig = []
    for p in dp_opt.params:
        sg = getattr(p, "summed_grad", None)
        if sg is not None and sg.numel() > 0:
            sig.append(sg.detach().flatten())
    sig_norm = torch.cat(sig).norm().item() if sig else float("nan")

    # 加噪后
    dp_opt.add_noise()
    noised = []
    for p in dp_opt.params:
        g = getattr(p, "grad", None)
        if g is not None and g.numel() > 0:
            noised.append(g.detach().flatten())
    noised_norm = torch.cat(noised).norm().item() if noised else float("nan")

    # 理论噪声规模
    sigma = float(dp_opt.noise_multiplier)
    # 取“落在 grad 上的维度”估计 d
    d = sum(p.numel() for p in model.parameters() if p.grad is not None)
    EB = (
        getattr(dp_opt, "actual_batch_size", None)
        or getattr(dp_opt, "expected_batch_size", None)
        or B
    )
    theo = (
        sigma * math.sqrt(d) if loss_reduction == "sum" else (sigma / EB) * math.sqrt(d)
    )
    return {
        "signal_norm": sig_norm,
        "noised_norm": noised_norm,
        "theory_noise_L2": theo,
    }


def _ensure_fake_grad_samples(dp_opt, B: int):
    for p in dp_opt.params:
        if getattr(p, "grad_sample", None) is None:
            # 用全零假样本，形状 [B, *p.shape]
            fake = torch.zeros(
                (B, *p.shape), device=p.device, dtype=p.dtype, requires_grad=False
            )
            p.grad_sample = fake


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--sigma", type=float, default=36.0)  # 噪声倍数
    ap.add_argument("--loss_reduction", choices=["mean", "sum"], default="mean")
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    model = TinyNet().to(args.device)
    model = GradSampleModule(model, batch_first=True)

    base_opt = optim.SGD(model.parameters(), lr=1e-3)
    dp_opt = AutoSFixedDPOptimizer(
        base_opt,
        noise_multiplier=args.sigma,
        max_grad_norm=1.0,
        expected_batch_size=args.batch,
        loss_reduction=args.loss_reduction,
        optim_args={"stability_const": 1e-2},
    )

    print("\n[A] pure-noise sanity:")
    dp_opt.zero_grad(set_to_none=True)
    resA = check_pure_noise(dp_opt, model, args.batch, args.loss_reduction)
    print(resA)

    print("\n[B] last-step delta:")
    dp_opt.zero_grad(set_to_none=True)
    valB = check_last_step(dp_opt, model, args.batch)
    print("last-step measured:", valB)

    print("\n[C] multi-step aggregate:")
    dp_opt.zero_grad(set_to_none=True)
    resC = check_multi_steps(dp_opt, model, args.batch, args.steps, args.loss_reduction)
    print("multi-step (std, target):", resC)

    print("\n[D] SNR check:")
    dp_opt.zero_grad(set_to_none=True)
    resD = check_snr(dp_opt, model, args.batch, args.loss_reduction)
    print(resD)


if __name__ == "__main__":
    main()
