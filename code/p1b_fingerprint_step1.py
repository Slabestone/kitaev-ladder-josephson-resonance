#!/usr/bin/env python3
"""Step 1：复现 Paper 1 两个 manuscript fingerprint CPR 点（回响规格 2026-08-20）
指纹点:
  P1: (L_M, μ, t⊥) = (4, 0, 0.1) → Fig.3(c) π-type CPR
  P2: (L_M, μ, t⊥) = (5, 1, 0.1) → Fig.4(c) predominantly 0-type CPR
正文参数: Δ=0.6t, Γ=0.1t, φ=π/3, t=1.0
输出: 完整 J(θ)，积分得 E_J(θ)，报告 θ_min、第一谐波 I1、NaN/Inf、周期性、分类 PASS/FAIL
"""
import sys

import numpy as np
from p1b_sns import build_sns, josephson_current

T = 1.0
DELTA = 0.6 * T
PHI = np.pi / 3.0
NL = 40          # lead 长度（半无限近似）
N_THETA = 200    # θ grid


def analyze_fingerprint(name, LM, mu, tperp):
    print("=" * 70)
    print(f"指纹点 {name}: (L_M, μ, t⊥) = ({LM}, {mu}, {tperp})")
    print("=" * 70)

    # 构造 H(θ) 的函数
    # 正文：S-N 接口 hopping = Γ = 0.1t（p1b_sns.build_sns 的 t_if 参数）
    GAMMA = 0.1 * T

    def H_fn(theta):
        return build_sns(LM, NL, mu, T, DELTA, tperp, PHI, theta, t_if=GAMMA)

    # 先验证 Hermiticity + PHS（数值健康检查）
    H0 = H_fn(np.pi / 3)
    dim = H0.shape[0]
    herm = np.max(np.abs(H0 - H0.conj().T))
    # Particle-hole exchange is a permutation; avoid dense BLAS for this check.
    exchange = np.r_[np.arange(dim // 2, dim), np.arange(dim // 2)]
    phs = np.max(np.abs(H0.conj()[exchange][:, exchange] + H0))
    if herm > 1e-12 or phs > 1e-12:
        raise RuntimeError("Hermiticity or particle-hole symmetry check failed")
    print(f"  dim={dim} | Hermiticity err={herm:.2e} | PHS err={phs:.2e}")

    thetas = np.linspace(0, 2 * np.pi, N_THETA)
    J, Egs = josephson_current(H_fn, thetas, dim // 2)

    # 报告
    finite = np.all(np.isfinite(J)) and np.all(np.isfinite(Egs))
    nan_inf = np.sum(~np.isfinite(J)) + np.sum(~np.isfinite(Egs))
    print(f"  NaN/Inf: {nan_inf}（{'✅ 无' if finite else '⚠️ 有'}）")

    # E_J(θ)：用 Egs 的过零点积分定义（J=+dE/dθ，E 是基态能）
    # 这里 Egs 就是 E_J(θ) 的载体（含 θ 无关常数，先中心化）
    EJ = Egs - np.mean(Egs)
    # θ_min: E_J 最小值位置
    i_min = np.argmin(EJ)
    theta_min = thetas[i_min] / np.pi
    print(f"  E_J 最小值位置 θ_min = {theta_min:.3f} π")

    # 第一谐波 I1：J 的 sin(θ) 傅里叶系数（J 应为奇函数）
    J_odd = (J - J[::-1]) / 2  # 提取奇部
    I1 = 2 * np.trapezoid(J_odd * np.sin(thetas), thetas) / np.pi
    I2 = 2 * np.trapezoid(J_odd * np.sin(2 * thetas), thetas) / np.pi
    print(f"  第一谐波 I1 = {I1:.4e} | 第二谐波 I2 = {I2:.4e}")

    # 周期性：J 的周期
    # 找 J 过零位置（从正到负 = 临界方向）
    signs = np.sign(J)
    crossings = np.where(np.diff(signs) != 0)[0]
    print(f"  过零次数: {len(crossings)}")

    # 分类：π-type vs 0-type
    # π-junction: E_J 最小值在 θ=π（J 过零在 π 附近且 E 最小在 π）
    # 0-junction: E_J 最小值在 θ=0/2π
    d_min_pi = abs(theta_min - 1.0) if abs(theta_min - 1.0) <= abs(theta_min - 3.0) else abs(theta_min - 3.0)
    d_min_0 = min(abs(theta_min), abs(theta_min - 2.0))
    if d_min_pi < d_min_0:
        classification = "π-type"
    else:
        classification = "0-type"
    print(f"  θ_min 距 π: {d_min_pi:.3f} | 距 0: {d_min_0:.3f} → 分类: {classification}")

    # J(θ) 头部输出
    print("  J(θ) 采样（θ/π: J）:")
    for i in range(0, N_THETA, 25):
        print(f"    {thetas[i]/np.pi:+.2f}: {J[i]:+.4e}")
    return J, EJ, classification, theta_min, I1, I2, finite


if __name__ == "__main__":
    results = {}
    # 指纹点 1: (4, 0, 0.1) → 期望 π-type
    J1, EJ1, c1, tm1, I1_1, I2_1, f1 = analyze_fingerprint("P1", LM=4, mu=0.0, tperp=0.1)
    results['P1'] = (c1, tm1, I1_1, I2_1, f1)

    print()
    # 指纹点 2: (5, 1, 0.1) → 期望 0-type
    J2, EJ2, c2, tm2, I1_2, I2_2, f2 = analyze_fingerprint("P2", LM=5, mu=1.0, tperp=0.1)
    results['P2'] = (c2, tm2, I1_2, I2_2, f2)

    print()
    print("=" * 70)
    print("最终判定")
    print("=" * 70)
    c1, tm1, I1_1, I2_1, f1 = results['P1']
    c2, tm2, I1_2, I2_2, f2 = results['P2']
    pass1 = (c1 == "π-type") and f1
    pass2 = (c2 == "0-type") and f2
    print(f"  P1 (L_M=4,μ=0) : {c1}, θ_min={tm1:.3f}π, I1={I1_1:.3e} → {'✅ PASS' if pass1 else '❌ FAIL'}（期望 π-type）")
    print(f"  P2 (L_M=5,μ=1) : {c2}, θ_min={tm2:.3f}π, I1={I1_2:.3e} → {'✅ PASS' if pass2 else '❌ FAIL'}（期望 0-type）")
    print(f"\n  Phase 0 封存判定: {'✅ 两个指纹点全部 PASS' if pass1 and pass2 else '⚠️ 未全部通过，需核查'}")

    if not (pass1 and pass2):
        raise SystemExit(1)
