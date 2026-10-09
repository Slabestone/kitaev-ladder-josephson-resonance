#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fig.6 (EJ_off_vs_res.png) 生成脚本——从 20260824 会话重建（2026-08-31 乐祈）。

原脚本来源：Reasonix 会话 20260824-185201（08-26 重画版，大字体+5ticks+bbox_inches）。
本版修改：
  - ylabel: E_J(θ)-E_J(0) → E_GS(θ)-E_GS(0)（notation audit，M 体系统一）
  - 物理量/参数不变：off-resonant μ_M=0.3 vs resonant μ_M=0，L_M=5, t⊥=0.1, φ=π/3, Γ=0.1
"""
import numpy as np
from scipy.linalg import eigh
from general_kitaev_sns import KitaevSNS
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

plt.rcParams.update({'font.size': 14, 'axes.labelsize': 15, 'axes.titlesize': 14,
                     'legend.fontsize': 12, 'xtick.labelsize': 12, 'ytick.labelsize': 12})

ths = np.linspace(-np.pi, np.pi, 401)
phi = np.pi / 3


def EGS_of(m):
    return np.array([m.ground_energy(th) for th in ths])


# 严格可比较：同 (L_M, t⊥, φ, Γ)，只改 detuning μ_M
m_off = KitaevSNS(L_N=5, mu_N=0.3, mu_SC=0.0, t_perp=0.1, Gamma=0.1, phi=phi, Delta=0.6, t=1.0)
m_res = KitaevSNS(L_N=5, mu_N=0.0, mu_SC=0.0, t_perp=0.1, Gamma=0.1, phi=phi, Delta=0.6, t=1.0)
E_off = EGS_of(m_off) - EGS_of(m_off)[0]
E_res = EGS_of(m_res) - EGS_of(m_res)[0]

fig, ax = plt.subplots(figsize=(8, 5.2))
ax.plot(ths / np.pi, E_off, 'b-', lw=2, label='off-resonant ($\\mu_{\\rm M}=0.3$): $\\pi$-type')
ax.plot(ths / np.pi, E_res, 'r-', lw=2, label='resonant ($\\mu_{\\rm M}=0$): $0$-type')
ax.axvline(0, color='gray', lw=0.8)
ax.axvline(1, color='gray', lw=0.8, ls='--')
ax.set_xlabel('$\\theta/\\pi$')
ax.set_ylabel('$E_{\\rm GS}(\\theta)-E_{\\rm GS}(0)$')
ax.yaxis.set_major_locator(MaxNLocator(5))
ax.xaxis.set_major_locator(MaxNLocator(5))
ax.legend(fontsize=12)
plt.tight_layout()
plt.savefig('EJ_off_vs_res.png', dpi=200, bbox_inches='tight')
print("Fig.6 重画（E_GS 记号 + μ_M 标签）")
