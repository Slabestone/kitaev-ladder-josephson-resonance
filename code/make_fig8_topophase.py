#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fig.8 (topophase.png) 重写——孤立 superconducting Kitaev ladder 的拓扑相图。

承融 2026-08-31 方案：图本质上只由两条拓扑条件决定
  |μ_S + t_⊥| < 2t   (bonding band)
  |μ_S - t_⊥| < 2t   (antibonding band)
φ=0: winding number w = 满足的 sector 数（0/1/2）
0<φ<π: chiral 破缺 → ν = w mod 2（0/1）

横轴统一 μ_S/t（孤立 ladder 的 bulk 拓扑，只出现 μ_S 不出现 μ_M）。
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

# ============================================================
# Parameters
# ============================================================

t = 1.0

muS_min, muS_max = 0.0, 3.0
tp_min, tp_max = 0.0, 3.0

N_mu = 801
N_tp = 801

muS = np.linspace(muS_min, muS_max, N_mu)
tperp = np.linspace(tp_min, tp_max, N_tp)

MU, TP = np.meshgrid(muS, tperp)

# ============================================================
# Topological classification
# ============================================================

# Two bonding / antibonding sectors of the isolated ladder
topo_plus = np.abs(MU + TP) < 2.0 * t
topo_minus = np.abs(MU - TP) < 2.0 * t

# phi = 0: BDI winding number = number of topological sectors
w = topo_plus.astype(int) + topo_minus.astype(int)

# 0 < phi < pi: chiral symmetry broken, Z2 parity
nu = w % 2

# ============================================================
# Color maps
# ============================================================

cmap_w = ListedColormap(["white", "#1786b8", "#ffe000"])
norm_w = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap_w.N)

cmap_nu = ListedColormap(["white", "#1786b8"])
norm_nu = BoundaryNorm([-0.5, 0.5, 1.5], cmap_nu.N)

# ============================================================
# Figure
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.7))
fig.subplots_adjust(left=0.10, right=0.94, bottom=0.18, top=0.84, wspace=0.38)

# --- Panel (a): phi = 0 ---
ax = axes[0]
im0 = ax.pcolormesh(MU / t, TP / t, w, cmap=cmap_w, norm=norm_w, shading="nearest")
ax.set_xlim(muS_min / t, muS_max / t)
ax.set_ylim(tp_min / t, tp_max / t)
ax.set_xlabel(r"$\mu_S/t$", fontsize=15)
ax.set_ylabel(r"$t_\perp/t$", fontsize=15)
ax.set_title(r"(a) $\phi=0$", fontsize=16, pad=10)
ax.set_xticks([0, 1, 2, 3])
ax.set_yticks([0, 1, 2, 3])
ax.tick_params(direction="out", length=4, width=1, labelsize=12)
cbar0 = fig.colorbar(im0, ax=ax, ticks=[0, 1, 2], fraction=0.046, pad=0.055)
cbar0.ax.set_ylabel(r"$w$", rotation=0, labelpad=10, fontsize=14)
cbar0.ax.tick_params(labelsize=11)

# --- Panel (b): 0 < phi < pi ---
ax = axes[1]
im1 = ax.pcolormesh(MU / t, TP / t, nu, cmap=cmap_nu, norm=norm_nu, shading="nearest")
ax.set_xlim(muS_min / t, muS_max / t)
ax.set_ylim(tp_min / t, tp_max / t)
ax.set_xlabel(r"$\mu_S/t$", fontsize=15)
ax.set_ylabel(r"$t_\perp/t$", fontsize=15)
ax.set_title(r"(b) $0<\phi<\pi$", fontsize=16, pad=10)
ax.set_xticks([0, 1, 2, 3])
ax.set_yticks([0, 1, 2, 3])
ax.tick_params(direction="out", length=4, width=1, labelsize=12)
cbar1 = fig.colorbar(im1, ax=ax, ticks=[0, 1], fraction=0.046, pad=0.055)
cbar1.ax.set_ylabel(r"$\nu$", rotation=0, labelpad=10, fontsize=14)
cbar1.ax.tick_params(labelsize=11)

# ============================================================
# Output（文件名匹配 tex 引用 topophase.png）
# ============================================================

import pathlib
out_dir = pathlib.Path(__file__).resolve().parent.parent / "generated"
out_dir.mkdir(exist_ok=True)

plt.savefig(out_dir / "topophase.png", dpi=400, bbox_inches="tight")
plt.savefig(out_dir / "topophase.pdf", bbox_inches="tight")
print("saved: topophase.png (Fig.8 重写——μ_S/t 横轴，w/ν 分类)")
