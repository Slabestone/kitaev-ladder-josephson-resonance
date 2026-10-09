#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fig. 5: resonance diagnostics (v3，回响 2026-08-30 版式修复版).

Main logic
----------
(a) Normal-level alignment with the end-mode energy.
(b) J1(mu_N) at phi = pi/3.
(c) Continuation of J1(mu_N) with inter-leg phase phi.
    Inset: fourth-order residual at the analytic resonant anchor
    phi = pi, mu_N = 0, in the noncrossing regime.

v3 相对 v2 的修改（回响 00:58 审查）：
1. inset 从右下角 [0.55,0.12] 移到两 resonance 之间 [0.31,0.54]——不再压住 q=4 peak
2. inset ticks/labels 极简化：只留 0.03/0.06 两个 x tick，关 minor labels，
   "slope=4.00" 改 inset 内部轻量 "p=4.00"
3. panel (a) alignment 点改黑点（ko）——颜色=能级 q，黑点=spectral alignment
4. panel (c) 局部加密：81 global + q=3 区间(-0.12,0.08)31 点 + q=4 区间(0.88,1.08)31 点
5. 纵轴统一 J1/t（J_n 是 energy harmonic，用 t 作单位）
"""

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, NullFormatter

from general_kitaev_sns import KitaevSNS


# ============================================================
# Parameters
# ============================================================

t = 1.0
t_perp = 0.1 * t
Gamma = 0.1 * t
Delta = 0.6 * t
L_N = 5

phi = np.pi / 3

# Numerically extracted end-mode energy at phi = pi/3
E_end = 0.049 * t

# 401 points are retained for the production scan.
ths = np.linspace(-np.pi, np.pi, 401)


# ============================================================
# First Josephson harmonic
# ============================================================

def J1_of(model):
    """
    J1 = -(1/pi) int E_GS(theta) cos(theta) dtheta.

    J1 > 0 : 0-type
    J1 < 0 : pi-type
    """
    E = np.array([
        model.ground_energy(th)
        for th in ths
    ])

    return -(1.0 / np.pi) * np.trapezoid(
        E * np.cos(ths),
        ths
    )


# ============================================================
# Cached scan helper
#
# J1 计算是整张图最贵的部分（每点 ~0.7s，全图 ~8 min）。
# 结果按 (参数, mu 网格, theta 网格) 哈希缓存到 data/fig5_J1_cache.npz。
# 改版式/配色时重跑只需读缓存，秒级完成；只有物理参数或网格变化才重算。
# ============================================================

import hashlib
import json
import os

_CACHE_DIR = Path(__file__).resolve().parent.parent / "data"
_CACHE_PATH = _CACHE_DIR / "fig5_J1_cache.npz"


def _scan_key(mus, phi_val, gamma_val, theta_n=401, delta=0.6, tperp=0.1):
    """计算参数的稳定哈希（不含 J1 结果）。"""
    payload = json.dumps({
        "L_N": 5, "t": 1.0, "Delta": delta, "t_perp": tperp,
        "mu_min": float(mus[0]), "mu_max": float(mus[-1]),
        "mu_n": int(len(mus)),
        "phi": float(phi_val), "Gamma": float(gamma_val),
        "theta_n": theta_n,
    }, sort_keys=True)
    return hashlib.sha1(payload.encode()).hexdigest()[:16]


def compute_J1_scan(mus, phi_val, gamma_val, use_cache=True):
    """
    计算 J1(mu) for a fixed (phi, Gamma) over the mu grid.
    带 npz 缓存：同参数二次调用直接读缓存。
    """
    key = _scan_key(mus, phi_val, gamma_val)
    mu_key = f"mus_{key}"
    j1_key = f"J1_{key}"

    if use_cache and _CACHE_PATH.exists():
        try:
            with np.load(_CACHE_PATH, allow_pickle=False) as data:
                if mu_key in data and j1_key in data:
                    cached_mus = data[mu_key]
                    cached_j1 = data[j1_key]
                    if (len(cached_mus) == len(mus)
                            and np.allclose(cached_mus, mus)):
                        print(f"  [cache] phi/pi={phi_val/np.pi:.3f} "
                              f"Gamma={gamma_val:.3f} ({len(mus)} pts) — hit")
                        return cached_j1
        except Exception as e:
            print(f"  [cache] read failed ({e}); recomputing")

    print(f"  [compute] phi/pi={phi_val/np.pi:.3f} "
          f"Gamma={gamma_val:.3f} ({len(mus)} pts)")
    j1 = np.array([
        J1_of(
            KitaevSNS(
                L_N=L_N,
                mu_N=mu,
                mu_SC=0.0,
                t_perp=t_perp,
                Gamma=gamma_val,
                phi=phi_val,
                Delta=Delta,
                t=t,
            )
        )
        for mu in mus
    ])

    if use_cache:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        # 合并写入（保留其他 key）
        store = {}
        if _CACHE_PATH.exists():
            try:
                with np.load(_CACHE_PATH, allow_pickle=False) as data:
                    for k in data.files:
                        store[k] = data[k]
            except Exception:
                store = {}
        store[mu_key] = np.asarray(mus)
        store[j1_key] = j1
        np.savez(_CACHE_PATH, **store)
        print(f"  [cache] saved {_CACHE_PATH}")

    return j1


# ============================================================
# Panel (a): normal-level alignment
# ============================================================

mus_a = np.linspace(-0.5, 1.5, 161)

# L_N = 5:
# epsilon_q^M = mu_N + t_perp + 2t cos(pi q / 6)
epsN_q3 = (
    mus_a
    + t_perp
    + 2.0 * t * np.cos(np.pi * 3 / 6)
)

epsN_q4 = (
    mus_a
    + t_perp
    + 2.0 * t * np.cos(np.pi * 4 / 6)
)

# Actual dressed alignment epsilon_q^M = E_end
mu_q3 = (
    E_end
    - t_perp
    - 2.0 * t * np.cos(np.pi * 3 / 6)
)

mu_q4 = (
    E_end
    - t_perp
    - 2.0 * t * np.cos(np.pi * 4 / 6)
)

print(
    f"alignments: "
    f"q=3 at mu={mu_q3:.3f}, "
    f"q=4 at mu={mu_q4:.3f}"
)


# ============================================================
# Panel (b): dense mu scan at phi = pi/3
# ============================================================

J1s = compute_J1_scan(mus_a, phi, Gamma)


# ============================================================
# Panel (c): phase continuation
#
# Coarse global grid + local refinement near the two resonance
# windows.  This gives smoother peaks without paying for a
# uniformly dense four-phi scan.
# ============================================================

mus_c = np.unique(
    np.concatenate([
        # Global background
        np.linspace(-0.5, 1.5, 81),

        # q = 3 resonance family
        np.linspace(-0.12, 0.08, 31),

        # q = 4 resonance family
        np.linspace(0.88, 1.08, 31),
    ])
)

phis = [
    np.pi,
    2 * np.pi / 3,
    np.pi / 2,
    np.pi / 3,
]

phi_labels = {
    np.pi:
        r"$\phi=\pi$",

    2 * np.pi / 3:
        r"$\phi=2\pi/3$",

    np.pi / 2:
        r"$\phi=\pi/2$",

    np.pi / 3:
        r"$\phi=\pi/3$",
}

J1_by_phi = {}

for pv in phis:

    print(
        f"calculating panel (c): "
        f"phi/pi = {pv / np.pi:.3f}, "
        f"N_mu = {len(mus_c)}"
    )

    J1_by_phi[pv] = compute_J1_scan(mus_c, pv, Gamma)


# ============================================================
# Gamma^4 inset
#
# Analytic anchor:
# phi = pi, mu_N = 0
#
# Restrict all points to
#     t_perp > sqrt(2) Gamma
# so that the low-energy sector remains noncrossing.
# ============================================================

Gamma_max = t_perp / np.sqrt(2.0)

Gs = np.linspace(
    0.025,
    0.065,
    6
)

assert np.all(Gs < Gamma_max), (
    "Gamma points leave the noncrossing regime: "
    f"max Gamma = {Gs.max():.4f}, "
    f"t_perp/sqrt(2) = {Gamma_max:.4f}"
)

# inset 需要沿 Γ 扫描，mu 固定 0：逐 Γ 算（每个 Γ 一个单点 scan）
J1_G_signed = np.array([
    compute_J1_scan(np.array([0.0]), np.pi, G)[0]
    for G in Gs
])

print(
    "Gamma^4 signed J1 "
    "(phi=pi, mu_N=0):"
)

for G, J in zip(Gs, J1_G_signed):
    print(
        f"  Gamma={G:.4f}, "
        f"J1={J:.6e}"
    )

# The resonant residual should remain 0-type.
assert np.all(J1_G_signed > 0), (
    "Inset contains a J1 sign change."
)

J1_G = np.abs(J1_G_signed)

# Log-log exponent
slope = np.polyfit(
    np.log(Gs),
    np.log(J1_G),
    1
)[0]

print(
    f"Gamma^4 slope = {slope:.3f}"
)


# ============================================================
# Plot
# ============================================================

fig, axes = plt.subplots(
    1,
    3,
    figsize=(13.5, 4.15),
)

fig.subplots_adjust(
    left=0.065,
    right=0.985,
    bottom=0.15,
    top=0.90,
    wspace=0.34,
)


# ============================================================
# Panel (a)
# ============================================================

ax = axes[0]

ax.plot(
    mus_a,
    epsN_q3,
    lw=2.0,
    label=r"$\epsilon_3^{\mathrm{M}}$ (bonding)",
)

ax.plot(
    mus_a,
    epsN_q4,
    lw=2.0,
    label=r"$\epsilon_4^{\mathrm{M}}$ (bonding)",
)

ax.axhline(
    E_end,
    color="0.35",
    ls=":",
    lw=1.2,
    label=r"$E_{\rm end}$ ($\phi=\pi/3$)",
)

# Use black markers for the physical alignment itself.
# Curve color identifies q; black dot identifies "alignment".
for mq in [mu_q3, mu_q4]:
    ax.plot(
        mq,
        E_end,
        "ko",
        ms=5.5,
        zorder=6,
    )

ax.annotate(
    r"$\mu_{\mathrm{M}}\simeq-0.05$",
    xy=(mu_q3, E_end),
    xytext=(-0.42, 0.105),
    arrowprops=dict(
        arrowstyle="->",
        lw=0.8,
    ),
    fontsize=8,
)

ax.annotate(
    r"$\mu_{\mathrm{M}}\simeq0.95$",
    xy=(mu_q4, E_end),
    xytext=(0.60, 0.135),
    arrowprops=dict(
        arrowstyle="->",
        lw=0.8,
    ),
    fontsize=8,
)

ax.set_xlim(
    -0.5,
    1.5,
)

ax.set_ylim(
    -0.15,
    0.60,
)

ax.set_xlabel(
    r"$\mu_{\mathrm{M}}/t$"
)

ax.set_ylabel(
    r"energy / $t$"
)

ax.set_title(
    "(a) Level alignment",
    fontsize=10,
)

ax.legend(
    fontsize=7.3,
    loc="lower left",
    frameon=True,
)


# ============================================================
# Panel (b)
# ============================================================

ax = axes[1]

ax.plot(
    mus_a,
    J1s / t,
    "k-",
    lw=1.6,
)

ax.axhline(
    0,
    color="0.65",
    lw=0.7,
    zorder=0,
)

# Actual alignment positions (grey: vertical line = spectral alignment,
# colour-coded only in panel (a) where q matters)
ax.axvline(
    mu_q3,
    color="0.4",
    ls="--",
    lw=1.0,
)

ax.axvline(
    mu_q4,
    color="0.4",
    ls="--",
    lw=1.0,
)

ax.set_xlim(
    -0.5,
    1.5,
)

ax.set_xlabel(
    r"$\mu_{\mathrm{M}}/t$"
)

ax.set_ylabel(
    r"$J_1/t$"
)

ax.set_title(
    r"(b) $J_1(\mu_{\mathrm{M}})$ at $\phi=\pi/3$",
    fontsize=10,
)


# ============================================================
# Panel (c)
# ============================================================

ax = axes[2]

for pv in phis:

    ax.plot(
        mus_c,
        J1_by_phi[pv] / t,
        lw=1.5,
        zorder=2,
        label=phi_labels[pv],
    )

ax.axhline(
    0,
    color="0.65",
    lw=0.7,
    zorder=0,
)

ax.set_xlim(
    -0.5,
    1.5,
)

ax.set_xlabel(
    r"$\mu_{\mathrm{M}}/t$"
)

ax.set_ylabel(
    r"$J_1/t$"
)

ax.set_title(
    r"(c) $J_1(\mu_{\mathrm{M}})$ vs $\phi$",
    fontsize=10,
)

# Keep the main legend in the upper-right corner.
ax.legend(
    fontsize=7.2,
    loc="upper right",
    frameon=True,
)


# ============================================================
# Inset
#
# Put it in the quiet region BETWEEN the q=3 and q=4
# resonances, rather than over the q=4 peak.
#
# Axes coordinates:
# left, bottom, width, height
# ============================================================

ax_ins = axes[2].inset_axes(
    [
        0.40,   # left
        0.57,   # bottom
        0.29,   # width
        0.31,   # height
    ]
)

ax_ins.loglog(
    Gs / t,
    J1_G / t,
    "ko-",
    ms=3.7,
    lw=1.1,
)

# Minimal labels: no vertical axis label (caption states |J1| vs Gamma);
# a light in-set annotation carries the axis name instead.
ax_ins.set_xlabel(
    r"$\Gamma/t$",
    fontsize=6.8,
    labelpad=0,
)

ax_ins.set_ylabel("")

# Do not use a large "slope=..." title.
# Put only the exponent inside the inset.
ax_ins.text(
    0.07,
    0.88,
    rf"$p={slope:.2f}$",
    transform=ax_ins.transAxes,
    fontsize=7.2,
    ha="left",
    va="top",
)

ax_ins.text(
    0.07,
    0.72,
    r"$|J_1|/t$",
    transform=ax_ins.transAxes,
    fontsize=6.5,
    ha="left",
    va="top",
)

# Only two labelled x ticks are enough for such a small inset.
ax_ins.set_xticks([
    0.03,
    0.06,
])

ax_ins.set_xticklabels([
    "0.03",
    "0.06",
])

# Keep only major logarithmic labels on y.
# Minor-label suppression is important because the inset is small.
ax_ins.xaxis.set_minor_formatter(
    NullFormatter()
)

ax_ins.yaxis.set_minor_formatter(
    NullFormatter()
)

ax_ins.tick_params(
    axis="both",
    which="major",
    labelsize=6.2,
    pad=1.5,
)

ax_ins.tick_params(
    axis="both",
    which="minor",
    length=2,
)


# ============================================================
# Common cosmetic settings
# ============================================================

for ax in axes:

    ax.tick_params(
        direction="in",
        top=True,
        right=True,
    )
    # 刻度 ≤5 个（承融要求：纵轴/横轴数字不超过 5 个）
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_locator(MaxNLocator(5))


# ============================================================
# Output
# ============================================================

out_dir = Path(__file__).resolve().parent.parent / "generated"
out_dir.mkdir(exist_ok=True)

png_path = (
    out_dir
    / "fig5_resonance_diagnostic_v3.png"
)

pdf_path = (
    out_dir
    / "fig5_resonance_diagnostic_v3.pdf"
)

fig.savefig(
    png_path,
    dpi=400,
    bbox_inches="tight",
)

fig.savefig(
    pdf_path,
    bbox_inches="tight",
)

print(
    f"saved: {png_path}"
)

print(
    f"saved: {pdf_path}"
)
