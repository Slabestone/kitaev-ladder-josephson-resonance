import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ============================================================
# Figure 6: selective -> collective clamping
#
# Effective isolated-channel representation.
#
# Directly realized for L_M = 1.
# For a finite normal segment, interpret
#       epsilon -> epsilon_q
#       Gamma   -> Gamma_q
# for an isolated normal mode q.
# ============================================================

# -------------------------
# Parameters
# -------------------------
t = 1.0
Gamma = 0.10 * t
t_perp = 0.16 * t

# Only epsilon is changed between the two panels.
epsilon_off = 0.10 * t
epsilon_res = 0.0

theta = np.linspace(0.0, 2.0 * np.pi, 1001)


# ============================================================
# Basic analytic functions
# ============================================================

def detuned_branches(theta, epsilon, Gamma):
    s = np.sin(theta / 2.0)
    rad = np.sqrt(
        epsilon**4
        + 4.0 * epsilon**2 * Gamma**2
        + 4.0 * Gamma**4 * s**2
    )
    em2 = (epsilon**2 + 2.0 * Gamma**2 - rad) / 2.0
    ep2 = (epsilon**2 + 2.0 * Gamma**2 + rad) / 2.0
    em = np.sqrt(np.maximum(em2, 0.0))
    ep = np.sqrt(np.maximum(ep2, 0.0))
    return em, ep


def helicity_split(e, t_perp):
    E1 = e + t_perp
    E2 = np.abs(e - t_perp)
    return E1, E2


def equilibrium_contribution(e, t_perp):
    return -np.maximum(e, t_perp)


def thresholds(epsilon, Gamma):
    root = np.sqrt(epsilon**2 + 4.0 * Gamma**2)
    t1 = (root - np.abs(epsilon)) / 2.0
    t2 = (root + np.abs(epsilon)) / 2.0
    t3 = np.sqrt(epsilon**2 + 2.0 * Gamma**2)
    return t1, t2, t3


# ============================================================
# Data: off resonance
# ============================================================
em_off, ep_off = detuned_branches(theta, epsilon_off, Gamma)
Em1_off, Em2_off = helicity_split(em_off, t_perp)
Ep1_off, Ep2_off = helicity_split(ep_off, t_perp)
calEm_off = equilibrium_contribution(em_off, t_perp)
calEp_off = equilibrium_contribution(ep_off, t_perp)
t1_off, t2_off, t3_off = thresholds(epsilon_off, Gamma)

# ============================================================
# Data: resonance
# ============================================================
em_res, ep_res = detuned_branches(theta, epsilon_res, Gamma)
Em1_res, Em2_res = helicity_split(em_res, t_perp)
Ep1_res, Ep2_res = helicity_split(ep_res, t_perp)
calEm_res = equilibrium_contribution(em_res, t_perp)
calEp_res = equilibrium_contribution(ep_res, t_perp)
t1_res, t2_res, t3_res = thresholds(epsilon_res, Gamma)

# ============================================================
# Sanity checks
# ============================================================
print("OFF resonance")
print(f"  epsilon/t = {epsilon_off/t:.4f}")
print(f"  t1/t      = {t1_off/t:.6f}")
print(f"  t_perp/t  = {t_perp/t:.6f}")
print(f"  t2/t      = {t2_off/t:.6f}")
print(f"  t3/t      = {t3_off/t:.6f}")
print("  selective corridor?", t1_off < t_perp < t2_off)
print("\nRESONANCE")
print(f"  t1=t2     = {t1_res/t:.6f}")
print(f"  t3/t      = {t3_res/t:.6f}")
print("  complete clamping?", t_perp > t3_res)


# ============================================================
# Figure layout:  a1|b1|c / a2|b2|c
# ============================================================
fig = plt.figure(figsize=(12.8, 6.6))
gs = GridSpec(2, 3, figure=fig, width_ratios=[1.0, 1.0, 1.05],
              height_ratios=[1.15, 0.85], wspace=0.42, hspace=0.16)
ax_a1 = fig.add_subplot(gs[0, 0])
ax_a2 = fig.add_subplot(gs[1, 0], sharex=ax_a1)
ax_b1 = fig.add_subplot(gs[0, 1])
ax_b2 = fig.add_subplot(gs[1, 1], sharex=ax_b1)
ax_c = fig.add_subplot(gs[:, 2])

xticks = [0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi]
xticklabels = [r"$0$", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"]

def format_theta_axis(ax, show_xlabel=True):
    ax.set_xlim(0, 2*np.pi)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    if show_xlabel:
        ax.set_xlabel(r"$\theta$")
    else:
        ax.tick_params(labelbottom=False)


# (a1) OFF RESONANCE: helicity-resolved positive ladder spectrum
ax_a1.plot(theta, Em1_off / t, lw=2.1, label=r"$e_-+t_\perp$")
ax_a1.plot(theta, Em2_off / t, lw=2.1, ls="--", label=r"$|e_- - t_\perp|$")
ax_a1.plot(theta, Ep1_off / t, lw=2.1, label=r"$e_++t_\perp$")
ax_a1.plot(theta, Ep2_off / t, lw=2.1, ls="--", label=r"$|e_+ - t_\perp|$")
ax_a1.axhline(t_perp/t, lw=1.2, ls=":")
ax_a1.set_title(r"(a) Off resonance: selective clamping" "\n"
                r"$\epsilon/t=0.10,\quad t_\perp/t=0.16$", fontsize=11)
ax_a1.set_ylabel(r"positive ladder energy $E/t$")
ax_a1.set_ylim(0.0, 0.355)
format_theta_axis(ax_a1, show_xlabel=False)
ax_a1.annotate(r"$E_{-,1}+E_{-,2}=2t_\perp$",
               xy=(np.pi, t_perp/t), xytext=(0.18*np.pi, 0.275),
               bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.55, edgecolor="0.7", linewidth=0.5),
               arrowprops=dict(arrowstyle="->", lw=1.0), fontsize=9.5)
ax_a1.text(1.08*np.pi, 0.115, r"mirror pair about $t_\perp$", fontsize=9, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.55, edgecolor="0.7", linewidth=0.5))
ax_a1.legend(loc="upper right", fontsize=7.8, frameon=False)

# (a2) OFF RESONANCE: resulting equilibrium contributions
ax_a2.plot(theta, calEm_off / t, lw=3.0, label=r"$\mathcal{E}_-$")
ax_a2.plot(theta, calEp_off / t, lw=3.0, label=r"$\mathcal{E}_+$")
ax_a2.axhline(-t_perp/t, lw=1.0, ls=":")
ax_a2.set_ylabel(r"low-energy contribution" "\n" r"$\mathcal{E}/t$")
ax_a2.set_ylim(-0.181, -0.145)
format_theta_axis(ax_a2, show_xlabel=True)
ax_a2.annotate(r"$\mathcal{E}_-=-t_\perp$" "\n" r"(clamped)",
               xy=(0.95*np.pi, -t_perp/t), xytext=(0.13*np.pi, -0.1515),
               bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.55, edgecolor="0.7", linewidth=0.5),
               arrowprops=dict(arrowstyle="->", lw=1.0), fontsize=9.5)
ax_a2.annotate(r"$\mathcal{E}_+=-e_+(\theta)$" "\n" r"(phase active)",
               xy=(np.pi, calEp_off[len(theta)//2]/t),
               xytext=(1.15*np.pi, -0.177),
               bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.55, edgecolor="0.7", linewidth=0.5),
               arrowprops=dict(arrowstyle="->", lw=1.0), fontsize=9.3)
ax_a2.legend(loc="upper right", fontsize=8.5, frameon=False)

# (b1) RESONANCE: both parent pairs mirror around t_perp
ax_b1.plot(theta, Em1_res / t, lw=2.1, label=r"$e_-+t_\perp$")
ax_b1.plot(theta, Em2_res / t, lw=2.1, ls="--", label=r"$|e_- - t_\perp|$")
ax_b1.plot(theta, Ep1_res / t, lw=2.1, label=r"$e_++t_\perp$")
ax_b1.plot(theta, Ep2_res / t, lw=2.1, ls="--", label=r"$|e_+ - t_\perp|$")
ax_b1.axhline(t_perp/t, lw=1.2, ls=":")
ax_b1.set_title(r"(b) Resonance: low-energy sector collectively clamped" "\n"
                r"$\epsilon=0,\quad t_\perp/t=0.16>\sqrt{2}\Gamma/t$", fontsize=11)
ax_b1.set_ylabel(r"positive ladder energy $E/t$")
ax_b1.set_ylim(0.0, 0.33)
format_theta_axis(ax_b1, show_xlabel=False)
ax_b1.annotate(r"$E_{\pm,1}+E_{\pm,2}=2t_\perp$",
               xy=(np.pi, t_perp/t), xytext=(0.17*np.pi, 0.285),
               bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.55, edgecolor="0.7", linewidth=0.5),
               arrowprops=dict(arrowstyle="->", lw=1.0), fontsize=9.5)
ax_b1.text(0.17*np.pi, 0.055, "both helicity pairs\nmirror about " r"$t_\perp$", fontsize=9.2, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.55, edgecolor="0.7", linewidth=0.5))
ax_b1.legend(loc="upper right", fontsize=7.8, frameon=False)

# (b2) RESONANCE: both equilibrium contributions clamped
ax_b2.plot(theta, calEm_res / t, lw=3.2, label=r"$\mathcal{E}_-$")
ax_b2.plot(theta, calEp_res / t, lw=2.2, ls="--", label=r"$\mathcal{E}_+$")
ax_b2.axhline(-t_perp/t, lw=1.0, ls=":")
ax_b2.set_ylabel(r"low-energy contribution" "\n" r"$\mathcal{E}/t$")
ax_b2.set_ylim(-0.176, -0.144)
format_theta_axis(ax_b2, show_xlabel=True)
ax_b2.annotate(r"$\mathcal{E}_-=\mathcal{E}_+=-t_\perp$" "\n"
               r"$\Rightarrow\ J_{1,\mathrm{low}}=0$",
               xy=(np.pi, -t_perp/t), xytext=(0.25*np.pi, -0.149),
               bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.55, edgecolor="0.7", linewidth=0.5),
               arrowprops=dict(arrowstyle="->", lw=1.0), fontsize=9.5)
ax_b2.legend(loc="lower right", fontsize=8.5, frameon=False)

# (c) DETUNING--CLAMPING MAP
eps_map = np.linspace(0.0, 0.30*t, 600)
t1 = np.zeros_like(eps_map); t2 = np.zeros_like(eps_map); t3 = np.zeros_like(eps_map)
for i, eps in enumerate(eps_map):
    t1[i], t2[i], t3[i] = thresholds(eps, Gamma)
epsx = eps_map / t
ax_c.plot(epsx, t1/t, lw=2.4, label=r"$t_1$")
ax_c.plot(epsx, t2/t, lw=2.4, label=r"$t_2$")
ax_c.plot(epsx, t3/t, lw=2.4, label=r"$t_3$")
ax_c.fill_between(epsx, 0.0, t1/t, alpha=0.12)
ax_c.fill_between(epsx, t1/t, t2/t, alpha=0.16)
ax_c.fill_between(epsx, t2/t, t3/t, alpha=0.12)
ax_c.fill_between(epsx, t3/t, 0.35, alpha=0.08, hatch="//")
ax_c.set_xlim(0.0, 0.30)
ax_c.set_ylim(0.0, 0.35)
ax_c.set_xlabel(r"$|\epsilon|/t$")
ax_c.set_ylabel(r"$t_\perp/t$")
ax_c.set_title(r"(c) Detuning--clamping map", fontsize=11)
ax_c.scatter([0.0], [Gamma/t], s=28, zorder=5)
ax_c.scatter([0.0], [np.sqrt(2.0)*Gamma/t], s=28, zorder=5)
ax_c.annotate(r"$t_1=t_2=\Gamma$", xy=(0.0, Gamma/t), xytext=(0.012, 0.080),
               bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.55, edgecolor="0.7", linewidth=0.5),
              arrowprops=dict(arrowstyle="->", lw=0.9), fontsize=9)
ax_c.annotate(r"$t_3=\sqrt{2}\Gamma$", xy=(0.0, np.sqrt(2.0)*Gamma/t),
              xytext=(0.012, 0.156), arrowprops=dict(arrowstyle="->", lw=0.9), fontsize=9)
ax_c.text(0.215, 0.047, "chain-like /\nbranch switching", ha="center", fontsize=9.2, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.55, edgecolor="0.7", linewidth=0.5))
ax_c.text(0.145, 0.175, "one branch active\n" r"(smooth-$\pi$)", ha="center", fontsize=9.2, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.55, edgecolor="0.7", linewidth=0.5))
ax_c.text(0.225, 0.255, "upper-branch\npartial clamping", ha="center", fontsize=9.2, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.55, edgecolor="0.7", linewidth=0.5))
ax_c.text(0.205, 0.323, "low-energy fully clamped\n" r"$J_{1,\mathrm{low}}=0$", ha="center", fontsize=9.2, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.55, edgecolor="0.7", linewidth=0.5))
ax_c.legend(loc="center right", fontsize=8.5, frameon=False)

# ============================================================
# Global figure annotation
# ============================================================
fig.text(0.5, 0.01,
         r"Effective isolated-channel schematic: "
         r"$L_M=1$ directly, or an isolated finite-segment mode "
         r"$q$ under $(\epsilon,\Gamma)\rightarrow(\epsilon_q,\Gamma_q)$.",
         ha="center", va="bottom", fontsize=9.5)

# ============================================================
# Save
# ============================================================
plt.subplots_adjust(left=0.075, right=0.985, top=0.93, bottom=0.12)

import os
outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'generated')
os.makedirs(outdir, exist_ok=True)
plt.savefig(os.path.join(outdir, 'fig-clamping-mechanism.png'), dpi=300, bbox_inches="tight")
plt.savefig(os.path.join(outdir, 'fig-clamping-mechanism.pdf'), bbox_inches="tight")
print("Saved fig-clamping-mechanism.png/pdf to generated/")
