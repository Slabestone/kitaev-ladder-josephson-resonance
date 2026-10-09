#!/usr/bin/env python3
"""P1-B.2: S/N/S junction BdG（L lead + M normal + R lead，三区，接口 bond 无配对）。
模型: Kitaev ladder S/N/S（Paper 1 正文模型）
- 左/右 superconducting leads: Kitaev ladder（Δ≠0）
- 中间 normal segment: ladder，Δ=0（L_M sites）
- 接口 bond (L-M, M-R): hopping only, 无 pairing
块状基 BdG（已验证 convention）:
  Ψ = (c_1..c_N, c†_1..c†_N)
  H = [[h, Δ], [-Δ*, -h*]]
"""
import numpy as np

def build_sns(LM, NL, mu, t, Delta, t_perp, phi, theta, t_if=1.0):
    """S/N/S junction。
    LM: normal segment 长度；NL: 每侧 lead 长度
    theta: junction phase（R lead pairing phase）
    phi: interchain phase（ladder 腿间）
    """
    N = 2*NL + LM  # 总 site 数（单腿）
    dim = 2*N
    h = np.zeros((dim, dim), dtype=complex)
    # 基顺序: 每条腿 (L-lead NL sites, M LM sites, R-lead NL sites)
    # 腿 β: sites 0..N-1; 全粒子块先排两腿
    def hi(beta, x):
        return beta*N + x

    # on-site
    for beta in range(2):
        for x in range(N):
            h[hi(beta,x), hi(beta,x)] = -mu

    # 区域内 hopping + 接口 hopping
    for beta in range(2):
        for x in range(N-1):
            # 判断是否接口 bond
            # L-M 接口: L 最后 site (NL-1) - M 第一 site (NL)
            # M-R 接口: M 最后 (NL+LM-1) - R 第一 (NL+LM)
            if x == NL-1 or x == NL+LM-1:
                h[hi(beta,x), hi(beta,x+1)] = -t_if
                h[hi(beta,x+1), hi(beta,x)] = -t_if
            else:
                h[hi(beta,x), hi(beta,x+1)] = -t
                h[hi(beta,x+1), hi(beta,x)] = -t

    # inter-leg hopping（所有区域，normal 区也保留 t⊥）
    for x in range(N):
        h[hi(0,x), hi(1,x)] = -t_perp
        h[hi(1,x), hi(0,x)] = -t_perp

    # pairing（仅 SC lead 内部，M 区与接口 bond 无 pairing）
    Dmat = np.zeros((dim, dim), dtype=complex)
    for beta in range(2):
        for x in range(N-1):
            # 只在 lead 内部 bond 放 pairing
            in_L = x < NL-1
            in_R = x >= NL+LM
            if not (in_L or in_R):
                continue
            if x == NL-1 or x == NL+LM-1:
                continue  # 接口无 pairing
            # 相位: L lead θ=0, R lead θ=theta; interchain φ_β
            dphi = phi if beta == 1 else 0.0
            theta_lead = 0.0 if in_L else theta
            Db = Delta * np.exp(1j*(theta_lead + dphi))
            Dmat[hi(beta,x), hi(beta,x+1)] = Db
            Dmat[hi(beta,x+1), hi(beta,x)] = -Db

    H = np.zeros((2*dim, 2*dim), dtype=complex)
    H[:dim, :dim] = h
    H[:dim, dim:] = Dmat
    H[dim:, :dim] = -Dmat.conj()
    H[dim:, dim:] = -h.conj()
    return H


def ground_state_energy(H, Ntot=None):
    """E_GS = -1/2 * sum of positive BdG eigenvalues（BdG 标准约定，与 general_kitaev_sns 一致）。
    注：2026-08-21 统一——原为 sum of N most-negative（= -Σ正，2 倍偏差）。"""
    evals = np.linalg.eigvalsh(H)
    return -0.5 * np.sum(evals[evals > 0])


def josephson_current(H_fn, thetas, Ntot=None):
    """I(θ) = +dE_GS/dθ（2026-08-21 统一：与公式 36 / general_kitaev_sns 一致）。"""
    Egs = [ground_state_energy(H_fn(th), Ntot) for th in thetas]
    return np.gradient(Egs, thetas), Egs


if __name__ == "__main__":
    t = 1.0
    # 验证: Hermiticity + PHS
    LM, NL = 4, 40
    H = build_sns(LM, NL, 0.5, t, 0.6*t, 1.0, np.pi/3, np.pi/2)
    herm = np.max(np.abs(H - H.conj().T))
    dim = H.shape[0]
    Xi = np.zeros((dim, dim))
    Xi[:dim//2, dim//2:] = np.eye(dim//2)
    Xi[dim//2:, :dim//2] = np.eye(dim//2)
    phs = np.max(np.abs(Xi @ H.conj() @ Xi + H))
    print(f"S/N/S junction: dim={dim}, |H-H†|={herm:.2e}, |ΞHΞ+H|={phs:.2e}")
    print(f"  总 site = {2*NL+LM} (L:{NL} + M:{LM} + R:{NL})")
