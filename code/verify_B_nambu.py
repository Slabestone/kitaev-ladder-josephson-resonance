#!/usr/bin/env python3
"""Independent Nambu audit of δ_θ P ~ Γ⁴ B cosθ (contact-reduced sweet point).

Two-site SC contacts (L_S=2) match D_L=D_R=E²(E²-4t²) in the appendix.
Expect at t=1:
  det(E-H_0)-det(E-H_π) = 16 (-1)^{L_M-1} Γ⁴ E²
hence B = 8 (-1)^{L_M-1} after stripping E².
"""
from __future__ import annotations

import numpy as np


def bdg(
    L_M: int,
    theta: float,
    Gamma: float,
    t: float = 1.0,
    mu_M: float = 0.0,
    L_S: int = 2,
    t_list: list[float] | None = None,
) -> np.ndarray:
    N = 2 * L_S + L_M
    h = np.zeros((N, N), complex)
    Delta = np.zeros((N, N), complex)

    def idx(region: int, x: int) -> int:
        if region == 0:
            return x
        if region == 1:
            return L_S + x
        return L_S + L_M + x

    for x in range(L_M):
        h[idx(1, x), idx(1, x)] = -mu_M
    for region, length in [(0, L_S), (2, L_S)]:
        for x in range(length - 1):
            i, j = idx(region, x), idx(region, x + 1)
            h[i, j] = h[j, i] = -t
    if t_list is None:
        t_list = [t] * max(L_M - 1, 0)
    for x, tj in enumerate(t_list):
        i, j = idx(1, x), idx(1, x + 1)
        h[i, j] = h[j, i] = -tj
    i, j = idx(0, L_S - 1), idx(1, 0)
    h[i, j] = h[j, i] = -Gamma
    i, j = idx(1, L_M - 1), idx(2, 0)
    h[i, j] = h[j, i] = -Gamma
    for x in range(L_S - 1):
        i, j = idx(0, x), idx(0, x + 1)
        Delta[i, j], Delta[j, i] = t, -t
        i, j = idx(2, x), idx(2, x + 1)
        d = t * np.exp(1j * theta)
        Delta[i, j], Delta[j, i] = d, -d
    return np.block([[h, Delta], [Delta.conj().T, -h.T]])


def phase_ratio(L_M: int, E: float = 0.37, Gamma: float = 1e-3, **kw) -> float:
    H0 = bdg(L_M, 0.0, Gamma, **kw)
    Hpi = bdg(L_M, np.pi, Gamma, **kw)
    n = H0.shape[0]
    d0 = np.linalg.det(E * np.eye(n) - H0)
    dpi = np.linalg.det(E * np.eye(n) - Hpi)
    return ((d0 - dpi) / (Gamma**4 * E**2)).real


if __name__ == "__main__":
    print("L_S=2, t=1: (det0-detπ)/(Γ⁴ E²) vs 16(-1)^{L_M-1}")
    for LM in range(1, 6):
        got = phase_ratio(LM, Gamma=0.11)
        exp = 16 * ((-1) ** (LM - 1))
        assert abs(got/exp-1) < 2e-8
        print(f"  L_M={LM}: {got:.6f}  expect {exp}  err={abs(got - exp):.2e}")
