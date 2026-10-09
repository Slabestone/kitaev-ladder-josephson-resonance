#!/usr/bin/env python3
"""D117 Fortran transfer-matrix RGF 的 Python 复刻（Gate 2 目标：直接复现 jcurrent1.dat）。
严格照 recur_ladder_D117_original.f90 算法转译（不发明）：
  - h11(4x4): ladder onsite（-mu + tperp 非对角，PHS）
  - t12(4x4): hopping + pairing（Δ1, Δ2 = del, del*e^{iφ}）
  - x(8x8) transfer matrix → eigendecomposition → gl = q12 @ q22^{-1}
  - junction: u/un 双层 embedding（tins=Γ=0.1t, tn=ttn=1.0t）
  - current: jo = -Im(gii1 - gi1i + gii1(3,3)-gi1i(3,3) + gii1(4,4)-gi1i(4,4))/π
  - js1 = Σ_n jo * temp；输出 (θ/π, temp*js, ...)
"""
import numpy as np

PI = np.pi
mu = 2.0
t = 1.0
del_ = 0.6 * t
temp = 0.0001 * t
phi = PI / 3.0
tperpl = 0.15 * t
tt = 0.1 * t
ttn = 1.0 * t
N_MATSUBARA = 1000  # n = -1000..1000


def build_h11(tperp):
    h11 = np.zeros((4, 4), dtype=complex)
    h11[0, 0] = -mu
    h11[1, 1] = -mu
    h11[0, 1] = -tperp
    h11[1, 0] = -tperp
    # PHS: 空穴块
    h11[2, 2] = mu
    h11[3, 3] = mu
    h11[2, 3] = tperp
    h11[3, 2] = tperp
    return h11


def build_t12(del1, del2):
    t12 = np.zeros((4, 4), dtype=complex)
    thop = np.diag([-t, -t])
    thopc = -thop
    delpair = np.diag([del1, del2])
    delpairc = -np.conj(delpair)
    t12[0:2, 0:2] = thop
    t12[2:4, 2:4] = thopc
    t12[0:2, 2:4] = delpair
    t12[2:4, 0:2] = delpairc
    return t12


def surface_gf(z, h11, t12):
    """transfer-matrix 法：x(8x8) → 特征分解 → gl = q12 @ q22^{-1}"""
    id4 = np.eye(4, dtype=complex)
    tinv = np.linalg.inv(t12)
    x = np.zeros((8, 8), dtype=complex)
    x[0:4, 4:8] = tinv
    x[4:8, 0:4] = -t12.conj().T
    x[4:8, 4:8] = (z * id4 - h11) @ tinv
    # 特征分解（Fortran diag 用 zgeev；numpy eigvals 等价）
    evals, evecs = np.linalg.eig(x)
    # Fortran 按 |E| 排序后取 q12/q22——numpy 用相同排序
    order = np.argsort(np.abs(evals))
    q = evecs[:, order]
    q12 = q[0:4, 4:8]
    q22 = q[4:8, 4:8]
    return q12 @ np.linalg.inv(q22)


def current_at_theta(theta):
    del1 = del_
    del2 = del_ * np.exp(1j * phi)
    del3 = del_ * np.exp(1j * theta)
    del4 = del_ * np.exp(1j * (phi + theta))

    h11_L = build_h11(tperpl)
    h11_R = build_h11(tperpl)
    t12_L = build_t12(del1, del2)
    t12_R = build_t12(del3, del4)

    id4 = np.eye(4, dtype=complex)

    # u: insulator (muins=mu) + tperp；tins: Γ
    u = np.zeros((4, 4), dtype=complex)
    u[0, 0] = -mu
    u[1, 1] = -mu
    u[2, 2] = mu
    u[3, 3] = mu
    u[0, 1] = u[1, 0] = -tperpl
    u[2, 3] = u[3, 2] = tperpl
    tins12 = np.diag([-tt, -tt, tt, tt])
    tins21 = tins12.conj().T

    # un: normal (mun=mu) + tperp；tn: ttn
    un = u.copy()
    tn12 = np.diag([-ttn, -ttn, ttn, ttn])
    tn21 = tn12.conj().T

    js = 0.0
    for n in range(-N_MATSUBARA, N_MATSUBARA + 1):
        wn = (2 * n + 1) * PI * temp
        z = 1j * wn

        gl = surface_gf(z, h11_L, t12_L)
        gr = surface_gf(z, h11_R, t12_R)

        # 双层 embedding（与 Fortran L1261-1271 一致）
        gri = np.linalg.inv(z * id4 - u - tins12 @ gr @ tins21)
        gri = np.linalg.inv(z * id4 - un - tn12 @ gri @ tn21)
        gl = np.linalg.inv(z * id4 - u - tins21 @ gl @ tins12)
        gl = np.linalg.inv(z * id4 - un - tn21 @ gl @ tn12)

        glinv = np.linalg.inv(gl)
        gii = np.linalg.inv(glinv - tn12 @ gri @ tn21)
        griinv = np.linalg.inv(gri)
        gi1i1 = np.linalg.inv(griinv - tn21 @ gl @ tn12)
        gii1 = gii @ tn12 @ gri
        gi1i = gri @ tn21 @ gii

        jo = np.imag(gii1[0, 0] - gi1i[0, 0] + gii1[2, 2] - gi1i[2, 2] + gii1[3, 3] - gi1i[3, 3]) / PI
        js += jo
    return js * temp


if __name__ == "__main__":
    print("=" * 60)
    print("Python transfer-matrix RGF 复刻（D117 Fortran）")
    print("=" * 60)
    # 快测 3 个 θ 点（全 200 点需 ~200*2001 次 RGF，慢；先验证 3 点）
    for theta_pi in [-2.0, -1.0, 0.0, 1.0]:
        js = current_at_theta(theta_pi * PI)
        print(f"θ/π={theta_pi:+.1f}: J={js:.6e}")
