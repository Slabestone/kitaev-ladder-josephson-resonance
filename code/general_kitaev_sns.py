"""
general_kitaev_sns.py

Generalized Kitaev ladder SNS Josephson junction

Features:
- variable number of legs (N_leg)
- variable normal length (L_N)
- variable interface hopping Gamma
- finite BdG diagonalization
- CPR calculation

Author: Echo (回响) / Leqi (乐祈) — 2026-08-20
"""

import numpy as np
from scipy.linalg import eigh


class KitaevSNS:

    def __init__(
        self,
        N_leg=2,
        L_left=20,
        L_N=4,
        L_right=20,
        t=1.0,
        Delta=0.6,
        mu_SC=0.0,
        mu_N=0.0,
        t_perp=0.1,
        Gamma=0.1,
        phi=0.0,
    ):

        self.N_leg = N_leg
        self.L_left = L_left
        self.L_N = L_N
        self.L_right = L_right

        self.t = t
        self.Delta = Delta

        self.mu_SC = mu_SC
        self.mu_N = mu_N

        self.t_perp = t_perp
        self.Gamma = Gamma
        self.phi = phi  # ladder interchain phase (leg 1 vs leg 0)

    def index(self, region, x, leg):
        """
        region:
        0 left SC
        1 normal
        2 right SC
        """
        if region == 0:
            offset = 0
        elif region == 1:
            offset = self.L_left
        elif region == 2:
            offset = self.L_left + self.L_N

        return offset * self.N_leg + x * self.N_leg + leg

    def geometry(self):
        return self.L_left + self.L_N + self.L_right

    def build_hamiltonian(self, theta):
        L = self.geometry()
        nsite = L * self.N_leg

        h = np.zeros((nsite, nsite), dtype=complex)
        Delta_mat = np.zeros_like(h)

        # chemical potential
        for region, length, mu in [
            (0, self.L_left, self.mu_SC),
            (1, self.L_N, self.mu_N),
            (2, self.L_right, self.mu_SC),
        ]:
            for x in range(length):
                for leg in range(self.N_leg):
                    i = self.index(region, x, leg)
                    h[i, i] = -mu

        # longitudinal hopping
        for region, length in [(0, self.L_left), (1, self.L_N), (2, self.L_right)]:
            for x in range(length - 1):
                for leg in range(self.N_leg):
                    i = self.index(region, x, leg)
                    j = self.index(region, x + 1, leg)
                    h[i, j] = -self.t
                    h[j, i] = -self.t

        # transverse ladder hopping
        for region, length in [(0, self.L_left), (1, self.L_N), (2, self.L_right)]:
            for x in range(length):
                for leg in range(self.N_leg - 1):
                    i = self.index(region, x, leg)
                    j = self.index(region, x, leg + 1)
                    h[i, j] = -self.t_perp
                    h[j, i] = -self.t_perp

        # interface hopping Gamma
        for leg in range(self.N_leg):
            # left SC -> N
            i = self.index(0, self.L_left - 1, leg)
            j = self.index(1, 0, leg)
            h[i, j] = -self.Gamma
            h[j, i] = -self.Gamma

            # N -> right SC
            i = self.index(1, self.L_N - 1, leg)
            j = self.index(2, 0, leg)
            h[i, j] = -self.Gamma
            h[j, i] = -self.Gamma

        # p-wave pairing only in SC regions
        # v0.2: Paper 1 convention — interchain phase phi on leg 1 (matching p1b_sns.py):
        #   Db = Delta * exp(i*(theta_lead + dphi)), dphi = phi if leg==1 else 0
        for region, length, phase in [(0, self.L_left, 0.0), (2, self.L_right, theta)]:
            for x in range(length - 1):
                for leg in range(self.N_leg):
                    dphi = self.phi if leg == 1 else 0.0
                    i = self.index(region, x, leg)
                    j = self.index(region, x + 1, leg)
                    Delta_mat[i, j] = self.Delta * np.exp(1j * (phase + dphi))
                    Delta_mat[j, i] = -Delta_mat[i, j]

        return h, Delta_mat

    def BdG(self, theta):
        h, D = self.build_hamiltonian(theta)
        H = np.block([[h, D], [-D.conjugate(), -h.conjugate()]])
        return H

    def ground_energy(self, theta):
        H = self.BdG(theta)
        eig = eigh(H, eigvals_only=True)
        positive = eig[eig > 0]
        return -0.5 * np.sum(positive)

    def CPR(self, theta_list):
        energies = np.array([self.ground_energy(theta) for theta in theta_list])
        current = np.gradient(energies, theta_list)
        return energies, current


def classify(eigen_min_theta_pi):
    """Classify 0/pi by distance of E_min to theta=0 vs theta=pi."""
    d0 = min(abs(eigen_min_theta_pi), abs(eigen_min_theta_pi - 2.0))
    dpi = min(abs(eigen_min_theta_pi - 1.0), abs(eigen_min_theta_pi + 1.0))
    return "pi-type" if dpi < d0 else "0-type"


if __name__ == "__main__":

    theta = np.linspace(-np.pi, np.pi, 201)

    # -----------------------
    # Paper 1 fingerprint
    # Case 1: pi-junction
    # -----------------------
    model1 = KitaevSNS(
        N_leg=2, L_N=4, mu_N=0.0, Gamma=0.1, t_perp=0.1,
        mu_SC=0.0, phi=np.pi / 3.0,
    )
    E1, J1 = model1.CPR(theta)
    th_min1 = theta[np.argmin(E1)] / np.pi
    c1 = classify(th_min1)
    print(f"Case 1 (N_leg=2, L_N=4, mu_N=0): E_min at {th_min1:.3f} pi -> {c1}")

    # -----------------------
    # Case 2: 0-junction
    # -----------------------
    model2 = KitaevSNS(
        N_leg=2, L_N=5, mu_N=1.0, Gamma=0.1, t_perp=0.1,
        mu_SC=0.0, phi=np.pi / 3.0,
    )
    E2, J2 = model2.CPR(theta)
    th_min2 = theta[np.argmin(E2)] / np.pi
    c2 = classify(th_min2)
    print(f"Case 2 (N_leg=2, L_N=5, mu_N=1): E_min at {th_min2:.3f} pi -> {c2}")

    # 报告
    print("\n=== Fingerprint 判定 ===")
    print(f"Case 1: {c1} (期望 pi-type) -> {'PASS' if c1 == 'pi-type' else 'FAIL'}")
    print(f"Case 2: {c2} (期望 0-type) -> {'PASS' if c2 == '0-type' else 'FAIL'}")

    # 保存数据供对比
    np.savez(
        "general_kitaev_sns_cases.npz",
        theta=theta,
        E1=E1, J1=J1,
        E2=E2, J2=J2,
        th_min1=th_min1, th_min2=th_min2,
    )
    print("数据已保存 general_kitaev_sns_cases.npz")
