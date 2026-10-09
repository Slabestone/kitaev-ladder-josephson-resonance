import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv, eig
from mobius_recursive_green import get_surface_GF_mobius
# === Constants and Parameters === #
pi = np.pi
mu = 2.0
t = 1.0
delta = 0.6 * t
temp = 0.0001 * t
phi = pi / 3.0
uc = 1j
eta = 1e-5
# === Superconducting Phases === #
del1 = delta
del2 = delta * np.exp(uc * phi)
# === Helper Functions === #
def build_h11(tperp):
    h11 = np.zeros((4, 4), dtype=complex)
    h11[0, 0] = -mu
    h11[1, 1] = -mu
    h11[0, 1] = h11[1, 0] = -tperp
    h11[2, 2] = mu
    h11[3, 3] = mu
    h11[2, 3] = h11[3, 2] = tperp
    return h11
def build_t12(delpair_phases):
    t12 = np.zeros((4, 4), dtype=complex)
    thop = np.diag([-t, -t])
    thopc = -thop
    delpair = np.diag(delpair_phases[:2])
    delpairc = -np.conj(delpair)
    t12[0:2, 0:2] = thop
    t12[2:4, 2:4] = thopc
    t12[0:2, 2:4] = delpair
    t12[2:4, 0:2] = delpairc
    return t12
def josephson_current(gii1, gi1i):
    trace_term = np.trace(gii1 - gi1i)
    return -np.imag(trace_term) / pi
# === Main Simulation === #
def run_simulation():
    tperp = 0.15
    results = []
    for iedel in range(1, 201):
        theta = -2 * pi + 5.0 * 4.0 * pi * 1e-3 * (iedel - 1)
        js = 0.0
        for n in range(-100, 101):  # Reduced Matsubara range for speed
            wn = (2 * n + 1) * pi * temp
            z = uc * wn
            # Left SC
            h11_L = build_h11(tperp)
            t12_L = build_t12([del1, del2])
            gl = get_surface_GF_mobius(z, h11_L, t12_L)
            # Right SC
            del3 = delta * np.exp(uc * theta)
            del4 = delta * np.exp(uc * (phi + theta))
            h11_R = build_h11(tperp)
            t12_R = build_t12([del3, del4])
            gr = get_surface_GF_mobius(z, h11_R, t12_R)
            # Junction
            id4 = np.eye(4, dtype=complex)
            u = np.diag([-mu, -mu, mu, mu])
            un = np.diag([-mu, -mu, mu, mu])
            u[0, 1] = u[1, 0] = -tperp
            u[2, 3] = u[3, 2] = tperp
            un[0, 1] = un[1, 0] = -tperp
            un[2, 3] = un[3, 2] = tperp
            tt = 0.1 * t
            ttn = 1.0 * t
            tins12 = np.diag([-tt, -tt, tt, tt])
            tins21 = tins12.conj().T
            tn12 = np.diag([-ttn, -ttn, ttn, ttn])
            tn21 = tn12.conj().T
            gri = inv(z * id4 - u - tins12 @ gr @ tins21)
            gri = inv(z * id4 - un - tn12 @ gri @ tn21)
            gl = inv(z * id4 - u - tins21 @ gl @ tins12)
            gl = inv(z * id4 - un - tn21 @ gl @ tn12)
            glinv = inv(gl)
            griinv = inv(gri)
            gii = inv(glinv - tn12 @ gri @ tn21)
            gi1i1 = inv(griinv - tn21 @ gl @ tn12)
            gii1 = gii @ tn12 @ gri
            gi1i = gri @ tn21 @ gii
            js += josephson_current(gii1, gi1i)
        js_total = js * temp
        results.append((theta / pi, js_total))
    return results
# === Plot Function === #
def plot_results(results):
    theta_vals = np.array([item[0] for item in results])
    current_vals = np.array([item[1] for item in results])
    plt.figure(figsize=(8, 5))
    plt.plot(theta_vals, current_vals, label="Josephson Current", linewidth=2)
    plt.xlabel(r'$\\theta/\\pi$', fontsize=14)
    plt.ylabel(r'$J(\\theta)$', fontsize=14)
    plt.title("Josephson Current vs Phase Difference", fontsize=16)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("josephson_current_plot.pdf")
    plt.show()
# === Entry Point === #
if __name__ == "__main__":
    results = run_simulation()
    print("Simulation complete. Plotting...")
    plot_results(results)
