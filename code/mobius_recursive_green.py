import numpy as np
from numpy.linalg import inv, norm


def get_surface_GF_mobius(z, H0, T, max_iter=5000, tol=1e-12):
    """
    Compute the surface Green's function using the Möbius recursive method.

    Parameters:
    - z: complex energy (iω_n)
    - H0: on-site Hamiltonian matrix (square np.array)
    - T: hopping matrix (same shape as H0)
    - max_iter: maximum number of iterations
    - tol: convergence tolerance

    Returns:
    - G0: surface Green's function
    """
    dim = H0.shape[0]
    I = np.eye(dim, dtype=complex)

    # Initial guess
    G0 = inv(z * I - H0)
    G1 = G0.copy()

    for _ in range(max_iter):
        # Möbius update (regularized form)
        denom = inv(I - T @ G1 @ T @ G0)
        G_new = G0 + G0 @ T @ denom @ T @ G0

        # Check convergence
        if norm(G_new - G1) < tol:
            return G_new
        G1 = G_new

    raise RuntimeError("Mobius recursive Green's function did not converge.")
