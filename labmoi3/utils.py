
import numpy as np


def orthonormal_basis_TBN(N: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    N = np.asarray(N, dtype=np.float64)
    n = N / np.linalg.norm(N)
    if abs(n[2]) < 0.9:
        aux = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    else:
        aux = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    T = np.cross(aux, n)
    T = T / np.linalg.norm(T)
    B = np.cross(n, T)
    return T, B, n


def tangent_basis_from_c_p1_p2(
    N: np.ndarray, C: np.ndarray, P1: np.ndarray, P2: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_unit = np.asarray(N, dtype=np.float64)
    n_unit = n_unit / np.linalg.norm(n_unit)
    C = np.asarray(C, dtype=np.float64)
    P1 = np.asarray(P1, dtype=np.float64)
    P2 = np.asarray(P2, dtype=np.float64)

    def proj_plane(v: np.ndarray) -> np.ndarray:
        return v - np.dot(v, n_unit) * n_unit

    u1 = proj_plane(P1 - C)
    u2 = proj_plane(P2 - C)
    nu1 = np.linalg.norm(u1)
    nu2 = np.linalg.norm(u2)
    if nu1 < 1e-12 and nu2 < 1e-12:
        return orthonormal_basis_TBN(N)
    if nu1 >= 1e-12:
        T = u1 / nu1
    else:
        T = u2 / nu2
    w2 = u2 - np.dot(u2, T) * T
    nw2 = np.linalg.norm(w2)
    if nw2 >= 1e-12:
        B = w2 / nw2
    else:
        B = np.cross(n_unit, T)
        bn = np.linalg.norm(B)
        if bn < 1e-12:
            return orthonormal_basis_TBN(N)
        B = B / bn
    if np.dot(np.cross(T, B), n_unit) < 0:
        B = -B
    return T, B, n_unit
