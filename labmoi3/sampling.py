from __future__ import annotations

import numpy as np

from utils import orthonormal_basis_TBN


def sample_triangle(
    V1: np.ndarray, V2: np.ndarray, V3: np.ndarray, rng: np.random.Generator, n: int
) -> np.ndarray:
    r1 = rng.random(n)
    r2 = rng.random(n)
    flip = r1 + r2 > 1.0
    r1 = np.where(flip, 1.0 - r1, r1)
    r2 = np.where(flip, 1.0 - r2, r2)
    return V1 + r1[:, None] * (V2 - V1) + r2[:, None] * (V3 - V1)


def sample_point_in_triangle_uv(uv_vertices: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    r1, r2 = rng.random(2)
    if r1 + r2 > 1.0:
        r1, r2 = 1.0 - r1, 1.0 - r2
    return uv_vertices[0] + r1 * (uv_vertices[1] - uv_vertices[0]) + r2 * (uv_vertices[2] - uv_vertices[0])


def sample_uniform_disk(
    C: np.ndarray, R_c: float, plane_normal: np.ndarray, rng: np.random.Generator, n: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    T, B, n_unit = orthonormal_basis_TBN(plane_normal)
    u1 = rng.random(n)
    u2 = rng.random(n)
    rho = R_c * np.sqrt(u1)
    phi = 2.0 * np.pi * u2
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    P = C + x[:, None] * T + y[:, None] * B
    return P, T, B, n_unit, rho, phi


def sample_uniform_sphere(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    u1 = rng.random(n)
    u2 = rng.random(n)
    z = 2.0 * u1 - 1.0
    phi = 2.0 * np.pi * u2
    r_xy = np.sqrt(np.maximum(0.0, 1.0 - z**2))
    x = r_xy * np.cos(phi)
    y = r_xy * np.sin(phi)
    P = np.column_stack([x, y, z])
    return P, z, phi


def sample_cosine_hemisphere(
    rng: np.random.Generator, n: int, T: np.ndarray, B: np.ndarray, n_unit: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    u1 = rng.random(n)
    u2 = rng.random(n)
    z = np.sqrt(u1)
    sin_theta = np.sqrt(np.maximum(0.0, 1.0 - u1))
    phi = 2.0 * np.pi * u2
    x = sin_theta * np.cos(phi)
    y = sin_theta * np.sin(phi)
    direction = x[:, None] * T + y[:, None] * B + z[:, None] * n_unit
    cos_theta = np.sum(direction * n_unit, axis=1)
    return direction, cos_theta


def cosine_equal_area_expected_counts(N: int, K: int) -> np.ndarray:
    """Ожидаемые числа по полосам для пунктира на гистограмме косинуса."""
    z_edges = 1.0 - np.linspace(0.0, 1.0, K + 1)
    expected = np.zeros(K, dtype=np.float64)
    for j in range(K):
        z_j, z_jp = z_edges[j], z_edges[j + 1]
        expected[j] = N * (z_j * z_j - z_jp * z_jp)
    return expected
