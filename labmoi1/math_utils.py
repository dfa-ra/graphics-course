"""Векторы (x,y,z). Операции явно: add, sub, scale, dot, cross, length, normalize."""

import math


def add(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def scale(v, s):
    return (v[0] * s, v[1] * s, v[2] * s)


def dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def cross(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def length(v):
    return math.sqrt(dot(v, v))


_LEN_EPS = 1e-15


def normalize_safe(v):
    ln = length(v)
    if ln < _LEN_EPS:
        return None
    return scale(v, 1.0 / ln)


def normalize(v):
    u = normalize_safe(v)
    if u is None:
        return (0.0, 0.0, 1.0)
    return u


def triangle_normal_P0P1P2(P0, P1, P2):
    e_p1 = sub(P1, P0)
    e_p2 = sub(P2, P0)
    n = cross(e_p2, e_p1)
    u = normalize_safe(n)
    if u is None:
        return (0.0, 0.0, 1.0)
    return u


def point_P_T_from_local_xy(P0, P1, P2, x, y):
    d1 = sub(P1, P0)
    d2 = sub(P2, P0)
    u1 = normalize_safe(d1)
    u2 = normalize_safe(d2)
    if u1 is None or u2 is None:
        return P0
    return add(P0, add(scale(u1, x), scale(u2, y)))
