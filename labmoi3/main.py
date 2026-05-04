"""Интерфейс и визуализация для МОИ ЛР3. Сэмплирование — в sampling.py."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from sampling import (
    cosine_equal_area_expected_counts,
    sample_cosine_hemisphere,
    sample_point_in_triangle_uv,
    sample_triangle,
    sample_uniform_disk,
    sample_uniform_sphere,
)
from utils import orthonormal_basis_TBN, tangent_basis_from_c_p1_p2

DEFAULT_N = 100_000
DEFAULT_SEED = 42
DEFAULT_VIS = 12_000
DEBOUNCE_MS = 350

RING_K = 12

TRIANGLE_DISJOINT_CIRCLES = 5

plt.rcParams.update(
    {
        "figure.facecolor": "#ececf0",
        "axes.facecolor": "#ffffff",
        "axes.edgecolor": "#c7c9d1",
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "grid.alpha": 0.4,
        "font.family": "sans-serif",
    }
)


# --- Визуализация: подвыборка точек, гистограммы по полосам (демонстрация, см. подписи к графикам) ---


def _subidx(rng: np.random.Generator, n: int, k: int) -> np.ndarray:
    k = min(n, max(1, k))
    return rng.choice(n, size=k, replace=False)


def _equal_area_radial_edges(R: float, K: int) -> np.ndarray:
    """Границы радиусов: кольца равной площади, r_k = R * sqrt(k/K)."""
    return R * np.sqrt(np.linspace(0.0, 1.0, K + 1))


def _bin_radial(d: np.ndarray, r_edges: np.ndarray) -> np.ndarray:
    k = np.searchsorted(r_edges, d, side="right") - 1
    return np.clip(k, 0, r_edges.size - 2)


# --- Треугольник: плоскость, uv, пять равных дисков (только UI; не дублирует §1 формулы) ---


def _triangle_frame(V1: np.ndarray, V2: np.ndarray, V3: np.ndarray):
    n = np.cross(V2 - V1, V3 - V1)
    T, B, n_unit = orthonormal_basis_TBN(n)
    G = (V1 + V2 + V3) / 3.0
    return G, T, B


def _to_uv(P: np.ndarray, G: np.ndarray, T: np.ndarray, B: np.ndarray) -> np.ndarray:
    return np.array([np.dot(P - G, T), np.dot(P - G, B)], dtype=np.float64)


def _dist_point_seg_2d(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    ap = p - a
    t = np.clip(np.dot(ap, ab) / (np.dot(ab, ab) + 1e-15), 0.0, 1.0)
    proj = a + t * ab
    return float(np.linalg.norm(p - proj))


def _min_dist_uv_to_triangle_edges(p_uv: np.ndarray, uv: np.ndarray) -> float:
    return float(
        min(
            _dist_point_seg_2d(p_uv, uv[0], uv[1]),
            _dist_point_seg_2d(p_uv, uv[1], uv[2]),
            _dist_point_seg_2d(p_uv, uv[2], uv[0]),
        )
    )


def _triangle_inradius_uv(uv: np.ndarray) -> float:
    """Радиус вписанной окружности в 2D-треугольнике (вершины в uv)."""
    a = float(np.linalg.norm(uv[1] - uv[0]))
    b = float(np.linalg.norm(uv[2] - uv[1]))
    c = float(np.linalg.norm(uv[0] - uv[2]))
    s = (a + b + c) / 2.0
    area = 0.5 * abs(
        (uv[1, 0] - uv[0, 0]) * (uv[2, 1] - uv[0, 1]) - (uv[2, 0] - uv[0, 0]) * (uv[1, 1] - uv[0, 1])
    )
    if s < 1e-12 or area < 1e-15:
        return 0.0
    return area / s


def _try_place_five_equal_centers(
    uv: np.ndarray, r: float, rng: np.random.Generator, max_trials: int
) -> list[np.ndarray] | None:
    """Пять непересекающихся дисков радиуса r, центры случайны; диски внутри треугольника."""
    eps = 5e-4
    if r < 1e-8:
        return None
    centers: list[np.ndarray] = []
    for _ in range(max_trials):
        if len(centers) >= TRIANGLE_DISJOINT_CIRCLES:
            return centers
        p_uv = sample_point_in_triangle_uv(uv, rng)
        if _min_dist_uv_to_triangle_edges(p_uv, uv) < r - 1e-7:
            continue
        ok = True
        for c2 in centers:
            if float(np.linalg.norm(p_uv - c2)) < 2.0 * r + eps:
                ok = False
                break
        if ok:
            centers.append(p_uv.copy())
    return None


def _sample_five_disjoint_disks_uv(uv: np.ndarray, rng: np.random.Generator) -> list[tuple[np.ndarray, float]] | None:
    """
    TRIANGLE_DISJOINT_CIRCLES непересекающихся дисков в плоскости треугольника (uv).
    Все диски одного радиуса r; r подбирается уменьшением от доли вписанного радиуса, центры случайны.
    """
    r_in = _triangle_inradius_uv(uv)
    if r_in < 1e-9:
        return None
    r = 0.42 * r_in
    r_min = max(1e-6, 0.02 * r_in)
    for it in range(36):
        if r < r_min:
            break
        sub = np.random.default_rng(int(rng.integers(1, 2**31 - 2)) + it * 97_981)
        centers = _try_place_five_equal_centers(uv, r, sub, max_trials=14_000)
        if centers is not None:
            return [(c, r) for c in centers]
        r *= 0.9
    return None


def _plot_disjoint_circles_3d(
    ax3,
    G: np.ndarray,
    T: np.ndarray,
    B: np.ndarray,
    centers_uv: list[np.ndarray],
    radii: list[float],
    **kw,
) -> None:
    th = np.linspace(0, 2 * np.pi, 80)
    colors = kw.pop("colors", None)
    lw = kw.pop("lw", 0.55)
    alpha = kw.pop("alpha", 0.55)
    for i, (cuv, r) in enumerate(zip(centers_uv, radii)):
        if r <= 1e-9:
            continue
        col = colors[i % len(colors)] if colors is not None else "0.35"
        cx = G[0] + cuv[0] * T[0] + cuv[1] * B[0]
        cy = G[1] + cuv[0] * T[1] + cuv[1] * B[1]
        cz = G[2] + cuv[0] * T[2] + cuv[1] * B[2]
        x = cx + r * (np.cos(th) * T[0] + np.sin(th) * B[0])
        y = cy + r * (np.cos(th) * T[1] + np.sin(th) * B[1])
        z = cz + r * (np.cos(th) * T[2] + np.sin(th) * B[2])
        ax3.plot(x, y, z, color=col, lw=lw, alpha=alpha)


def _plot_plane_circles_3d(ax3, G: np.ndarray, T: np.ndarray, B: np.ndarray, radii: np.ndarray, **kw) -> None:
    th = np.linspace(0, 2 * np.pi, 80)
    color = kw.pop("color", "0.35")
    lw = kw.pop("lw", 0.45)
    alpha = kw.pop("alpha", 0.45)
    for r in radii:
        if r <= 1e-9:
            continue
        x = G[0] + r * (np.cos(th) * T[0] + np.sin(th) * B[0])
        y = G[1] + r * (np.cos(th) * T[1] + np.sin(th) * B[1])
        z = G[2] + r * (np.cos(th) * T[2] + np.sin(th) * B[2])
        ax3.plot(x, y, z, color=color, lw=lw, alpha=alpha)


def _plot_latitudes_unit_sphere(ax3, z_levels: np.ndarray, **kw) -> None:
    color = kw.pop("color", "0.35")
    lw = kw.pop("lw", 0.45)
    alpha = kw.pop("alpha", 0.4)
    th = np.linspace(0, 2 * np.pi, 80)
    for zc in z_levels:
        if abs(zc) >= 1.0 - 1e-9:
            continue
        rr = np.sqrt(max(0.0, 1.0 - zc * zc))
        xs = rr * np.cos(th)
        ys = rr * np.sin(th)
        zs = np.full_like(th, zc)
        ax3.plot(xs, ys, zs, color=color, lw=lw, alpha=alpha)


def _plot_latitudes_hemisphere_world(
    ax3, T: np.ndarray, B: np.ndarray, n_unit: np.ndarray, z_levels: np.ndarray, **kw
) -> None:
    color = kw.pop("color", "0.35")
    lw = kw.pop("lw", 0.45)
    alpha = kw.pop("alpha", 0.4)
    th = np.linspace(0, 2 * np.pi, 80)
    for zc in z_levels:
        if zc <= 1e-9 or zc >= 1.0 - 1e-9:
            continue
        rr = np.sqrt(max(0.0, 1.0 - zc * zc))
        xs = rr * np.cos(th) * T[0] + rr * np.sin(th) * B[0] + zc * n_unit[0]
        ys = rr * np.cos(th) * T[1] + rr * np.sin(th) * B[1] + zc * n_unit[1]
        zs = rr * np.cos(th) * T[2] + rr * np.sin(th) * B[2] + zc * n_unit[2]
        ax3.plot(xs, ys, zs, color=color, lw=lw, alpha=alpha)


def _bin_sphere_z_equal_area(z: np.ndarray, K: int) -> np.ndarray:
    """Полосы на единичной сфере равной площади: z ∈ [-1,1] с шагом 2/K."""
    z_edges = np.linspace(-1.0, 1.0, K + 1)
    k = np.searchsorted(z_edges, z, side="right") - 1
    return np.clip(k, 0, K - 1)


def _bin_hemisphere_equal_area(z: np.ndarray, K: int) -> np.ndarray:
    """Полосы на полусфере равной площади: z = cos θ от полюса к экватору, z_j = 1 - j/K."""
    z_edges = 1.0 - np.linspace(0.0, 1.0, K + 1)
    k = np.zeros(z.shape[0], dtype=np.int32)
    for j in range(K):
        lo, hi = z_edges[j + 1], z_edges[j]
        if j < K - 1:
            k[(z > lo) & (z <= hi)] = j
        else:
            k[(z >= lo) & (z <= hi)] = j
    return k


def _ring_histogram(
    ax,
    counts: np.ndarray,
    expected: np.ndarray | None,
    color: str | list[str],
    subtitle: str,
    xlabel: str = "полоса (равная площадь)",
) -> None:
    K = counts.size
    x = np.arange(K)
    ax.bar(x, counts, color=color, alpha=0.55, edgecolor="0.4", linewidth=0.4)
    if expected is not None:
        ax.plot(x, expected, "k--", linewidth=1.0, alpha=0.7, label="ожидание")
        ax.legend(loc="upper right", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([str(i + 1) for i in range(K)], fontsize=7)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("точек")
    ax.set_title(subtitle, fontsize=9)
    ax.grid(True, axis="y", linestyle=":", alpha=0.45)


TAB_LABELS = (
    "Треугольник",
    "Диск",
    "Сфера",
    "Косинус",
)


class LabApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("МОИ ЛР3 — распределения точек и направлений")
        self.geometry("1120x720")
        self.minsize(700, 520)
        self.configure(bg="#f0f0f0")
        self._debounce_after: str | None = None
        self._mode = 0

        if "clam" in ttk.Style().theme_names():
            ttk.Style().theme_use("clam")

        self._init_vars()
        self._build_shell()
        self._bind_traces()
        self.after(60, self._redraw_current)

    def _init_vars(self) -> None:
        self.t1_seed = tk.StringVar(value=str(DEFAULT_SEED))
        self.t1_n = tk.StringVar(value=str(DEFAULT_N))
        self.t1_vis = tk.StringVar(value=str(DEFAULT_VIS))
        self.t1_v1 = [tk.StringVar(value="0"), tk.StringVar(value="0"), tk.StringVar(value="0")]
        self.t1_v2 = [tk.StringVar(value="3"), tk.StringVar(value="0"), tk.StringVar(value="0")]
        self.t1_v3 = [tk.StringVar(value="1"), tk.StringVar(value="2.5"), tk.StringVar(value="0")]

        self.t2_seed = tk.StringVar(value=str(DEFAULT_SEED))
        self.t2_n = tk.StringVar(value=str(DEFAULT_N))
        self.t2_vis = tk.StringVar(value=str(DEFAULT_VIS))
        self.t2_rc = tk.StringVar(value="1.5")
        self.t2_c = [tk.StringVar(value="0"), tk.StringVar(value="0"), tk.StringVar(value="1")]
        self.t2_nv = [tk.StringVar(value="0"), tk.StringVar(value="0"), tk.StringVar(value="1")]

        self.t3_seed = tk.StringVar(value=str(DEFAULT_SEED))
        self.t3_n = tk.StringVar(value=str(DEFAULT_N))
        self.t3_vis = tk.StringVar(value=str(DEFAULT_VIS))

        self.t4_seed = tk.StringVar(value=str(DEFAULT_SEED))
        self.t4_n = tk.StringVar(value=str(DEFAULT_N))
        self.t4_vis = tk.StringVar(value=str(DEFAULT_VIS))
        self.t4_custom = tk.BooleanVar(value=False)
        self.t4_nv = [tk.StringVar(value="0"), tk.StringVar(value="0"), tk.StringVar(value="1")]
        self.t4_c = [tk.StringVar(value="0"), tk.StringVar(value="0"), tk.StringVar(value="0")]
        self.t4_p1 = [tk.StringVar(value="1"), tk.StringVar(value="0"), tk.StringVar(value="0")]
        self.t4_p2 = [tk.StringVar(value="0"), tk.StringVar(value="1"), tk.StringVar(value="0")]

    def _build_shell(self) -> None:
        root = tk.Frame(self, bg="#f0f0f0")
        root.pack(fill=tk.BOTH, expand=True)

        top = ttk.Frame(root, padding=(10, 0, 10, 0))
        top.pack(fill=tk.X, pady=(10, 0))

        self._nb = ttk.Notebook(top)
        for _lab in TAB_LABELS:
            self._nb.add(ttk.Frame(self._nb), text=_lab)
        self._nb.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self._nb.bind("<<NotebookTabChanged>>", self._on_notebook_tab)
        ttk.Button(top, text="Обновить", width=12, command=self._redraw_current).pack(side=tk.RIGHT, padx=(8, 0))

        main = tk.Frame(root, bg="#e8e8ed")
        main.pack(fill=tk.BOTH, expand=True, padx=10, pady=(8, 10))

        self._controls_host = ttk.Frame(main, padding=(4, 8, 4, 4))
        self._controls_host.pack(fill=tk.X)

        self._hint = tk.Label(
            main,
            text="",
            bg="#e8e8ed",
            fg="#444444",
            font=("TkDefaultFont", 9),
            justify=tk.LEFT,
        )
        self._hint.pack(anchor=tk.W, padx=8, pady=(0, 4))

        plot_wrap = tk.Frame(main, bg="#e8e8ed")
        plot_wrap.pack(fill=tk.BOTH, expand=True)

        self.fig = Figure(figsize=(10.4, 5.5), dpi=100, layout="constrained")
        self.cv = FigureCanvasTkAgg(self.fig, master=plot_wrap)
        self.cv.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self._rebuild_controls()

    def _mode_hints(self) -> str:
        return (
            "", "", "", ""
        )[self._mode]

    def _on_notebook_tab(self, _event: object | None = None) -> None:
        try:
            i = self._nb.index(self._nb.select())
        except tk.TclError:
            return
        self._mode = int(i)
        self._rebuild_controls()
        self._hint.configure(text=self._mode_hints())
        self._redraw_current()

    def _sched(self) -> None:
        if self._debounce_after is not None:
            self.after_cancel(self._debounce_after)
        self._debounce_after = self.after(DEBOUNCE_MS, self._redraw_current)

    def _bind_traces(self) -> None:
        all_sv = (
            self.t1_seed,
            self.t1_n,
            self.t1_vis,
            *self.t1_v1,
            *self.t1_v2,
            *self.t1_v3,
            self.t2_seed,
            self.t2_n,
            self.t2_vis,
            self.t2_rc,
            *self.t2_c,
            *self.t2_nv,
            self.t3_seed,
            self.t3_n,
            self.t3_vis,
            self.t4_seed,
            self.t4_n,
            self.t4_vis,
            *self.t4_nv,
            *self.t4_c,
            *self.t4_p1,
            *self.t4_p2,
        )
        for v in all_sv:
            v.trace_add("write", lambda *_: self._sched())
        self.t4_custom.trace_add("write", lambda *_: self._sched())

    def _cell(self, r: int, c: int, text: str, var: tk.StringVar, w: int = 7) -> None:
        ttk.Label(self._pg, text=text).grid(row=r, column=2 * c, sticky=tk.E, padx=(0, 2), pady=1)
        ttk.Entry(self._pg, textvariable=var, width=w).grid(row=r, column=2 * c + 1, sticky=tk.W, pady=1)

    def _rebuild_controls(self) -> None:
        for w in self._controls_host.winfo_children():
            w.destroy()
        self._pg = ttk.Frame(self._controls_host)
        self._pg.pack(fill=tk.X)

        m = self._mode
        if m == 0:
            self._cell(0, 0, "seed", self.t1_seed)
            self._cell(0, 1, "N", self.t1_n)
            self._cell(0, 2, "pts", self.t1_vis)
            for j, lab in enumerate("xyz"):
                self._cell(1, j, f"V1{lab}", self.t1_v1[j])
                self._cell(2, j, f"V2{lab}", self.t1_v2[j])
                self._cell(3, j, f"V3{lab}", self.t1_v3[j])
        elif m == 1:
            self._cell(0, 0, "seed", self.t2_seed)
            self._cell(0, 1, "N", self.t2_n)
            self._cell(0, 2, "pts", self.t2_vis)
            self._cell(0, 3, "Rc", self.t2_rc)
            for j, lab in enumerate("xyz"):
                self._cell(1, j, f"C{lab}", self.t2_c[j])
                self._cell(2, j, f"N{lab}", self.t2_nv[j])
        elif m == 2:
            self._cell(0, 0, "seed", self.t3_seed)
            self._cell(0, 1, "N", self.t3_n)
            self._cell(0, 2, "pts", self.t3_vis)
        else:
            self._cell(0, 0, "seed", self.t4_seed)
            self._cell(0, 1, "N", self.t4_n)
            self._cell(0, 2, "pts", self.t4_vis)
            for j, lab in enumerate("xyz"):
                self._cell(1, j, f"n{lab}", self.t4_nv[j])
            ttk.Checkbutton(
                self._pg,
                text="Базис из C, P1, P2 (проекции на ⊥ n)",
                variable=self.t4_custom,
                command=self._sched,
            ).grid(row=2, column=0, columnspan=6, sticky=tk.W, pady=2)
            for j, lab in enumerate("xyz"):
                self._cell(3, j, f"C{lab}", self.t4_c[j])
                self._cell(4, j, f"P1{lab}", self.t4_p1[j])
                self._cell(5, j, f"P2{lab}", self.t4_p2[j])

        self._hint.configure(text=self._mode_hints())

    def _pi(self, s: str) -> int:
        v = int(s.strip())
        if v < 1:
            raise ValueError("N")
        return v

    def _pf(self, s: str) -> float:
        return float(s.strip().replace(",", "."))

    def _v3(self, xs: list[tk.StringVar]) -> np.ndarray:
        return np.array([self._pf(xs[0].get()), self._pf(xs[1].get()), self._pf(xs[2].get())], dtype=np.float64)

    def _redraw_current(self) -> None:
        self._debounce_after = None
        if self._mode == 0:
            self._d1()
        elif self._mode == 1:
            self._d2()
        elif self._mode == 2:
            self._d3()
        else:
            self._d4()

    def _clear_err(self, fig: Figure, msg: str) -> None:
        fig.clear()
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, msg, ha="center", va="center", transform=ax.transAxes, fontsize=10)

    def _d1(self) -> None:
        fig, cv = self.fig, self.cv
        try:
            seed = self._pi(self.t1_seed.get())
            n = self._pi(self.t1_n.get())
            nv = self._pi(self.t1_vis.get())
            V1, V2, V3 = self._v3(self.t1_v1), self._v3(self.t1_v2), self._v3(self.t1_v3)
            rng = np.random.default_rng(seed)
            P = sample_triangle(V1, V2, V3, rng, n)
        except Exception as e:
            self._clear_err(fig, str(e))
            cv.draw_idle()
            return

        G, T, B = _triangle_frame(V1, V2, V3)
        uv = np.array([_to_uv(V1, G, T, B), _to_uv(V2, G, T, B), _to_uv(V3, G, T, B)])
        rng_circ = np.random.default_rng(int(seed) + 2_046_281_337)
        circles = _sample_five_disjoint_disks_uv(uv, rng_circ)
        if circles is None:
            self._clear_err(fig, "не удалось разместить 5 непересекающихся окружностей")
            cv.draw_idle()
            return

        centers_uv = [c[0] for c in circles]
        radii = [c[1] for c in circles]
        DG = P - G
        up = np.dot(DG, T)
        vp = np.dot(DG, B)
        Puv = np.column_stack([up, vp])
        counts = np.zeros(TRIANGLE_DISJOINT_CIRCLES, dtype=np.float64)
        for i, (cuv, rad) in enumerate(circles):
            d = np.sqrt((Puv[:, 0] - cuv[0]) ** 2 + (Puv[:, 1] - cuv[1]) ** 2)
            counts[i] = float(np.sum(d <= rad + 1e-9))

        fig.clear()
        gs = fig.add_gridspec(1, 2, width_ratios=[2.55, 1])
        ax3 = fig.add_subplot(gs[0, 0], projection="3d")
        axf = fig.add_subplot(gs[0, 1])

        poly = Poly3DCollection([[V1, V2, V3]], alpha=0.12, facecolor="C0", edgecolor="0.4", linewidths=0.6)
        ax3.add_collection3d(poly)
        _plot_disjoint_circles_3d(
            ax3,
            G,
            T,
            B,
            centers_uv,
            radii,
            colors=[f"C{i % 10}" for i in range(TRIANGLE_DISJOINT_CIRCLES)],
        )
        ix = _subidx(rng, n, nv)
        ax3.scatter(P[ix, 0], P[ix, 1], P[ix, 2], s=1, alpha=0.28, c="C3", depthshade=True, linewidths=0)
        ax3.plot([V1[0], V2[0], V3[0], V1[0]], [V1[1], V2[1], V3[1], V1[1]], [V1[2], V2[2], V3[2], V1[2]], "k-", lw=0.6, alpha=0.5)

        lim = np.array([P[ix].min(axis=0), P[ix].max(axis=0)])
        pad = 0.15 * (lim[1] - lim[0] + 1e-6)
        ax3.set_xlim(lim[0, 0] - pad[0], lim[1, 0] + pad[0])
        ax3.set_ylim(lim[0, 1] - pad[1], lim[1, 1] + pad[1])
        ax3.set_zlim(lim[0, 2] - pad[2], lim[1, 2] + pad[2])
        ax3.set_box_aspect((1, 1, 1))
        ax3.set_xticks([])
        ax3.set_yticks([])
        ax3.set_zticks([])
        ax3.view_init(elev=22, azim=42)
        ax3.set_title("")

        _ring_histogram(
            axf,
            counts,
            None,
            [f"C{i % 10}" for i in range(TRIANGLE_DISJOINT_CIRCLES)],
            "треугольник: 5 окружностей одинакового радиуса",
            xlabel="окружность №",
        )

        cv.draw_idle()

    def _d2(self) -> None:
        fig, cv = self.fig, self.cv
        try:
            seed = self._pi(self.t2_seed.get())
            n = self._pi(self.t2_n.get())
            nv = self._pi(self.t2_vis.get())
            Rc = self._pf(self.t2_rc.get())
            if Rc <= 0:
                raise ValueError("Rc")
            C = self._v3(self.t2_c)
            N = self._v3(self.t2_nv)
            if np.linalg.norm(N) < 1e-14:
                raise ValueError("N")
            rng = np.random.default_rng(seed)
            P, T, B, n_unit, r, phi = sample_uniform_disk(C, Rc, N, rng, n)
        except Exception as e:
            self._clear_err(fig, str(e))
            cv.draw_idle()
            return

        K = RING_K
        r_edges = _equal_area_radial_edges(Rc, K)
        bins = _bin_radial(r, r_edges)
        counts = np.bincount(bins, minlength=K).astype(np.float64)
        expected = np.full(K, n / K)

        fig.clear()
        gs = fig.add_gridspec(1, 2, width_ratios=[2.55, 1])
        ax3 = fig.add_subplot(gs[0, 0], projection="3d")
        axf = fig.add_subplot(gs[0, 1])

        th = np.linspace(0, 2 * np.pi, 72)
        rr = np.linspace(0, Rc, 24)
        TH, RR = np.meshgrid(th, rr)
        X = C[0] + RR * (np.cos(TH) * T[0] + np.sin(TH) * B[0])
        Y = C[1] + RR * (np.cos(TH) * T[1] + np.sin(TH) * B[1])
        Z = C[2] + RR * (np.cos(TH) * T[2] + np.sin(TH) * B[2])
        ax3.plot_surface(X, Y, Z, color="C0", alpha=0.11, linewidth=0, antialiased=True, shade=False)
        ax3.plot(
            C[0] + Rc * np.cos(th) * T[0] + Rc * np.sin(th) * B[0],
            C[1] + Rc * np.cos(th) * T[1] + Rc * np.sin(th) * B[1],
            C[2] + Rc * np.cos(th) * T[2] + Rc * np.sin(th) * B[2],
            "k-",
            lw=0.7,
            alpha=0.55,
        )
        _plot_plane_circles_3d(ax3, C, T, B, r_edges[1:-1])

        ix = _subidx(rng, n, nv)
        ax3.scatter(P[ix, 0], P[ix, 1], P[ix, 2], s=1, alpha=0.3, c="C1", depthshade=True, linewidths=0)

        span = max(Rc * 2.2, float(np.ptp(P[ix], axis=0).max()) + 0.1)
        ax3.set_xlim(C[0] - span / 2, C[0] + span / 2)
        ax3.set_ylim(C[1] - span / 2, C[1] + span / 2)
        ax3.set_zlim(C[2] - span / 2, C[2] + span / 2)
        ax3.set_box_aspect((1, 1, 1))
        ax3.set_xticks([])
        ax3.set_yticks([])
        ax3.set_zticks([])
        ax3.view_init(elev=28, azim=-55)

        _ring_histogram(axf, counts, expected, "C1", "диск: кольца равной площади")

        cv.draw_idle()

    def _d3(self) -> None:
        fig, cv = self.fig, self.cv
        try:
            seed = self._pi(self.t3_seed.get())
            n = self._pi(self.t3_n.get())
            nv = self._pi(self.t3_vis.get())
            rng = np.random.default_rng(seed)
            P, z, phi = sample_uniform_sphere(rng, n)
        except Exception as e:
            self._clear_err(fig, str(e))
            cv.draw_idle()
            return

        K = RING_K
        z_bins = _bin_sphere_z_equal_area(z, K)
        counts = np.bincount(z_bins, minlength=K).astype(np.float64)
        expected = np.full(K, n / K)
        z_levels = np.linspace(-1.0, 1.0, K + 1)[1:-1]

        fig.clear()
        gs = fig.add_gridspec(1, 2, width_ratios=[2.55, 1])
        ax3 = fig.add_subplot(gs[0, 0], projection="3d")
        axf = fig.add_subplot(gs[0, 1])

        u = np.linspace(0, 2 * np.pi, 36)
        v = np.linspace(0, np.pi, 18)
        U, V = np.meshgrid(u, v)
        xs = np.cos(U) * np.sin(V)
        ys = np.sin(U) * np.sin(V)
        zs = np.cos(V)
        ax3.plot_surface(xs, ys, zs, color="0.75", alpha=0.14, linewidth=0, antialiased=True, shade=False)
        _plot_latitudes_unit_sphere(ax3, z_levels)

        ix = _subidx(rng, n, nv)
        ax3.scatter(P[ix, 0], P[ix, 1], P[ix, 2], s=1, alpha=0.28, c="C2", depthshade=True, linewidths=0)
        ax3.set_xlim(-1.05, 1.05)
        ax3.set_ylim(-1.05, 1.05)
        ax3.set_zlim(-1.05, 1.05)
        ax3.set_box_aspect((1, 1, 1))
        ax3.set_xticks([])
        ax3.set_yticks([])
        ax3.set_zticks([])
        ax3.view_init(elev=18, azim=35)

        _ring_histogram(axf, counts, expected, "C2", "сфера: полосы равной площади по z")

        cv.draw_idle()

    def _d4(self) -> None:
        fig, cv = self.fig, self.cv
        try:
            seed = self._pi(self.t4_seed.get())
            n = self._pi(self.t4_n.get())
            nv = self._pi(self.t4_vis.get())
            N = self._v3(self.t4_nv)
            if np.linalg.norm(N) < 1e-14:
                raise ValueError("N")
            rng = np.random.default_rng(seed)
            if self.t4_custom.get():
                Cb = self._v3(self.t4_c)
                P1 = self._v3(self.t4_p1)
                P2 = self._v3(self.t4_p2)
                T, B, n_unit = tangent_basis_from_c_p1_p2(N, Cb, P1, P2)
            else:
                T, B, n_unit = orthonormal_basis_TBN(N)
            dir_vec, cos_theta = sample_cosine_hemisphere(rng, n, T, B, n_unit)
        except Exception as e:
            self._clear_err(fig, str(e))
            cv.draw_idle()
            return

        K = RING_K
        hb = _bin_hemisphere_equal_area(cos_theta, K)
        counts = np.bincount(hb, minlength=K).astype(np.float64)
        expected = cosine_equal_area_expected_counts(n, K)
        z_levels = (1.0 - np.linspace(0.0, 1.0, K + 1))[1:-1]

        fig.clear()
        gs = fig.add_gridspec(1, 2, width_ratios=[2.55, 1])
        ax3 = fig.add_subplot(gs[0, 0], projection="3d")
        axf = fig.add_subplot(gs[0, 1])

        u = np.linspace(0, 2 * np.pi, 40)
        vang = np.linspace(0, np.pi / 2, 16)
        U, VV = np.meshgrid(u, vang)
        xh = np.cos(U) * np.sin(VV)
        yh = np.sin(U) * np.sin(VV)
        zh = np.cos(VV)
        gx = xh * T[0] + yh * B[0] + zh * n_unit[0]
        gy = xh * T[1] + yh * B[1] + zh * n_unit[1]
        gz = xh * T[2] + yh * B[2] + zh * n_unit[2]
        ax3.plot_surface(gx, gy, gz, color="0.65", alpha=0.1, linewidth=0, antialiased=True, shade=False)
        _plot_latitudes_hemisphere_world(ax3, T, B, n_unit, z_levels)

        th = np.linspace(0, 2 * np.pi, 64)
        rd = np.cos(np.pi / 4)
        xd = rd * (np.cos(th) * T[0] + np.sin(th) * B[0])
        yd = rd * (np.cos(th) * T[1] + np.sin(th) * B[1])
        zd = rd * (np.cos(th) * T[2] + np.sin(th) * B[2])
        ax3.plot(xd, yd, zd, color="0.5", lw=0.6, alpha=0.45)

        ix = _subidx(rng, n, nv)
        ax3.scatter(dir_vec[ix, 0], dir_vec[ix, 1], dir_vec[ix, 2], s=1, alpha=0.3, c="C4", depthshade=True, linewidths=0)
        ax3.quiver(
            0,
            0,
            0,
            float(n_unit[0]),
            float(n_unit[1]),
            float(n_unit[2]),
            length=1.0,
            color="k",
            arrow_length_ratio=0.06,
            linewidth=0.8,
        )

        ax3.set_xlim(-1.05, 1.05)
        ax3.set_ylim(-1.05, 1.05)
        ax3.set_zlim(-1.05, 1.05)
        ax3.set_box_aspect((1, 1, 1))
        ax3.set_xticks([])
        ax3.set_yticks([])
        ax3.set_zticks([])
        ax3.view_init(elev=16, azim=28)

        _ring_histogram(axf, counts, expected, "C4", "косинус: полосы равной площади на полусфере")

        cv.draw_idle()


if __name__ == "__main__":
    LabApp().mainloop()
