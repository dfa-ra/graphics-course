import math

import math_utils as mu

INV_PI = 1.0 / math.pi
R_SQ_EPS = 1e-20

CANVAS_BG = "#e8e8e8"
OUTSIDE_FILL = "#d4d4d4"
OUTLINE_COLOR = "#606060"


def _clamp01(x):
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def _unpack_light(light):

    pos = (light[0], light[1], light[2])
    I0 = (light[3], light[4], light[5])
    if len(light) >= 9:
        O_axis = mu.normalize_safe((light[6], light[7], light[8]))
    else:
        O_axis = None
    return pos, I0, O_axis


def _orient_normal_toward_observer(N, P0, P1, P2, O_obs):
    ctr = mu.scale(mu.add(mu.add(P0, P1), P2), 1.0 / 3.0)
    if mu.dot(N, mu.sub(O_obs, ctr)) < 0.0:
        return mu.scale(N, -1.0)
    return N


def _scene(scene):
    P0 = (scene["ax"], scene["ay"], scene["az"])
    P1 = (scene["bx"], scene["by"], scene["bz"])
    P2 = (scene["cx"], scene["cy"], scene["cz"])
    O_obs = (scene["ox"], scene["oy"], scene["oz"])
    K = (scene["surf_r"], scene["surf_g"], scene["surf_b"])
    N_raw = mu.triangle_normal_P0P1P2(P0, P1, P2)
    N = _orient_normal_toward_observer(N_raw, P0, P1, P2, O_obs)
    len_e1 = mu.length(mu.sub(P1, P0))
    len_e2 = mu.length(mu.sub(P2, P0))
    return P0, P1, P2, O_obs, K, N, len_e1, len_e2


def _inside_local_xy(x, y, len_e1, len_e2):
    if len_e1 < 1e-15 or len_e2 < 1e-15:
        return False
    if x < 0.0 or y < 0.0:
        return False
    return (x / len_e1 + y / len_e2) <= 1.0 + 1e-12


def radiance_rgb(P_T, N, O_obs, K, kd, ks, ke, lights):
    v_raw = mu.sub(O_obs, P_T)
    v = mu.normalize_safe(v_raw)
    if v is None:
        return (0.0, 0.0, 0.0)

    sum_r, sum_g, sum_b = 0.0, 0.0, 0.0

    for light in lights:
        P_L, I0, O_axis = _unpack_light(light)

        s_vec = mu.sub(P_T, P_L)
        R_sq = mu.dot(s_vec, s_vec)
        if R_sq < R_SQ_EPS:
            continue
        R = math.sqrt(R_sq)

        to_light = mu.sub(P_L, P_T)
        cos_alpha = max(0.0, mu.dot(to_light, N) / R)

        if O_axis is None:
            cos_theta = 1.0
        else:
            cos_theta = max(0.0, mu.dot(s_vec, O_axis) / R)

        s_dir = mu.normalize_safe(to_light)
        if s_dir is None:
            continue

        h_raw = mu.add(v, s_dir)
        h = mu.normalize_safe(h_raw)
        if h is None:
            nh = 0.0
        else:
            nh = mu.dot(h, N)

        br = kd + ks * (nh ** ke)

        for c, I0c in enumerate(I0):
            Ic = I0c * cos_theta
            Ec = Ic * cos_alpha / R_sq
            fc = K[c] * br
            Lc = Ec * fc
            if c == 0:
                sum_r += Lc
            elif c == 1:
                sum_g += Lc
            else:
                sum_b += Lc

    return _clamp01(sum_r), _clamp01(sum_g), _clamp01(sum_b)


def _rgb_hex(r, g, b):
    ir = int(_clamp01(r) * 255.0 + 0.5)
    ig = int(_clamp01(g) * 255.0 + 0.5)
    ib = int(_clamp01(b) * 255.0 + 0.5)
    return "#%02x%02x%02x" % (ir, ig, ib)


def render_triangle(canvas, scene):
    P0, P1, P2, O_obs, K, N, len_e1, len_e2 = _scene(scene)
    kd = float(scene["kd"])
    ks = float(scene["ks"])
    ke = float(scene["p"])
    lights = scene["lights"]
    gw, gh = int(scene["grid_w"]), int(scene["grid_h"])
    pxsz = int(scene["pixel_size"])
    show_outside = bool(scene.get("show_outside_background", False))

    wpx, hpx = gw * pxsz, gh * pxsz
    canvas.config(width=wpx, height=hpx, bg=CANVAS_BG)
    canvas.delete("all")

    for j in range(gh):
        a = (j + 0.5) / gh
        for i in range(gw):
            b = (i + 0.5) / gw
            x_loc = a * len_e1
            y_loc = b * len_e2
            outside = not _inside_local_xy(x_loc, y_loc, len_e1, len_e2)
            if outside:
                if not show_outside:
                    continue
                fill = OUTSIDE_FILL
            else:
                P_T = mu.point_P_T_from_local_xy(P0, P1, P2, x_loc, y_loc)
                r, g, b = radiance_rgb(P_T, N, O_obs, K, kd, ks, ke, lights)
                fill = _rgb_hex(r, g, b)
            x0, y0 = i * pxsz, j * pxsz
            canvas.create_rectangle(x0, y0, x0 + pxsz, y0 + pxsz, fill=fill, outline="", width=0)

    canvas.create_polygon(0, 0, 0, hpx, wpx, 0, fill="", outline=OUTLINE_COLOR, width=1)


def build_report_results(scene, sampled_points):
    P0, P1, P2, O_obs, K, N, len_e1, len_e2 = _scene(scene)
    kd = float(scene["kd"])
    ks = float(scene["ks"])
    ke = float(scene["p"])
    lights = scene["lights"]
    results = []

    for k, (a_frac, b_frac) in enumerate(sampled_points):
        a_frac = float(a_frac)
        b_frac = float(b_frac)
        x_loc = a_frac * len_e1
        y_loc = b_frac * len_e2
        inside = _inside_local_xy(x_loc, y_loc, len_e1, len_e2)
        P_T = mu.point_P_T_from_local_xy(P0, P1, P2, x_loc, y_loc)

        v_raw = mu.sub(O_obs, P_T)
        V_obs = mu.normalize_safe(v_raw)
        V_out = V_obs if V_obs is not None else (0.0, 0.0, 0.0)

        per_light = []
        sum_r, sum_g, sum_b = 0.0, 0.0, 0.0

        if inside and V_obs is not None:
            for li, light in enumerate(lights):
                P_L, I0, O_axis = _unpack_light(light)

                s_vec = mu.sub(P_T, P_L)
                R_sq = mu.dot(s_vec, s_vec)
                if R_sq < R_SQ_EPS:
                    continue
                R = math.sqrt(R_sq)

                cos_alpha_sheet = mu.dot(s_vec, N) / R
                to_light = mu.sub(P_L, P_T)
                cos_alpha = max(0.0, mu.dot(to_light, N) / R)

                if O_axis is None:
                    cos_theta = 1.0
                else:
                    cos_theta = max(0.0, mu.dot(s_vec, O_axis) / R)

                Ir = I0[0] * cos_theta
                Ig = I0[1] * cos_theta
                Ib = I0[2] * cos_theta

                Er = Ir * cos_alpha / R_sq
                Eg = Ig * cos_alpha / R_sq
                Eb = Ib * cos_alpha / R_sq

                s_dir = mu.normalize_safe(to_light)
                if s_dir is None:
                    continue

                h_raw = mu.add(V_obs, s_dir)
                h = mu.normalize_safe(h_raw)
                if h is None:
                    nh = 0.0
                    H_store = (0.0, 0.0, 0.0)
                else:
                    nh = mu.dot(h, N)
                    H_store = h

                br = kd + ks * (nh ** ke)
                fr = K[0] * br
                fg = K[1] * br
                fb = K[2] * br

                Lir = Er * fr
                Lig = Eg * fg
                Lib = Eb * fb

                sum_r += Lir
                sum_g += Lig
                sum_b += Lib

                per_light.append(
                    {
                        "i": li + 1,
                        "position": P_L,
                        "I0_rgb": I0,
                        "O_axis": O_axis,
                        "s_vec": s_vec,
                        "R": R,
                        "R_sq": R_sq,
                        "cos_theta": cos_theta,
                        "I_rgb_after_cos": (Ir, Ig, Ib),
                        "cos_alpha_sheet": cos_alpha_sheet,
                        "cos_alpha": cos_alpha,
                        "E_rgb": (Er, Eg, Eb),
                        "s_dir_to_light": s_dir,
                        "H": H_store,
                        "NdotH": nh,
                        "brdf_bracket": br,
                        "f_rgb": (fr, fg, fb),
                        "L_contrib_rgb": (Lir, Lig, Lib),
                        "L_contrib_with_pi": (
                            INV_PI * Lir,
                            INV_PI * Lig,
                            INV_PI * Lib,
                        ),
                    }
                )

        results.append(
            {
                "idx": k,
                "a_frac": a_frac,
                "b_frac": b_frac,
                "x_local": x_loc,
                "y_local": y_loc,
                "inside": inside,
                "P": P_T,
                "N": N,
                "V": V_out,
                "observer_dir_ok": V_obs is not None,
                "lights": per_light,
                "L_rgb": (_clamp01(sum_r), _clamp01(sum_g), _clamp01(sum_b)),
            }
        )

    return results


def _fmt3(v):
    return "(%.6f, %.6f, %.6f)" % (v[0], v[1], v[2])


def print_report_values(scene, sampled_points, results):
    mode = scene.get("report_mode", "verbose")
    P0, P1, P2, O_obs, K, N, len_e1, len_e2 = _scene(scene)
    e_ab = mu.sub(P1, P0)
    e_ac = mu.sub(P2, P0)

    def blk(title):
        print()
        print(title)

    if mode == "verbose":
        blk("=== ВХОДНЫЕ ДАННЫЕ (методичка) ===")
        print("P0 = A = " + _fmt3(P0))
        print("P1 = B = " + _fmt3(P1))
        print("P2 = C = " + _fmt3(P2))
        print("Источники: строка «x y z I0r I0g I0b» или «... I0r I0g I0b Ox Oy Oz» (ось O, |O|→1).")
        print("  Если оси нет: cos(theta)=1 (изотропия).")
        for i, Ln in enumerate(scene["lights"], start=1):
            pos, I0, oa = _unpack_light(Ln)
            extra = "  O=" + _fmt3(oa) if oa is not None else "  (без оси, cos theta=1)"
            print("  источник %d: P_L=%s, I0=%s%s" % (i, _fmt3(pos), _fmt3(I0), extra))
        print("Наблюдатель V_pos = " + _fmt3(O_obs))
        print("K(RGB) поверхности = " + _fmt3(K))
        print("kd = %.6f, ks = %.6f, k_e (p) = %.6f" % (scene["kd"], scene["ks"], scene["p"]))
        print("Сетка: %d x %d, пиксель (квадрат) = %d" % (scene["grid_w"], scene["grid_h"], scene["pixel_size"]))

        blk("=== ПРОМЕЖУТОЧНЫЕ ВЕКТОРЫ ===")
        print("P1 - P0 = " + _fmt3(e_ab))
        print("P2 - P0 = " + _fmt3(e_ac))
        print("|P1-P0| = %.6f, |P2-P0| = %.6f" % (len_e1, len_e2))
        print("N = нормализованное (P2-P0)x(P1-P0), ориентированное к наблюдателю = " + _fmt3(N))
        print("Локальные x,y — смещения вдоль (P1-P0)/|.| и (P2-P0)/|.| от P0; внутри: x>=0, y>=0, x/|P1-P0|+y/|P2-P0|<=1")
        for sp, res in zip(sampled_points, results):
            print(
                "  доли (a,b)=(%.6f,%.6f) -> x=%.6f, y=%.6f -> P_T = %s, внутри: %s"
                % (sp[0], sp[1], res["x_local"], res["y_local"], _fmt3(res["P"]), res["inside"])
            )

        blk("=== КОНТРОЛЬНЫЕ ТОЧКИ (по формулам) ===")
        for res in results:
            print()
            print("--- Точка #%d ---" % res["idx"])
            print(
                "a = %.6f, b = %.6f (доли); x = %.6f, y = %.6f (вдоль рёбер от P0)"
                % (res["a_frac"], res["b_frac"], res["x_local"], res["y_local"])
            )
            print("P_T = " + _fmt3(res["P"]))
            print("N = " + _fmt3(res["N"]))
            print("v = normalize(V_pos - P_T) = " + _fmt3(res["V"]))
            if not res.get("observer_dir_ok", True):
                print("  (V_pos - P_T ≈ 0: v не определяется)")
            if not res["inside"]:
                print("(вне треугольника по условию на x,y)")
                continue
            if not res.get("observer_dir_ok", True):
                continue
            for Ld in res["lights"]:
                print()
                print("  источник i = %d" % Ld["i"])
                print("  s = P_T - P_L = " + _fmt3(Ld["s_vec"]))
                print("  R^2 = |s|^2 = %.6f, R = %.6f" % (Ld["R_sq"], Ld["R"]))
                print(
                    "  (s·N)/|s| (s=P_T-P_L) = %.6f; cos(alpha) для E = max(0,((P_L-P_T)·N)/R) = %.6f"
                    % (Ld["cos_alpha_sheet"], Ld["cos_alpha"])
                )
                if Ld["O_axis"] is not None:
                    print("  cos(theta) = (s·O)/|s| = %.6f" % Ld["cos_theta"])
                else:
                    print("  cos(theta) = 1 (ось источника не задана)")
                print("  I(RGB) = I0*cos(theta) = " + _fmt3(Ld["I_rgb_after_cos"]))
                print("  E(RGB) = I*cos(alpha)/R^2 = " + _fmt3(Ld["E_rgb"]))
                print("  направление к свету (ед.) s_i = " + _fmt3(Ld["s_dir_to_light"]))
                print("  h = normalize(v + s_i) = " + _fmt3(Ld["H"]))
                print("  (h·N) = %.6f" % Ld["NdotH"])
                print("  f = K*(kd + ks*(h·N)^k_e); скаляр в скобках = %.6f" % Ld["brdf_bracket"])
                print("  f(RGB) = " + _fmt3(Ld["f_rgb"]))
                print("  L_i = E*f = " + _fmt3(Ld["L_contrib_rgb"]))
                print("  (1/pi)*E*f = " + _fmt3(Ld["L_contrib_with_pi"]))
            print("  L(RGB) (clamp [0,1]) = " + _fmt3(res["L_rgb"]))

    else:
        blk("=== КОМПАКТНЫЙ ВЫВОД ===")
        print("N = " + _fmt3(N))
        print("V_pos = " + _fmt3(O_obs))
        print("kd=%.6f ks=%.6f k_e=%.6f  K=%s" % (scene["kd"], scene["ks"], scene["p"], _fmt3(K)))
        print("Сетка %dx%d  px=%d" % (scene["grid_w"], scene["grid_h"], scene["pixel_size"]))
        print()
        for res in results:
            print("P%d = %s" % (res["idx"], _fmt3(res["P"])))
            print("v%d = %s" % (res["idx"], _fmt3(res["V"])))
        print()
        print("# idx\ta\tb\tx\ty\tPx\tPy\tPz\tvx\tvy\tvz\tLr\tLg\tLb")
        for res in results:
            P, V, Lc = res["P"], res["V"], res["L_rgb"]
            print(
                "%d\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f"
                % (
                    res["idx"],
                    res["a_frac"],
                    res["b_frac"],
                    res["x_local"],
                    res["y_local"],
                    P[0],
                    P[1],
                    P[2],
                    V[0],
                    V[1],
                    V[2],
                    Lc[0],
                    Lc[1],
                    Lc[2],
                )
            )
        print()
        for res in results:
            k = res["idx"]
            if not res["inside"]:
                print("P%d: вне треугольника" % k)
                continue
            for Ld in res["lights"]:
                print("E%d(P%d) = %s" % (Ld["i"], k, _fmt3(Ld["E_rgb"])))
            print("L(P%d) = %s" % (k, _fmt3(res["L_rgb"])))
        print()
        for res in results:
            print(
                "a%d=%.6f, b%d=%.6f  |  x%d=%.6f, y%d=%.6f"
                % (
                    res["idx"],
                    res["a_frac"],
                    res["idx"],
                    res["b_frac"],
                    res["idx"],
                    res["x_local"],
                    res["idx"],
                    res["y_local"],
                )
            )
