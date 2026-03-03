# slab_geometries.py
# Geometrías 3D para slab y overriding plate (Underworld / UWGeodynamics)
#
# Diseño:
# - Shapes por unión/intersección de HalfSpaces (dot<=0).
# - Slab "single-layer" (espesor total) por banda en Y:
#     * dip constante: slab_band_shape
#     * dip(x) suave: slab_band_shape_xvary (segmentación en X con tan(dip) derivada de dz/dx)
#     * transiciones en Y: slab_transition_shape / add_transition_y_xvary
# - Overriding: cuña acoplada al slab con espesor capado (max_thickness)
#
# Nota clave para robustez (anti-gaps):
# - En x-vary, cada subsegmento usa tan = -(Δz/Δx) entre endpoints integrados (continuidad exacta).
# - Opcionalmente se puede "cortar" el slab a una cota z_cut (ej. -660 km) para que TODOS los segmentos
#   terminen consistente (evita agujeros por distintos x_end efectivos entre bandas).

import numpy as np


# -----------------------------------------------------------------------------
# Utilidades
# -----------------------------------------------------------------------------
def _u(GEO):
    return GEO.UnitRegistry


def _as_qty(q, unit):
    """Si q no tiene unidades, asume `unit`."""
    return q if hasattr(q, "units") else q * unit


def _smoothstep01(t: float) -> float:
    if t <= 0.0:
        return 0.0
    if t >= 1.0:
        return 1.0
    return t * t * (3.0 - 2.0 * t)


def dip_piecewise_smooth(x, x_nodes, dip_nodes) -> float:
    """
    Dip suave por tramos entre nodos (x_nodes, dip_nodes) usando smoothstep.
    Soporta float o Quantity en x y x_nodes.
    """
    if len(x_nodes) != len(dip_nodes):
        raise ValueError("x_nodes y dip_nodes deben tener la misma longitud.")
    if len(x_nodes) < 2:
        raise ValueError("x_nodes debe tener al menos 2 nodos.")

    def _val(v):
        return float(v.magnitude) if hasattr(v, "magnitude") else float(v)

    xx = _val(x)
    xs = [_val(v) for v in x_nodes]
    ds = [float(d) for d in dip_nodes]

    if xx <= xs[0]:
        return ds[0]
    if xx >= xs[-1]:
        return ds[-1]

    for i in range(len(xs) - 1):
        xa, xb = xs[i], xs[i + 1]
        if xa <= xx <= xb:
            t = (xx - xa) / (xb - xa)
            s = _smoothstep01(t)
            return ds[i] + (ds[i + 1] - ds[i]) * s

    return ds[-1]


def integrate_slab_top_z(*, x0, z0, x1, x_nodes, dip_nodes, n=4000):
    """
    Integra z(x) desde x0 a x1 usando: dz = -tan(dip(x))*dx
    Soporta float o Quantity.
    """
    use_units = hasattr(x0, "units")
    if use_units:
        ux = x0.units
        uz = z0.units
        x0v = float(x0.magnitude)
        x1v = float(x1.magnitude)
        z0v = float(z0.magnitude)
    else:
        ux = uz = None
        x0v = float(x0)
        x1v = float(x1)
        z0v = float(z0)

    xs = np.linspace(x0v, x1v, int(n) + 1)
    zs = np.zeros_like(xs)
    zs[0] = z0v

    for i in range(len(xs) - 1):
        xa = xs[i]
        xb = xs[i + 1]
        dx = xb - xa
        dip = dip_piecewise_smooth(xa, x_nodes, dip_nodes)
        zs[i + 1] = zs[i] - np.tan(np.deg2rad(dip)) * dx

    if use_units:
        xs = xs * ux
        zs = zs * uz

    return xs, zs


def find_x_at_depth(*, x0, z0, x1, x_nodes, dip_nodes, z_target, n=8000, return_profile=False):
    """
    Encuentra el primer x donde z_top(x) <= z_target (z_target negativo).
    """
    use_units = hasattr(x0, "units")
    if use_units:
        if not hasattr(z_target, "units"):
            z_target = z_target * z0.units

    xs, zs = integrate_slab_top_z(x0=x0, z0=z0, x1=x1, x_nodes=x_nodes, dip_nodes=dip_nodes, n=n)

    zmag = zs.magnitude if hasattr(zs, "magnitude") else zs
    zt = z_target.magnitude if hasattr(z_target, "magnitude") else z_target
    idx = np.where(zmag <= zt)[0]
    if idx.size == 0:
        return (None, xs, zs) if return_profile else None

    i = int(idx[0])
    if i == 0:
        x_hit = xs[0]
        return (x_hit, xs, zs) if return_profile else x_hit

    z0m, z1m = zmag[i - 1], zmag[i]
    x0v, x1v = xs[i - 1], xs[i]

    if z1m == z0m:
        x_hit = x1v
    else:
        t = (zt - z0m) / (z1m - z0m)
        x_hit = x0v + (x1v - x0v) * t

    return (x_hit, xs, zs) if return_profile else x_hit


def debug_slab_profile(*, x_break, x_nodes, dip_nodes, x_end, dip_label="slab", z0=None, n=6000,
                       checkpoints=None, target_depth_km=660.0):
    z0 = 0.0 * x_break.units if (z0 is None and hasattr(x_break, "units")) else (0.0 if z0 is None else z0)
    xs, zs = integrate_slab_top_z(x0=x_break, z0=z0, x1=x_end, x_nodes=x_nodes, dip_nodes=dip_nodes, n=n)

    def _tofloat(v):
        return float(v.magnitude) if hasattr(v, "magnitude") else float(v)

    if checkpoints is None:
        checkpoints = [x_break] + list(x_nodes) + [x_end]

    print(f"\n=== DEBUG {dip_label}: slab top ===")
    xs_f = np.array([_tofloat(v) for v in xs])
    for xc in checkpoints:
        xc_f = _tofloat(xc)
        idx = int(np.argmin(np.abs(xs_f - xc_f)))
        dipc = dip_piecewise_smooth(xc, x_nodes, dip_nodes)
        zc = zs[idx]
        print(f"x={xs[idx]:>10} | z={zc:>10} | dip~{dipc:5.1f}°")

    z_end = zs[-1]
    z_end_f = _tofloat(z_end)
    print("\n=== RESULTADO FINAL ===")
    print(f"En x_end={x_end}, z_top={z_end} (target: ~{-target_depth_km} km)")
    if z_end_f > -target_depth_km:
        print(f"Te falta bajar ~{abs(-target_depth_km - z_end_f):.1f} km antes del borde.")
    else:
        print(f"Llegás/pasás 660 km por ~{abs(z_end_f + target_depth_km):.1f} km.")


# -----------------------------------------------------------------------------
# Primitivas geométricas
# -----------------------------------------------------------------------------
def horizontal_layer(GEO, *, x_left, x_right, y_back, y_front, z_top, z_bottom):
    """
    Bloque horizontal:
      X: [x_left, x_right]
      Y: [y_front, y_back]
      Z: [z_bottom, z_top]
    """
    y_mid = 0.5 * (y_back + y_front)
    x_mid = 0.5 * (x_left + x_right)
    z_mid = 0.5 * (z_top + z_bottom)

    top = GEO.shapes.HalfSpace(normal=(0., 0.,  1.), origin=(x_mid, y_mid, z_top))     # z <= z_top
    bot = GEO.shapes.HalfSpace(normal=(0., 0., -1.), origin=(x_mid, y_mid, z_bottom))  # z >= z_bottom

    lft = GEO.shapes.HalfSpace(normal=(-1., 0., 0.), origin=(x_left,  y_mid, z_mid))   # x >= x_left
    rgt = GEO.shapes.HalfSpace(normal=( 1., 0., 0.), origin=(x_right, y_mid, z_mid))   # x <= x_right

    fr  = GEO.shapes.HalfSpace(normal=(0., -1., 0.), origin=(x_mid, y_front, z_mid))   # y >= y_front
    bk  = GEO.shapes.HalfSpace(normal=(0.,  1., 0.), origin=(x_mid, y_back,  z_mid))   # y <= y_back

    return top & bot & lft & rgt & fr & bk


def _layer_shape_simple(GEO, *, y_back, y_front, z_top, z_bottom, x_break, x_end, dip_deg):
    """
    Capa simple:
      - tramo horizontal: x in [0, x_break], z in [z_bottom, z_top]
      - tramo dip:        x in [x_break, x_end], top/bottom con dip_deg

    OJO: para evitar artefactos, x_end debe ser consistente entre bandas si no vas a usar z_cut.
    """
    y_mid = 0.5 * (y_back + y_front)
    x_mid_h = 0.5 * x_break
    x_mid_d = 0.5 * (x_break + x_end)
    z_mid = 0.5 * (z_top + z_bottom)

    # horizontal
    top_h = GEO.shapes.HalfSpace(normal=(0., 0.,  1.), origin=(x_mid_h, y_mid, z_top))
    bot_h = GEO.shapes.HalfSpace(normal=(0., 0., -1.), origin=(x_mid_h, y_mid, z_bottom))
    lft_h = GEO.shapes.HalfSpace(normal=(-1., 0., 0.), origin=(0.0 * x_break, y_mid, z_mid))  # x >= 0
    rgt_h = GEO.shapes.HalfSpace(normal=( 1., 0., 0.), origin=(x_break, y_mid, z_mid))        # x <= x_break
    fr_h  = GEO.shapes.HalfSpace(normal=(0., -1., 0.), origin=(x_mid_h, y_front, z_mid))
    bk_h  = GEO.shapes.HalfSpace(normal=(0.,  1., 0.), origin=(x_mid_h, y_back,  z_mid))
    layer_h = top_h & bot_h & lft_h & rgt_h & fr_h & bk_h

    if dip_deg is None:
        return layer_h

    tan_dip = float(np.tan(np.deg2rad(float(dip_deg))))

    top_d = GEO.shapes.HalfSpace(normal=( tan_dip, 0.,  1.), origin=(x_break, y_mid, z_top))
    bot_d = GEO.shapes.HalfSpace(normal=(-tan_dip, 0., -1.), origin=(x_break, y_mid, z_bottom))
    lft_d = GEO.shapes.HalfSpace(normal=(-1., 0., 0.), origin=(x_break, y_mid, z_mid))        # x >= x_break
    rgt_d = GEO.shapes.HalfSpace(normal=( 1., 0., 0.), origin=(x_end,   y_mid, z_mid))        # x <= x_end
    fr_d  = GEO.shapes.HalfSpace(normal=(0., -1., 0.), origin=(x_mid_d, y_front, z_mid))
    bk_d  = GEO.shapes.HalfSpace(normal=(0.,  1., 0.), origin=(x_mid_d, y_back,  z_mid))
    layer_d = top_d & bot_d & lft_d & rgt_d & fr_d & bk_d

    return layer_h | layer_d


def _slab_piece_x_segment_tan(GEO, *, x_left, x_right, y_back, y_front, z_top_left, thickness, tan_dip: float):
    """
    Segmento de slab entre x_left y x_right.
    Techo pasa por (x_left, z_top_left) y usa tan_dip consistente con endpoints.
    """
    y_mid = 0.5 * (y_back + y_front)
    x_mid = 0.5 * (x_left + x_right)

    th = thickness if hasattr(thickness, "units") else thickness * y_back.units
    z_bot0 = z_top_left - th

    # caja X/Y
    lft = GEO.shapes.HalfSpace(normal=(-1., 0., 0.), origin=(x_left,  y_mid, z_top_left))
    rgt = GEO.shapes.HalfSpace(normal=( 1., 0., 0.), origin=(x_right, y_mid, z_top_left))
    fr  = GEO.shapes.HalfSpace(normal=(0., -1., 0.), origin=(x_mid, y_front, z_top_left))
    bk  = GEO.shapes.HalfSpace(normal=(0.,  1., 0.), origin=(x_mid, y_back,  z_top_left))

    # top/base paralelos
    top = GEO.shapes.HalfSpace(normal=( tan_dip, 0.,  1.), origin=(x_left, y_mid, z_top_left))
    bot = GEO.shapes.HalfSpace(normal=(-tan_dip, 0., -1.), origin=(x_left, y_mid, z_bot0))

    return top & bot & lft & rgt & fr & bk


def _apply_z_cut(GEO, shape, *, z_cut, x_ref, y_ref):
    """Interseca shape con z >= z_cut (corte a profundidad)."""
    if z_cut is None:
        return shape
    z_cut = _as_qty(z_cut, y_ref.units)
    z_mid = z_cut
    cut = GEO.shapes.HalfSpace(normal=(0., 0., -1.), origin=(x_ref, y_ref, z_mid))  # z >= z_cut
    return shape & cut


# -----------------------------------------------------------------------------
# Slab: dip constante (con corte opcional)
# -----------------------------------------------------------------------------
def slab_band_shape(GEO, *, y_back, y_front, dip_deg, thickness, x_break, x_end, z_cut=None):
    units = y_back.units
    z_top = 0.0 * units
    z_bottom = z_top - (thickness if hasattr(thickness, "units") else thickness * units)

    shp = _layer_shape_simple(
        GEO, y_back=y_back, y_front=y_front,
        z_top=z_top, z_bottom=z_bottom,
        x_break=x_break, x_end=x_end,
        dip_deg=dip_deg
    )

    # corte uniforme si se pide
    x_ref = 0.5 * (x_break + x_end)
    y_ref = 0.5 * (y_back + y_front)
    return _apply_z_cut(GEO, shp, z_cut=z_cut, x_ref=x_ref, y_ref=y_ref)


def slab_transition_shape(GEO, *, y_back, y_front, dip_start, dip_end, thickness, x_break, x_end,
                          n_segments, z_cut=None):
    if n_segments <= 0:
        raise ValueError("n_segments debe ser > 0")
    if y_back <= y_front:
        raise ValueError("Se espera y_back > y_front")

    dy = (y_back - y_front) / n_segments
    dips = np.linspace(float(dip_start), float(dip_end), n_segments)

    shape = None
    y_top = y_back
    for dip in dips:
        y_bot = y_top - dy
        seg = slab_band_shape(
            GEO, y_back=y_top, y_front=y_bot,
            dip_deg=float(dip),
            thickness=thickness,
            x_break=x_break, x_end=x_end,
            z_cut=z_cut
        )
        shape = seg if shape is None else (shape | seg)
        y_top = y_bot

    return shape


# -----------------------------------------------------------------------------
# Slab: dip(x) variable (con corte opcional)
# -----------------------------------------------------------------------------
def slab_band_shape_xvary(GEO, *, y_back, y_front, thickness, x_break,
                          x_nodes, dip_nodes, n_subsegments=50,
                          z_top_at_break=None, x_overlap=0.0, z_cut=None):

    if len(x_nodes) != len(dip_nodes):
        raise ValueError("x_nodes y dip_nodes deben tener la misma longitud.")
    if len(x_nodes) < 2:
        raise ValueError("x_nodes debe tener al menos 2 nodos.")
    if n_subsegments <= 0:
        raise ValueError("n_subsegments debe ser > 0")
    if y_back <= y_front:
        raise ValueError("Se espera y_back > y_front")

    # asegurar x_nodes crecientes
    for i in range(len(x_nodes) - 1):
        if x_nodes[i + 1] <= x_nodes[i]:
            raise ValueError("x_nodes debe ser estrictamente creciente.")

    units_y = y_back.units
    z_top0 = (0.0 * units_y) if (z_top_at_break is None) else z_top_at_break
    th = thickness if hasattr(thickness, "units") else thickness * units_y

    x0 = x_break
    x1 = x_nodes[-1]
    if x1 <= x0:
        raise ValueError("x_nodes[-1] debe ser > x_break")

    # ------------------------------------------------------------------
    # (A) CUT EN X si se pide: encontrar x donde z_top(x) == z_cut
    #     y truncar x1 + x_nodes ANTES de integrar/discretizar
    # ------------------------------------------------------------------
    if z_cut is not None:
        zc = z_cut if hasattr(z_cut, "units") else z_cut * units_y

        # Ojo: acá asumimos que en slab_geometries tenés find_x_at_depth
        # y que devuelve Quantity con mismas unidades que x0/x1.
        x_end_cut = find_x_at_depth(
            x0=x0, z0=z_top0,
            x1=x1,
            x_nodes=x_nodes,
            dip_nodes=dip_nodes,
            z_target=zc,
            n=max(12000, int(n_subsegments) * 200)  # robusto
        )

        if x_end_cut is not None and x_end_cut > x0 and x_end_cut < x1:
            x1 = x_end_cut

            # truncar lista de nodos hasta x1 y forzar último nodo = x1
            x_nodes_old = list(x_nodes)
            dip_nodes_old = list(dip_nodes)

            x_nodes = [xn for xn in x_nodes_old if xn <= x1]
            dip_nodes = dip_nodes_old[:len(x_nodes)]

            dip_x1 = dip_piecewise_smooth(x1, x_nodes_old, dip_nodes_old)

            if len(x_nodes) == 0 or x_nodes[-1] != x1:
                x_nodes = x_nodes + [x1]
                dip_nodes = dip_nodes + [dip_x1]

    # tramo horizontal 0..x_break
    y_mid = 0.5 * (y_back + y_front)
    x_mid_h = 0.5 * x0
    z_bot0 = z_top0 - th

    horiz = horizontal_layer(
        GEO,
        x_left=0.0 * x0, x_right=x0,
        y_back=y_back, y_front=y_front,
        z_top=z_top0, z_bottom=z_bot0
    )

        
    # discretización x_break..x_end (ahora x1 ya está "cortado" si corresponde)
    xs = np.linspace(float(x0.magnitude), float(x1.magnitude), int(n_subsegments) + 1) * x0.units

    # integrar z_top(x) con oversampling
    xs_int, z_int = integrate_slab_top_z(
        x0=x0, z0=z_top0, x1=x1,
        x_nodes=x_nodes, dip_nodes=dip_nodes,
        n=int(n_subsegments) * 40
    )

    z_top_vals = np.interp(
        [float(v.magnitude) for v in xs],
        [float(v.magnitude) for v in xs_int],
        [float(v.magnitude) for v in z_int],
    ) * z_top0.units

    eps = _as_qty(x_overlap, x0.units)

    shape_dip = None
    for i in range(len(xs) - 1):
        xl = xs[i]   - (eps if i > 0 else 0.0 * x0.units)
        xr = xs[i+1] + (eps if i < len(xs) - 2 else 0.0 * x0.units)

        dx = xs[i + 1] - xs[i]
        dz = z_top_vals[i + 1] - z_top_vals[i]
        tan_f = float((-dz / dx).magnitude)

        seg = _slab_piece_x_segment_tan(
            GEO,
            x_left=xl, x_right=xr,
            y_back=y_back, y_front=y_front,
            z_top_left=z_top_vals[i],
            thickness=th,          # <-- FIX: usar th (Quantity), no thickness “crudo”
            tan_dip=tan_f
        )
        shape_dip = seg if shape_dip is None else (shape_dip | seg)

    shp = horiz if shape_dip is None else (horiz | shape_dip)

    # corte uniforme final (ya no debería "deformar" porque no hay cola extra)
    x_ref = 0.5 * (x0 + x1)
    y_ref = 0.5 * (y_back + y_front)
    return _apply_z_cut(GEO, shp, z_cut=z_cut, x_ref=x_ref, y_ref=y_ref)



def add_transition_y_xvary(
    GEO,
    slab_shapes,
    *,
    y_back, y_front,
    thickness,
    x_break,
    x_nodes,
    n_trans_segments,
    n_xsub,
    z_top_at_break,
    x_overlap=0.0,
    y_overlap=0.0,
    # --- Modo A (legacy): interpola solo el dip inicial (en x_break)
    dip_start=None,
    dip_end=None,
    dip0=None,
    dip_steep=None,
    # --- Modo B: interpola listas completas de dip_nodes
    dip_nodes_start=None,
    dip_nodes_end=None,
    # --- Corte opcional del slab (p.ej. -660 km)
    z_cut=None,
):
    if n_trans_segments <= 0:
        raise ValueError("n_trans_segments debe ser > 0")
    if n_xsub <= 0:
        raise ValueError("n_xsub debe ser > 0")
    if y_back <= y_front:
        raise ValueError("Se espera y_back > y_front")

    dy = (y_back - y_front) / n_trans_segments
    ts = np.linspace(0.0, 1.0, n_trans_segments)

    yo = y_overlap if hasattr(y_overlap, "units") else float(y_overlap) * y_back.units

    use_nodes_mode = (dip_nodes_start is not None) or (dip_nodes_end is not None)
    if use_nodes_mode:
        if dip_nodes_start is None or dip_nodes_end is None:
            raise ValueError("Si usás dip_nodes_start/end, tenés que pasar ambos.")
        if len(dip_nodes_start) != len(dip_nodes_end):
            raise ValueError("dip_nodes_start y dip_nodes_end deben tener la misma longitud.")
        if len(dip_nodes_start) < 2:
            raise ValueError("dip_nodes_start/end deben tener al menos 2 valores.")
    else:
        if dip_start is None or dip_end is None or dip0 is None or dip_steep is None:
            raise ValueError(
                "Modo legacy: falta alguno de dip_start, dip_end, dip0, dip_steep. "
                "Alternativamente pasá dip_nodes_start y dip_nodes_end."
            )

    y_top = y_back
    for t in ts:
        y_bot = y_top - dy

        # overlap en Y para evitar gaps entre subfranjas
        y_back_eff  = y_top + 0.5 * yo
        y_front_eff = y_bot - 0.5 * yo

        if use_nodes_mode:
            dip_nodes = [
                float(a) + (float(b) - float(a)) * float(t)
                for a, b in zip(dip_nodes_start, dip_nodes_end)
            ]
        else:
            dip_in = float(dip_start) + (float(dip_end) - float(dip_start)) * float(t)
            dip_nodes = [dip_in, float(dip0), float(dip0), float(dip_steep), float(dip_steep)]

        slab_shapes.append(
            slab_band_shape_xvary(
                GEO,
                y_back=y_back_eff,
                y_front=y_front_eff,
                thickness=thickness,
                x_break=x_break,
                x_nodes=x_nodes,
                dip_nodes=dip_nodes,
                n_subsegments=n_xsub,
                z_top_at_break=z_top_at_break,
                x_overlap=x_overlap,
                z_cut=z_cut,
            )
        )

        y_top = y_bot



# -----------------------------------------------------------------------------
# Overriding wedge capped thickness
# -----------------------------------------------------------------------------
def overriding_wedge_capped_thickness(GEO, *, x_left, x_right, y_back, y_front,
                                      z_slab_top_at_break, z_top_over, dip_deg, max_thickness):
    """
    Cuña de overriding (single-layer):
      - Planta: x in [x_left, x_right], y in [y_front, y_back]
      - Techo: z <= z_top_over
      - Base: por arriba del plano del slab (dip_deg) y además z >= z_top_over - max_thickness
    """
    u = _u(GEO)
    max_thickness = _as_qty(max_thickness, u.kilometer)
    if max_thickness.dimensionality != u.kilometer.dimensionality:
        raise ValueError("max_thickness debe ser una longitud (km).")

    y_mid = 0.5 * (y_back + y_front)
    x_mid = 0.5 * (x_left + x_right)

    # caja en planta
    lft = GEO.shapes.HalfSpace(normal=(-1., 0., 0.), origin=(x_left,  y_mid, z_top_over))
    rgt = GEO.shapes.HalfSpace(normal=( 1., 0., 0.), origin=(x_right, y_mid, z_top_over))
    fr  = GEO.shapes.HalfSpace(normal=(0., -1., 0.), origin=(x_mid, y_front, z_top_over))
    bk  = GEO.shapes.HalfSpace(normal=(0.,  1., 0.), origin=(x_mid, y_back,  z_top_over))

    # techo (z <= z_top_over)
    top = GEO.shapes.HalfSpace(normal=(0., 0., 1.), origin=(x_mid, y_mid, z_top_over))

    # base slab (z >= z_slab(x) con plano por (x_left, z_slab_top_at_break))
    tan_dip = float(np.tan(np.deg2rad(float(dip_deg))))
    base_slab = GEO.shapes.HalfSpace(normal=(-tan_dip, 0., -1.), origin=(x_left, y_mid, z_slab_top_at_break))

    # cap: z >= z_top_over - max_thickness
    z_cap = z_top_over - max_thickness
    base_cap = GEO.shapes.HalfSpace(normal=(0., 0., -1.), origin=(x_mid, y_mid, z_cap))

    return top & base_slab & base_cap & lft & rgt & fr & bk


def overriding_band_shape(GEO, *, x_left_plate, x_knee, y_back, y_front,
                          z_slab_top_at_break, z_top_surface, dip_deg, max_over_thickness):
    return overriding_wedge_capped_thickness(
        GEO,
        x_left=x_left_plate, x_right=x_knee,
        y_back=y_back, y_front=y_front,
        z_slab_top_at_break=z_slab_top_at_break,
        z_top_over=z_top_surface,
        dip_deg=dip_deg,
        max_thickness=max_over_thickness
    )


def overriding_transition_shape(GEO, *, x_left_plate, x_knee, y_back, y_front,
                                z_slab_top_at_break, z_top_surface, dip_start, dip_end,
                                max_over_thickness, n_segments, max_over_thickness_end=None):
    if n_segments <= 0:
        raise ValueError("n_segments debe ser > 0")
    if y_back <= y_front:
        raise ValueError("Se espera y_back > y_front")

    dy = (y_back - y_front) / n_segments
    dips = np.linspace(float(dip_start), float(dip_end), n_segments)

    if max_over_thickness_end is None:
        ths = [max_over_thickness] * n_segments
    else:
        ts = np.linspace(0.0, 1.0, n_segments)
        ths = [max_over_thickness + (max_over_thickness_end - max_over_thickness) * t for t in ts]

    shape = None
    y_top = y_back
    for dip, th in zip(dips, ths):
        y_bot = y_top - dy
        seg = overriding_band_shape(
            GEO,
            x_left_plate=x_left_plate, x_knee=x_knee,
            y_back=y_top, y_front=y_bot,
            z_slab_top_at_break=z_slab_top_at_break,
            z_top_surface=z_top_surface,
            dip_deg=float(dip),
            max_over_thickness=th
        )
        shape = seg if shape is None else (shape | seg)
        y_top = y_bot

    return shape

def union_balanced(shapes):
    shapes = list(shapes)
    if not shapes:
        return None
    while len(shapes) > 1:
        nxt = []
        for i in range(0, len(shapes), 2):
            if i + 1 < len(shapes):
                nxt.append(shapes[i] | shapes[i+1])
            else:
                nxt.append(shapes[i])
        shapes = nxt
    return shapes[0]


# -----------------------------------------------------------------------------
# Timing utilities (agnóstico a UW/MPI)
# -----------------------------------------------------------------------------
import time
from contextlib import contextmanager

def make_timer(*, barrier_fn=None, printer=print):
    """
    Crea un helper de timing para instrumentar armado de geometrías.

    Parameters
    ----------
    barrier_fn : callable | None
        Si estás corriendo en MPI, pasá uw.mpi.barrier (sin paréntesis).
        Si es None, no sincroniza.
    printer : callable
        Función para imprimir logs (por defecto print).

    Returns
    -------
    timed : contextmanager(label, sync=True)
    timings : list[(label, seconds)]
    dump : function(path)
    """
    timings = []

    @contextmanager
    def timed(label: str, sync: bool = True):
        if sync and barrier_fn is not None:
            try:
                barrier_fn()
            except Exception:
                pass

        t0 = time.perf_counter()
        yield
        dt = time.perf_counter() - t0

        if sync and barrier_fn is not None:
            try:
                barrier_fn()
            except Exception:
                pass

        timings.append((label, dt))
        if printer is not None:
            try:
                printer(f"[TIMER] {label:45s} {dt:10.3f} s")
            except Exception:
                pass

    def dump(path: str):
        total = sum(dt for _, dt in timings)
        with open(path, "w") as f:
            f.write("=== SLAB BUILD TIMINGS (seconds) ===\n")
            for lab, dt in timings:
                f.write(f"{lab:45s} {dt:12.6f}\n")
            f.write(f"{'TOTAL':45s} {total:12.6f}\n")

    return timed, timings, dump
