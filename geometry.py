from contextlib import contextmanager

import numpy as np

from slab_geometries import (
    add_transition_y_xvary,
    debug_slab_profile,
    find_x_at_depth,
    horizontal_layer,
    overriding_band_shape,
    overriding_transition_shape,
    slab_band_shape_xvary,
    union_balanced,
)


@contextmanager
def _noop_timed(_label: str):
    yield


def _as_km(x, u):
    return x if hasattr(x, "units") else float(x) * u.kilometer


def x_at_depth_constant_dip(*, dip_deg, z_target, x0, u):
    dip = np.deg2rad(float(dip_deg))
    return x0 + (abs(z_target.to(u.kilometer).magnitude) / np.tan(dip)) * u.kilometer


def define_geometry_params(
    *,
    u,
    x_break_km,
    x_end_km,
    x_flat_start_km,
    x_flat_end_km,
    x_steep_start_km,
    x_steep_end_km,
    dip_in_deg,
    dip_0_deg,
    dip_steep_deg,
    slab_thickness_km,
    z_cut,
):
    x_break = float(x_break_km) * u.kilometer
    x_end_max = float(x_end_km) * u.kilometer

    x_nodes_fixed = [
        float(x_flat_start_km) * u.kilometer,
        float(x_flat_end_km) * u.kilometer,
        float(x_steep_start_km) * u.kilometer,
        float(x_steep_end_km) * u.kilometer,
    ]

    z_target_top = z_cut
    dip_nodes_flat = [float(dip_in_deg), float(dip_0_deg), float(dip_0_deg), float(dip_steep_deg), float(dip_steep_deg)]
    dip_nodes_30curve = [30.0, 32.0, 34.0, float(dip_steep_deg), float(dip_steep_deg)]

    x1_safe = min(
        x_at_depth_constant_dip(dip_deg=dip_steep_deg, z_target=z_target_top, x0=x_break, u=u) + 50.0 * u.kilometer,
        x_end_max,
    )
    x_nodes_base = x_nodes_fixed + [x_end_max]

    def _find_x_end_for_profile(*, dip_nodes):
        x_end = find_x_at_depth(
            x0=x_break,
            z0=0.0 * u.kilometer,
            x1=x1_safe,
            x_nodes=x_nodes_base,
            dip_nodes=dip_nodes,
            z_target=z_target_top,
            n=12000,
        )
        if x_end is None:
            x_end = x_end_max

        x_min = float(x_steep_end_km) * u.kilometer
        if x_end <= x_min:
            x_end = x_min + 1e-6 * u.kilometer
        return x_end

    x_end_flat = _find_x_end_for_profile(dip_nodes=dip_nodes_flat)
    x_end_30curve = _find_x_end_for_profile(dip_nodes=dip_nodes_30curve)

    x_nodes_flat = x_nodes_fixed + [x_end_flat]
    x_nodes_30curve = x_nodes_fixed + [x_end_30curve]
    x_end_trans_flat_30 = min(x_end_flat, x_end_30curve)
    x_nodes_trans = x_nodes_fixed + [x_end_trans_flat_30]

    th = float(slab_thickness_km) * u.kilometer
    slab_thickness_total = 3.0 * th

    y_bands = {
        "y_back_30N": 4000.0 * u.kilometer,
        "y_front_30N": 3850.0 * u.kilometer,
        "y_back_T30_15_N": 3850.0 * u.kilometer,
        "y_front_T30_15_N": 3750.0 * u.kilometer,
        "y_back_15C": 3750.0 * u.kilometer,
        "y_front_15C": 2450.0 * u.kilometer,
        "y_back_T15_30_1": 2450.0 * u.kilometer,
        "y_front_T15_30_1": 2350.0 * u.kilometer,
        "y_back_30S1": 2350.0 * u.kilometer,
        "y_front_30S1": 1150.0 * u.kilometer,
        "y_back_T30_15_1": 1150.0 * u.kilometer,
        "y_front_T30_15_1": 1050.0 * u.kilometer,
        "y_back_15S2": 1050.0 * u.kilometer,
        "y_front_15S2": 350.0 * u.kilometer,
        "y_back_T15_30_2": 350.0 * u.kilometer,
        "y_front_T15_30_2": 250.0 * u.kilometer,
        "y_back_30S2": 250.0 * u.kilometer,
        "y_front_30S2": 0.0 * u.kilometer,
    }

    return {
        "x_break": x_break,
        "x_end_max": x_end_max,
        "x_nodes_fixed": x_nodes_fixed,
        "z_target_top": z_target_top,
        "dip_nodes_flat": dip_nodes_flat,
        "dip_nodes_30curve": dip_nodes_30curve,
        "x_end_flat": x_end_flat,
        "x_end_30curve": x_end_30curve,
        "x_nodes_flat": x_nodes_flat,
        "x_nodes_30curve": x_nodes_30curve,
        "x_end_trans_flat_30": x_end_trans_flat_30,
        "x_nodes_trans": x_nodes_trans,
        "th": th,
        "slab_thickness_total": slab_thickness_total,
        "y_bands": y_bands,
    }


def run_geometry_debug_profile(*, params, dip_in_deg, dip_0_deg, dip_steep_deg):
    debug_slab_profile(
        x_break=params["x_break"],
        x_nodes=params["x_nodes_flat"],
        dip_nodes=params["dip_nodes_flat"],
        x_end=params["x_end_flat"],
        dip_label=f"flat {int(dip_in_deg)}->{int(dip_0_deg)}->{int(dip_steep_deg)} (x_end by z=-660)",
    )


def build_slab_shape(
    *,
    GEO,
    params,
    slab_thickness_km,
    n_xsub,
    n_trans_segments,
    x_overlap,
    y_overlap,
    z_cut,
    timed=None,
):
    timed = timed or _noop_timed
    y = params["y_bands"]

    with timed("SLAB total build (all segments + union)"):
        slab_shapes = []

        with timed("slab seg 1: 30N 30curve"):
            slab_shapes.append(
                slab_band_shape_xvary(
                    GEO,
                    y_back=y["y_back_30N"],
                    y_front=y["y_front_30N"],
                    thickness=slab_thickness_km,
                    x_break=params["x_break"],
                    x_nodes=params["x_nodes_30curve"],
                    dip_nodes=params["dip_nodes_30curve"],
                    n_subsegments=n_xsub,
                    z_top_at_break=0.0 * y["y_back_30N"].units,
                    x_overlap=x_overlap,
                    z_cut=z_cut,
                )
            )

        with timed("slab seg 2: T 30curve->flat (N)"):
            add_transition_y_xvary(
                GEO,
                slab_shapes,
                y_back=y["y_back_T30_15_N"],
                y_front=y["y_front_T30_15_N"],
                thickness=slab_thickness_km,
                x_break=params["x_break"],
                x_nodes=params["x_nodes_trans"],
                n_trans_segments=n_trans_segments,
                n_xsub=n_xsub,
                z_top_at_break=0.0 * y["y_back_30N"].units,
                x_overlap=x_overlap,
                y_overlap=y_overlap,
                dip_nodes_start=params["dip_nodes_30curve"],
                dip_nodes_end=params["dip_nodes_flat"],
                z_cut=z_cut,
            )

        with timed("slab seg 3: flat (C)"):
            slab_shapes.append(
                slab_band_shape_xvary(
                    GEO,
                    y_back=y["y_back_15C"],
                    y_front=y["y_front_15C"],
                    thickness=slab_thickness_km,
                    x_break=params["x_break"],
                    x_nodes=params["x_nodes_flat"],
                    dip_nodes=params["dip_nodes_flat"],
                    n_subsegments=n_xsub,
                    z_top_at_break=0.0 * y["y_back_30N"].units,
                    x_overlap=x_overlap,
                    z_cut=z_cut,
                )
            )

        with timed("slab seg 4: T flat->30curve (N)"):
            add_transition_y_xvary(
                GEO,
                slab_shapes,
                y_back=y["y_back_T15_30_1"],
                y_front=y["y_front_T15_30_1"],
                thickness=slab_thickness_km,
                x_break=params["x_break"],
                x_nodes=params["x_nodes_trans"],
                n_trans_segments=n_trans_segments,
                n_xsub=n_xsub,
                z_top_at_break=0.0 * y["y_back_30N"].units,
                x_overlap=x_overlap,
                y_overlap=y_overlap,
                dip_nodes_start=params["dip_nodes_flat"],
                dip_nodes_end=params["dip_nodes_30curve"],
                z_cut=z_cut,
            )

        with timed("slab seg 5: 30S1 30curve"):
            slab_shapes.append(
                slab_band_shape_xvary(
                    GEO,
                    y_back=y["y_back_30S1"],
                    y_front=y["y_front_30S1"],
                    thickness=slab_thickness_km,
                    x_break=params["x_break"],
                    x_nodes=params["x_nodes_30curve"],
                    dip_nodes=params["dip_nodes_30curve"],
                    n_subsegments=n_xsub,
                    z_top_at_break=0.0 * y["y_back_30N"].units,
                    x_overlap=x_overlap,
                    z_cut=z_cut,
                )
            )

        with timed("slab seg 6: T 30curve->flat (S)"):
            add_transition_y_xvary(
                GEO,
                slab_shapes,
                y_back=y["y_back_T30_15_1"],
                y_front=y["y_front_T30_15_1"],
                thickness=slab_thickness_km,
                x_break=params["x_break"],
                x_nodes=params["x_nodes_trans"],
                n_trans_segments=n_trans_segments,
                n_xsub=n_xsub,
                z_top_at_break=0.0 * y["y_back_30N"].units,
                x_overlap=x_overlap,
                y_overlap=y_overlap,
                dip_nodes_start=params["dip_nodes_30curve"],
                dip_nodes_end=params["dip_nodes_flat"],
                z_cut=z_cut,
            )

        with timed("slab seg 7: flat (S2)"):
            slab_shapes.append(
                slab_band_shape_xvary(
                    GEO,
                    y_back=y["y_back_15S2"],
                    y_front=y["y_front_15S2"],
                    thickness=slab_thickness_km,
                    x_break=params["x_break"],
                    x_nodes=params["x_nodes_flat"],
                    dip_nodes=params["dip_nodes_flat"],
                    n_subsegments=n_xsub,
                    z_top_at_break=0.0 * y["y_back_30N"].units,
                    x_overlap=x_overlap,
                    z_cut=z_cut,
                )
            )

        with timed("slab seg 8: T flat->30curve (S)"):
            add_transition_y_xvary(
                GEO,
                slab_shapes,
                y_back=y["y_back_T15_30_2"],
                y_front=y["y_front_T15_30_2"],
                thickness=slab_thickness_km,
                x_break=params["x_break"],
                x_nodes=params["x_nodes_trans"],
                n_trans_segments=n_trans_segments,
                n_xsub=n_xsub,
                z_top_at_break=0.0 * y["y_back_30N"].units,
                x_overlap=x_overlap,
                y_overlap=y_overlap,
                dip_nodes_start=params["dip_nodes_flat"],
                dip_nodes_end=params["dip_nodes_30curve"],
                z_cut=z_cut,
            )

        with timed("slab seg 9: 30S2 30curve"):
            slab_shapes.append(
                slab_band_shape_xvary(
                    GEO,
                    y_back=y["y_back_30S2"],
                    y_front=y["y_front_30S2"],
                    thickness=slab_thickness_km,
                    x_break=params["x_break"],
                    x_nodes=params["x_nodes_30curve"],
                    dip_nodes=params["dip_nodes_30curve"],
                    n_subsegments=n_xsub,
                    z_top_at_break=0.0 * y["y_back_30N"].units,
                    x_overlap=x_overlap,
                    z_cut=z_cut,
                )
            )

        with timed("slab union_balanced(slab_shapes)"):
            slab_shape = union_balanced(slab_shapes)

    return slab_shape


def build_overriding_shape(
    *,
    GEO,
    u,
    params,
    x_domain,
    ovr_knee_length_km,
    x_flat_start_km,
    dip_normal_deg,
    dip_wedge_flat_deg,
    th_flat_km,
    th_cent_km,
    ovr_max_thick_wedge_km,
    n_trans_segments,
):
    y = params["y_bands"]
    x_left_plate = params["x_break"]
    x_knee = max(
        x_left_plate + float(ovr_knee_length_km) * u.kilometer,
        float(x_flat_start_km) * u.kilometer,
    )
    x_right_plate = float(x_domain[1]) * u.kilometer

    z_top_surface = 0.0 * u.kilometer
    z_slab_top_at_break = 0.0 * u.kilometer
    max_over_thickness = float(ovr_max_thick_wedge_km) * u.kilometer
    th_flat = _as_km(th_flat_km, u)
    th_cent = _as_km(th_cent_km, u)

    franjas_y = [
        (y["y_back_30N"], y["y_front_30N"]),
        (y["y_back_T30_15_N"], y["y_front_T30_15_N"]),
        (y["y_back_15C"], y["y_front_15C"]),
        (y["y_back_T15_30_1"], y["y_front_T15_30_1"]),
        (y["y_back_30S1"], y["y_front_30S1"]),
        (y["y_back_T30_15_1"], y["y_front_T30_15_1"]),
        (y["y_back_15S2"], y["y_front_15S2"]),
        (y["y_back_T15_30_2"], y["y_front_T15_30_2"]),
        (y["y_back_30S2"], y["y_front_30S2"]),
    ]

    over_shapes = [
        overriding_band_shape(
            GEO,
            x_left_plate=x_left_plate,
            x_knee=x_knee,
            y_back=y["y_back_30N"],
            y_front=y["y_front_30N"],
            z_slab_top_at_break=z_slab_top_at_break,
            z_top_surface=z_top_surface,
            dip_deg=dip_normal_deg,
            max_over_thickness=max_over_thickness,
        ),
        overriding_transition_shape(
            GEO,
            x_left_plate=x_left_plate,
            x_knee=x_knee,
            y_back=y["y_back_T30_15_N"],
            y_front=y["y_front_T30_15_N"],
            z_slab_top_at_break=z_slab_top_at_break,
            z_top_surface=z_top_surface,
            dip_start=dip_normal_deg,
            dip_end=dip_wedge_flat_deg,
            max_over_thickness=max_over_thickness,
            n_segments=n_trans_segments,
        ),
        overriding_band_shape(
            GEO,
            x_left_plate=x_left_plate,
            x_knee=x_knee,
            y_back=y["y_back_15C"],
            y_front=y["y_front_15C"],
            z_slab_top_at_break=z_slab_top_at_break,
            z_top_surface=z_top_surface,
            dip_deg=dip_wedge_flat_deg,
            max_over_thickness=th_flat,
        ),
        overriding_transition_shape(
            GEO,
            x_left_plate=x_left_plate,
            x_knee=x_knee,
            y_back=y["y_back_T15_30_1"],
            y_front=y["y_front_T15_30_1"],
            z_slab_top_at_break=z_slab_top_at_break,
            z_top_surface=z_top_surface,
            dip_start=dip_wedge_flat_deg,
            dip_end=dip_normal_deg,
            max_over_thickness=th_flat,
            max_over_thickness_end=th_cent,
            n_segments=n_trans_segments,
        ),
        overriding_band_shape(
            GEO,
            x_left_plate=x_left_plate,
            x_knee=x_knee,
            y_back=y["y_back_30S1"],
            y_front=y["y_front_30S1"],
            z_slab_top_at_break=z_slab_top_at_break,
            z_top_surface=z_top_surface,
            dip_deg=dip_normal_deg,
            max_over_thickness=th_cent,
        ),
        overriding_transition_shape(
            GEO,
            x_left_plate=x_left_plate,
            x_knee=x_knee,
            y_back=y["y_back_T30_15_1"],
            y_front=y["y_front_T30_15_1"],
            z_slab_top_at_break=z_slab_top_at_break,
            z_top_surface=z_top_surface,
            dip_start=dip_normal_deg,
            dip_end=dip_wedge_flat_deg,
            max_over_thickness=th_flat,
            max_over_thickness_end=th_flat,
            n_segments=n_trans_segments,
        ),
        overriding_band_shape(
            GEO,
            x_left_plate=x_left_plate,
            x_knee=x_knee,
            y_back=y["y_back_15S2"],
            y_front=y["y_front_15S2"],
            z_slab_top_at_break=z_slab_top_at_break,
            z_top_surface=z_top_surface,
            dip_deg=dip_wedge_flat_deg,
            max_over_thickness=th_flat,
        ),
        overriding_transition_shape(
            GEO,
            x_left_plate=x_left_plate,
            x_knee=x_knee,
            y_back=y["y_back_T15_30_2"],
            y_front=y["y_front_T15_30_2"],
            z_slab_top_at_break=z_slab_top_at_break,
            z_top_surface=z_top_surface,
            dip_start=dip_wedge_flat_deg,
            dip_end=dip_normal_deg,
            max_over_thickness=max_over_thickness,
            n_segments=n_trans_segments,
        ),
        overriding_band_shape(
            GEO,
            x_left_plate=x_left_plate,
            x_knee=x_knee,
            y_back=y["y_back_30S2"],
            y_front=y["y_front_30S2"],
            z_slab_top_at_break=z_slab_top_at_break,
            z_top_surface=z_top_surface,
            dip_deg=dip_normal_deg,
            max_over_thickness=max_over_thickness,
        ),
    ]

    over_shape_wedges = union_balanced(over_shapes)

    def flat_thickness_for_segment(y_back, y_front):
        if (y_back == y["y_back_15C"] and y_front == y["y_front_15C"]) or (
            y_back == y["y_back_15S2"] and y_front == y["y_front_15S2"]
        ):
            return th_flat
        if y_back == y["y_back_30S1"] and y_front == y["y_front_30S1"]:
            return th_cent
        return max_over_thickness

    flat_parts = []
    for y_back, y_front in franjas_y:
        if y_back == y["y_back_T15_30_1"] and y_front == y["y_front_T15_30_1"]:
            dy = (y_back - y_front) / n_trans_segments
            ts = np.linspace(0.0, 1.0, n_trans_segments)
            ths = [th_flat + (th_cent - th_flat) * t for t in ts]
            y_top = y_back
            for th in ths:
                y_bot = y_top - dy
                flat_parts.append(
                    horizontal_layer(
                        GEO,
                        x_left=x_knee,
                        x_right=x_right_plate,
                        y_back=y_top,
                        y_front=y_bot,
                        z_top=z_top_surface,
                        z_bottom=-th,
                    )
                )
                y_top = y_bot
            continue

        if y_back == y["y_back_T30_15_1"] and y_front == y["y_front_T30_15_1"]:
            dy = (y_back - y_front) / n_trans_segments
            ts = np.linspace(0.0, 1.0, n_trans_segments)
            ths = [th_cent + (th_flat - th_cent) * t for t in ts]
            y_top = y_back
            for th in ths:
                y_bot = y_top - dy
                flat_parts.append(
                    horizontal_layer(
                        GEO,
                        x_left=x_knee,
                        x_right=x_right_plate,
                        y_back=y_top,
                        y_front=y_bot,
                        z_top=z_top_surface,
                        z_bottom=-th,
                    )
                )
                y_top = y_bot
            continue

        th = flat_thickness_for_segment(y_back, y_front)
        flat_parts.append(
            horizontal_layer(
                GEO,
                x_left=x_knee,
                x_right=x_right_plate,
                y_back=y_back,
                y_front=y_front,
                z_top=z_top_surface,
                z_bottom=-th,
            )
        )

    over_shape_flat = union_balanced(flat_parts)
    return over_shape_wedges | over_shape_flat
