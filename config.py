from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class ModelConfig:
    exp_name: str = "M01_H"
    vel_tag: str = "IV"

    # dips (deg)
    dip_in: float = 15.0
    dip_0: float = 15.0
    dip_steep: float = 50.0
    dip_normal: float = 30.0
    dip_wedge_flat: Optional[float] = None

    # thickness (km)
    th_flat: float = 50.0
    th_cent: float = 50.0

    # mesh + swarm
    nx: int = 88
    ny: int = 176
    nz: int = 68
    ppc: int = 30
    pop: int = 40

    # optional vertical regrid
    regrid_z: bool = False
    regrid_beta: float = 0.5

    # global geometry
    x_domain: Tuple[float, float] = (0.0, 2000.0)
    y_domain: Tuple[float, float] = (0.0, 4000.0)
    z_domain: Tuple[float, float] = (-800.0, 0.0)

    n_trans_segments: int = 8
    n_xsub: int = 80

    # subduction
    x_break: float = 500.0
    x_end: float = 2000.0
    x_flat_start: Optional[float] = None
    x_flat_end: Optional[float] = None
    x_steep_start: Optional[float] = None
    x_steep_end: Optional[float] = None
    overlap: float = 2.0
    slab_thickness: float = 30.0
    slab_channel_thickness: float = 12.0

    # overriding
    ovr_total_thick: Optional[float] = None
    ovr_layer_thick: Optional[float] = None
    ovr_knee_length: float = 250.0
    ovr_max_thick_wedge: Optional[float] = None

    # runtime
    total_time_years: int = 25000
    checkpoint_every: int = 5000
    smoke_test: bool = False
    restart: bool = False

    def __post_init__(self):
        if self.dip_wedge_flat is None:
            self.dip_wedge_flat = float(self.dip_in)

        if self.x_flat_start is None:
            self.x_flat_start = self.x_break + 260.0
        if self.x_flat_end is None:
            self.x_flat_end = self.x_break + 360.0
        if self.x_steep_start is None:
            self.x_steep_start = self.x_break + 380.0
        if self.x_steep_end is None:
            self.x_steep_end = self.x_break + 1000.0

        if self.ovr_total_thick is None:
            self.ovr_total_thick = self.th_cent
        if self.ovr_layer_thick is None:
            self.ovr_layer_thick = round(self.ovr_total_thick / 3.0, 6)
        if self.ovr_max_thick_wedge is None:
            self.ovr_max_thick_wedge = self.th_cent


def validate_config(cfg: ModelConfig) -> None:
    errors = []

    if cfg.vel_tag not in {"IV", "NV"}:
        errors.append("vel_tag debe ser 'IV' o 'NV'.")

    for key in ("x_domain", "y_domain", "z_domain"):
        dom = getattr(cfg, key)
        if len(dom) != 2:
            errors.append(f"{key} debe tener exactamente 2 valores.")
            continue
        if dom[0] >= dom[1]:
            errors.append(f"{key} debe ser creciente: min < max.")

    for name in ("dip_in", "dip_0", "dip_steep", "dip_normal", "dip_wedge_flat"):
        v = float(getattr(cfg, name))
        if not (0.0 < v < 90.0):
            errors.append(f"{name} debe estar en (0, 90) grados.")

    if abs(float(cfg.dip_wedge_flat) - float(cfg.dip_in)) > 1e-12:
        errors.append("dip_wedge_flat debe ser coherente con dip_in.")

    for name in ("th_flat", "th_cent", "slab_thickness", "slab_channel_thickness", "ovr_knee_length"):
        if float(getattr(cfg, name)) <= 0.0:
            errors.append(f"{name} debe ser > 0.")
    if float(cfg.slab_channel_thickness) > float(cfg.slab_thickness):
        errors.append("slab_channel_thickness no puede exceder slab_thickness.")
    dz_km = (float(cfg.z_domain[1]) - float(cfg.z_domain[0])) / float(cfg.nz)
    if float(cfg.slab_channel_thickness) < float(dz_km):
        errors.append(
            f"slab_channel_thickness debe ser >= dz de malla ({dz_km:.6g} km) para resolverse numéricamente."
        )

    if float(cfg.overlap) < 0.0:
        errors.append("overlap debe ser >= 0.")

    for name in ("nx", "ny", "nz", "ppc", "pop", "n_trans_segments", "n_xsub"):
        if int(getattr(cfg, name)) <= 0:
            errors.append(f"{name} debe ser entero positivo.")

    if float(cfg.regrid_beta) <= 0.0:
        errors.append("regrid_beta debe ser > 0.")

    if not (cfg.x_domain[0] <= cfg.x_break <= cfg.x_domain[1]):
        errors.append("x_break debe quedar dentro de x_domain.")
    if not (cfg.x_domain[0] <= cfg.x_end <= cfg.x_domain[1]):
        errors.append("x_end debe quedar dentro de x_domain.")
    if cfg.x_end < cfg.x_break:
        errors.append("x_end debe ser >= x_break.")

    x_nodes = [
        cfg.x_break,
        cfg.x_flat_start,
        cfg.x_flat_end,
        cfg.x_steep_start,
        cfg.x_steep_end,
        cfg.x_end,
    ]
    if any(float(x_nodes[i]) >= float(x_nodes[i + 1]) for i in range(len(x_nodes) - 1)):
        errors.append("Nodos en X no monotónicos: x_break < x_flat_start < x_flat_end < x_steep_start < x_steep_end < x_end.")

    if float(cfg.ovr_total_thick) <= 0.0:
        errors.append("ovr_total_thick debe ser > 0.")
    if float(cfg.ovr_layer_thick) <= 0.0:
        errors.append("ovr_layer_thick debe ser > 0.")
    expected_layer = round(float(cfg.ovr_total_thick) / 3.0, 6)
    if abs(float(cfg.ovr_layer_thick) - expected_layer) > 1e-9:
        errors.append("ovr_layer_thick debe ser coherente con round(ovr_total_thick/3, 6).")
    if float(cfg.ovr_max_thick_wedge) <= 0.0:
        errors.append("ovr_max_thick_wedge debe ser > 0.")
    if float(cfg.ovr_max_thick_wedge) > float(cfg.ovr_total_thick):
        errors.append("ovr_max_thick_wedge no debe exceder ovr_total_thick.")

    if int(cfg.total_time_years) <= 0:
        errors.append("total_time_years debe ser > 0.")
    if int(cfg.checkpoint_every) <= 0:
        errors.append("checkpoint_every debe ser > 0.")
    if int(cfg.checkpoint_every) > int(cfg.total_time_years):
        errors.append("checkpoint_every no puede ser mayor que total_time_years.")

    if errors:
        raise ValueError("Configuración inválida:\n- " + "\n- ".join(errors))
