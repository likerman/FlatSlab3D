#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
3D Flat-slab model

Description:
    - 3D subduction with multiple along-strike segments (30° / 15° dips + smooth transitions)
    - Overriding plate with capped wedge (≤ 60 km) and 3-layer structure continuous to the east
    - Setup prepared for Underworld UWGeodynamics

Author: J.L. + ChatGPT helper
"""
# ============================================================
# USER PARAMETERS / CONFIG (from dataclass)
# ============================================================

from config import ModelConfig, validate_config

cfg = ModelConfig()
validate_config(cfg)

EXP_NAME = cfg.exp_name
VEL_TAG = cfg.vel_tag

DIP_IN = cfg.dip_in
DIP_0 = cfg.dip_0
DIP_STEEP = cfg.dip_steep
DIP_normal = cfg.dip_normal
DIP_wedge_flat = cfg.dip_wedge_flat

TH_FLAT = cfg.th_flat
TH_CENT = cfg.th_cent

NX, NY, NZ = cfg.nx, cfg.ny, cfg.nz
ppc = cfg.ppc
pop = cfg.pop

reGridZ = cfg.regrid_z
reGrid_beta = cfg.regrid_beta

X_DOMAIN = cfg.x_domain
Y_DOMAIN = cfg.y_domain
Z_DOMAIN = cfg.z_domain

N_TRANS_SEGMENTS = cfg.n_trans_segments
N_XSUB = cfg.n_xsub

X_BREAK = cfg.x_break
X_END = cfg.x_end

X_FLAT_START = cfg.x_flat_start
X_FLAT_END = cfg.x_flat_end
X_STEEP_START = cfg.x_steep_start
X_STEEP_END = cfg.x_steep_end

OVERLAP = cfg.overlap
SLAB_THICKNESS = cfg.slab_thickness
SLAB_CHANNEL_THICKNESS = cfg.slab_channel_thickness

OVR_TOTAL_THICK = cfg.ovr_total_thick
OVR_LAYER_THICK = cfg.ovr_layer_thick
OVR_KNEE_LENGTH = cfg.ovr_knee_length
OVR_MAX_THICK_WEDGE = cfg.ovr_max_thick_wedge

TOTAL_TIME_YEARS = cfg.total_time_years
CHECKPOINT_EVERY = cfg.checkpoint_every
SMOKE_TEST = cfg.smoke_test
RESTART = cfg.restart


# ============================================================
# BLOCK 1 — Imports & basic parameters
# ============================================================

import underworld as uw
import underworld.function as fn
from underworld import UWGeodynamics as GEO
import numpy as np
import time
import os
import builtins
import re
import slab_geometries
from rheology_G import RHEOLOGY

try:
    ROOT_RANK = uw.mpi.rank
except Exception:
    ROOT_RANK = 0


def _rank0_print(*args, **kwargs):
    if ROOT_RANK == 0:
        builtins.print(*args, **kwargs)


print = _rank0_print
print("[DEBUG] slab_geometries =", slab_geometries.__file__)


start_time = time.time()

GEO.rcParams["initial.nonlinear.tolerance"] = 1e-2
GEO.rcParams['initial.nonlinear.max.iterations'] = 50
GEO.rcParams["nonlinear.tolerance"] = 1e-2
GEO.rcParams['nonlinear.max.iterations'] = 50
GEO.rcParams["swarm.particles.per.cell.3D"] = ppc
GEO.rcParams["popcontrol.particles.per.cell.3D"] = pop

u = GEO.UnitRegistry

print("=== MODEL CONFIGURATION ===")
print(f"  modelname       = {EXP_NAME}")
print(f"  domain X,Y,Z (km) = {X_DOMAIN}, {Y_DOMAIN}, {Z_DOMAIN}")
print(f"  resolution        = {NX} x {NY} x {NZ}")
print(f"  X_BREAK, X_END    = {X_BREAK} km, {X_END} km")
print(f"  dips              = normal={DIP_normal}°, flat={DIP_wedge_flat}°")
print(f"  thickness         = th_flat={TH_FLAT}, th_cent={TH_CENT}")
print(f"  slab thickness    = {SLAB_THICKNESS} km")
print(f"  slab channel thk  = {SLAB_CHANNEL_THICKNESS} km")
print(f"  overriding total  = {OVR_TOTAL_THICK} km (layers {OVR_LAYER_THICK} km)")
print(f"  wedge length      = {OVR_KNEE_LENGTH} km (max thick {OVR_MAX_THICK_WEDGE} km)")
print("===========================")

print("[OK] Loaded imports and basic parameters")


# ============================================================
# BLOCK 2 — Scaling and physical constants
# ============================================================

CONTROL_WHOLE_SLAB = True  # True: ni el fondo pasa -660. False: el techo llega a -660

z_target_bottom = -660.0 * u.kilometer

if CONTROL_WHOLE_SLAB:
    z_cut = z_target_bottom + float(SLAB_THICKNESS) * u.kilometer   # -630 km
else:
    z_cut = z_target_bottom                                         # -660 km



velocity = 1.0 * u.centimeter / u.year
model_length = (X_DOMAIN[1] - X_DOMAIN[0]) * u.kilometer
model_width  = (Y_DOMAIN[1] - Y_DOMAIN[0]) * u.kilometer
model_height = (Z_DOMAIN[1] - Z_DOMAIN[0]) * u.kilometer

refViscosity = 1e19 * u.pascal * u.second
bodyforce = 3300 * u.kilogram / u.metre**3 * 9.81 * u.meter / u.second**2
Tsurf   = 273.15 * u.degK
Tint    = 1573.0 * u.degK

KL = 1000. * u.kilometer        # dejamos la escala interna en 1000 km para no romper no-dim
Kt = KL / velocity
KM = bodyforce * KL**2 * Kt**2
KT = Tint - Tsurf
GEO.scaling_coefficients["[length]"] = KL
GEO.scaling_coefficients["[time]"] = Kt
GEO.scaling_coefficients["[mass]"]= KM
GEO.scaling_coefficients["[temperature]"] = KT

print("[OK] Scaling coefficients set")


# ============================================================
# BLOCK 3 — Create Model & resolution
# ============================================================

x_res, y_res, z_res = NX, NY, NZ

print(x_res,'x',y_res,'x',z_res)

Model = GEO.Model(
    elementRes=(x_res, y_res, z_res),
    minCoord=(X_DOMAIN[0] * u.kilometer,
              Y_DOMAIN[0] * u.kilometer,
              Z_DOMAIN[0] * u.kilometer),
    maxCoord=(X_DOMAIN[1] * u.kilometer,
              Y_DOMAIN[1] * u.kilometer,
              Z_DOMAIN[1] * u.kilometer),
    gravity=(0.0, 0.0, -9.8 * u.meter / u.second**2)
)

print((Model.maxCoord[0]-Model.minCoord[0])/x_res,
      'x',(Model.maxCoord[1]-Model.minCoord[1])/y_res,
      'x',(Model.maxCoord[2]-Model.minCoord[2])/z_res)

print("[OK] Model created with geometry and resolution")



if reGridZ:
    beta = reGrid_beta

    # límites verticales del dominio
    am = Model.mesh.minCoord[2]  # z_min  ~ -1000 km
    bm = Model.mesh.maxCoord[2]  # z_max  ~ 0 km
    
    # coordenadas actuales en z de todos los nodos
    z = Model.mesh.data[:, 2]

    # normalizás a [0,1] bottom→top
    z0 = (z - am) / (bm - am)

    # power-law: beta < 1 -> más puntos cerca de z0 = 1 (top)
    z1 = z0**beta

    # llevás de vuelta a coordenadas físicas
    z_new = am + z1 * (bm - am)

    # deformás la malla
    with Model.mesh.deform_mesh():
        Model.mesh.data[:, 2] = z_new

print("[OK] Model regrid (if enabled)")

dx = (Model.maxCoord[0]-Model.minCoord[0]) / x_res
dy = (Model.maxCoord[1]-Model.minCoord[1]) / y_res
dz = (Model.maxCoord[2]-Model.minCoord[2]) / z_res

print(f"Cell size: {dx.to(u.kilometer):.2f}, {dy.to(u.kilometer):.2f}, {dz.to(u.kilometer):.2f}")

# Solapes para robustecer continuidad geométrica en mallas relativamente gruesas
# (evita "gaps" en las uniones de segmentos por discretización)
X_OVERLAP = OVERLAP * dx.to(u.kilometer)
Y_OVERLAP = OVERLAP * dy.to(u.kilometer)
print(f"[INFO] Overlaps: X_OVERLAP={X_OVERLAP:.2f}, Y_OVERLAP={Y_OVERLAP:.2f}")



# ============================================================
# BLOCK 4 — Output folder and colormaps
# ============================================================

base_outdir = "/gpfs/scratch/upc27/JL/Flat3D/final_runs/IV"

name = (
    f"{EXP_NAME}"
    f"__{VEL_TAG}"
    f"__R{NX}x{NY}x{NZ}"
    f"__DI{int(DIP_IN)}_D0{int(DIP_0)}_ST{int(DIP_STEEP)}"
    f"__TF{int(TH_FLAT)}_TC{int(TH_CENT)}"
)

Model.outputDir = os.path.join(base_outdir, name)
print(f"[INFO] Output directory = {Model.outputDir}")

print("[OK] Output directory set")


# ============================================================
# BLOCK 5 — Base mantle materials
# ============================================================

Mantle       = Model.add_material(name="Mantle",       shape=GEO.shapes.Layer(top=Model.top, bottom=-660. * u.kilometer))
Lowermantle  = Model.add_material(name="LowerMantle",  shape=GEO.shapes.Layer(top=-660. * u.kilometer, bottom=Model.bottom))

print("[OK] Basic mantle materials created")


# ============================================================
# BLOCK 6 — Load slab geometry functions
# ============================================================

from geometry import (
    build_overriding_shape,
    build_slab_shape,
    define_geometry_params,
    run_geometry_debug_profile,
)
from slab_geometries import make_timer

print("[OK] Slab geometry functions imported")


# ============================================================
# BLOCK 7 — Define geometry parameters (X, Y, dips, thickness)
# ============================================================

Model.defaultStrainRate = 5e-15  # 1/s
Model.minViscosity = 1e18  * u.pascal * u.second
Model.maxViscosity = 1e23 * u.pascal * u.second

geom = define_geometry_params(
    u=u,
    x_break_km=X_BREAK,
    x_end_km=X_END,
    x_flat_start_km=X_FLAT_START,
    x_flat_end_km=X_FLAT_END,
    x_steep_start_km=X_STEEP_START,
    x_steep_end_km=X_STEEP_END,
    dip_in_deg=DIP_IN,
    dip_0_deg=DIP_0,
    dip_steep_deg=DIP_STEEP,
    slab_thickness_km=SLAB_THICKNESS,
    z_cut=z_cut,
)

x_break = geom["x_break"]
z_target_top = geom["z_target_top"]
x_end_flat = geom["x_end_flat"]
x_end_30curve = geom["x_end_30curve"]
x_end_trans_flat_30 = geom["x_end_trans_flat_30"]
dip_nodes_flat = geom["dip_nodes_flat"]
dip_nodes_30curve = geom["dip_nodes_30curve"]
th = geom["th"]
slab_thickness_total = geom["slab_thickness_total"]

print(f"[INFO] x_end_{'flat':8s} (z_top={z_target_top}) = {x_end_flat}")
print(f"[INFO] x_end_{'30curve':8s} (z_top={z_target_top}) = {x_end_30curve}")

run_geometry_debug_profile(
    params=geom,
    dip_in_deg=DIP_IN,
    dip_0_deg=DIP_0,
    dip_steep_deg=DIP_STEEP,
)

print(f"[INFO] slab_thickness_total = {slab_thickness_total}")
print(f"[INFO] expected z_bottom (single-layer) ≈ {z_target_top - th}")
print("[INFO] X_NODES_TRANS end =", x_end_trans_flat_30)

print("[OK] Geometry parameters defined")


# ============================================================
# BLOCK 9 — Create slab segments (single-layer slab)
# ============================================================

# rank y barrier
try:
    rank = uw.mpi.rank
    barrier = uw.mpi.barrier
except Exception:
    rank = 0
    barrier = None

timed, TIMINGS, dump_timings = make_timer(
    barrier_fn=barrier,
    printer=(print if rank == 0 else None)  # imprime solo en rank 0
)

print("[INFO] Building single-layer slab as ONE composite shape (halfspaces)…")

slab_shape = build_slab_shape(
    GEO=GEO,
    params=geom,
    slab_thickness_km=SLAB_THICKNESS,
    n_xsub=N_XSUB,
    n_trans_segments=N_TRANS_SEGMENTS,
    x_overlap=X_OVERLAP,
    y_overlap=Y_OVERLAP,
    z_cut=z_cut,
    timed=timed,
)

with timed("Model.add_material(Slab)"):
    Slab = Model.add_material(name="Slab", shape=slab_shape)
    slab_channel_cap = GEO.shapes.Layer(
        top=0.0 * u.kilometer,
        bottom=-float(SLAB_CHANNEL_THICKNESS) * u.kilometer,
    )
    UpperSlabChannel = Model.add_material(
        name="UpperSlabChannel",
        shape=slab_shape & slab_channel_cap,
    )

print("[OK] Single-layer slab created + UpperSlabChannel shape (weak-zone candidate)")

if rank == 0:
    os.makedirs(Model.outputDir, exist_ok=True)
if barrier:
    barrier()

if rank == 0:
    timing_path = os.path.join(Model.outputDir, "timing_slab_build.txt")
    total = sum(dt for _, dt in TIMINGS)
    with open(timing_path, "w") as f:
        f.write("=== SLAB BUILD TIMINGS (seconds) ===\n")
        for lab, dt in TIMINGS:
            f.write(f"{lab:45s} {dt:12.6f}\n")
        f.write(f"{'TOTAL':45s} {total:12.6f}\n")
    print(f"[OK] Slab timing saved to {timing_path}")


# ============================================================
# BLOCK 8 — Overriding plate como UNA sola shape (cuña + parte plana)
# ============================================================

OverridingPlate_shape = build_overriding_shape(
    GEO=GEO,
    u=u,
    params=geom,
    x_domain=X_DOMAIN,
    ovr_knee_length_km=OVR_KNEE_LENGTH,
    x_flat_start_km=X_FLAT_START,
    dip_normal_deg=DIP_normal,
    dip_wedge_flat_deg=DIP_wedge_flat,
    th_flat_km=TH_FLAT,
    th_cent_km=TH_CENT,
    ovr_max_thick_wedge_km=OVR_MAX_THICK_WEDGE,
    n_trans_segments=N_TRANS_SEGMENTS,
)

OverridingPlate = Model.add_material(
    name="OverridingPlate",
    shape=OverridingPlate_shape
)

print("[OK] Overriding plate created as ONE composite shape (30°, 15° + graded transitions)")

print("[CHECK] z_cut =", z_cut)
print("[CHECK] x_end_flat =", x_end_flat, "x_end_30curve =", x_end_30curve)
print("[CHECK] x_end_trans_flat_30 =", x_end_trans_flat_30)
print("[CHECK] expected bottom at cut (single-layer) ≈", z_cut - float(SLAB_THICKNESS)*u.kilometer)
print("[CHECK] z_target_top =", z_target_top, " (should be -630 km if CONTROL_WHOLE_SLAB)")
print("[CHECK] single-layer expected bottom =", z_target_top - float(SLAB_THICKNESS)*u.kilometer)


# ============================================================
# BLOCK 10 — Rheology maps and assignments
# ============================================================

def get_material_by_name(Model, name):
    for m in Model.materials:
        if m.name == name:
            return m
    return None

density_map = {
    0:     1.0   * u.kilogram / u.metre**3,     # Sticky air (no usado)
    3250:  3250. * u.kilogram / u.metre**3,   # overriding menos denso
    3300:  3300. * u.kilogram / u.metre**3,     # Manto / Overriding
    3350:  3350. * u.kilogram / u.metre**3,     # Slab oceánico
}

viscosity_map = {
    0.02:  0.02  * 5e20 * u.pascal * u.second,  # 1e19 Pa·s (weak crust / weak zone A1)
    0.1:   0.1   * 5e20 * u.pascal * u.second,  # 5e19 Pa·s (weak crust / weak zone A2)
    0.2:   0.2   * 5e20 * u.pascal * u.second,  # 1e20 Pa·s (manto superior)
    10:    10.0  * 5e20 * u.pascal * u.second,  # 5e21 Pa·s (manto inferior)
    100:   100.0 * 5e20 * u.pascal * u.second,  # 5e22 Pa·s (litosfera)
    1000:  1000. * 5e20 * u.pascal * u.second,  # Sticky air muy viscoso
}

plasticity_map = {}
plasticity_registry = None
try:
    plasticity_registry = GEO.PlasticityRegistry()
except Exception as exc:
    print(f"[WARNING] PlasticityRegistry unavailable, using local fallback only: {exc}")


def _from_plasticity_registry(registry, key):
    if registry is None:
        return None
    # Distintas APIs posibles según versión de UWGeo
    try:
        return registry[key]
    except Exception:
        pass
    try:
        return registry.get(key, None)
    except Exception:
        pass
    try:
        return getattr(registry, key)
    except Exception:
        pass

    # Nombre estilo atributo: "Foo, 2001 (Bar)" -> "Foo_2001_Bar"
    attr = re.sub(r"[^0-9A-Za-z]+", "_", str(key)).strip("_")
    attr = re.sub(r"_+", "_", attr)
    candidates = [attr]
    # Compatibilidad con typo histórico visto en algunos registros
    if "Beaumont" in attr:
        candidates.append(attr.replace("Beaumont", "Beamount"))
    if "Beamount" in attr:
        candidates.append(attr.replace("Beamount", "Beaumont"))

    for cname in candidates:
        try:
            return getattr(registry, cname)
        except Exception:
            pass

    return None


try:
    plasticity_map["slab_channel_dp_soft"] = GEO.DruckerPrager(
        cohesion=20.0 * u.megapascal,
        cohesionAfterSoftening=10.0 * u.megapascal,
        frictionCoefficient=0.10,
        frictionAfterSoftening=0.03,
        epsilon1=0.10,
        epsilon2=0.50,
    )
except TypeError:
    # Fallback para variantes de API sin softening explícito
    plasticity_map["slab_channel_dp_soft"] = GEO.DruckerPrager(
        cohesion=20.0 * u.megapascal,
        frictionCoefficient=0.10,
    )
except Exception as exc:
    print(f"[WARNING] Could not build default channel plasticity: {exc}")

for mat_name, params in RHEOLOGY.items():
    mat = get_material_by_name(Model, mat_name)
    if mat is None:
        print(f"[WARNING] Material '{mat_name}' not found in Model.materials.")
        continue

    dens_code = params["density"]
    visc_code = params["viscosity"]

    if dens_code not in density_map:
        raise KeyError(f"Density code {dens_code} no definido (material '{mat_name}').")
    if visc_code not in viscosity_map:
        raise KeyError(f"Viscosity code {visc_code} no definido (material '{mat_name}').")

    mat.density   = density_map[dens_code]
    mat.viscosity = viscosity_map[visc_code]
    if "plasticity" in params and params["plasticity"] is not None:
        pdef = params["plasticity"]
        if isinstance(pdef, str):
            pobj = _from_plasticity_registry(plasticity_registry, pdef)
            if pobj is None:
                pobj = plasticity_map.get(pdef, None)
            if pobj is None:
                raise KeyError(
                    f"Plasticity code '{pdef}' no definido para registry/local map (material '{mat_name}')."
                )
            mat.plasticity = pobj
        else:
            mat.plasticity = pdef

print("[OK] Rheology assigned to all materials")


# ============================================================
# BLOCK 11 — BCs + nodeSets (slab / buffer / overriding)
#          + inflow left + controlled outflow right (smooth Z,Y)
# ============================================================

import numpy as np
from mpi4py import MPI
comm = MPI.COMM_WORLD

# OJO: no re-definimos u. En tu script ya tenés: u = GEO.UnitRegistry
# y ya importaste: import underworld as uw ; import underworld.function as fn ; from underworld import UWGeodynamics as GEO

# -------------------------
# Helper: Cuboid 3D via 6 HalfSpaces
# -------------------------
def cuboid_3d(GEO, *, xmin, xmax, ymin, ymax, zmin, zmax):
    """
    Shape 3D tipo caja [xmin,xmax]x[ymin,ymax]x[zmin,zmax]
    usando intersección de 6 HalfSpaces.

    Convención UWGeo HalfSpace: se queda con el lado opuesto al vector normal.
    """
    hs_xmin = GEO.shapes.HalfSpace(normal=(-1., 0., 0.), origin=(xmin, 0.*xmin, 0.*xmin))  # x >= xmin
    hs_xmax = GEO.shapes.HalfSpace(normal=( 1., 0., 0.), origin=(xmax, 0.*xmax, 0.*xmax))  # x <= xmax

    hs_ymin = GEO.shapes.HalfSpace(normal=(0., -1., 0.), origin=(0.*ymin, ymin, 0.*ymin))  # y >= ymin
    hs_ymax = GEO.shapes.HalfSpace(normal=(0.,  1., 0.), origin=(0.*ymax, ymax, 0.*ymax))  # y <= ymax

    hs_zmin = GEO.shapes.HalfSpace(normal=(0., 0., -1.), origin=(0.*zmin, 0.*zmin, zmin))  # z >= zmin
    hs_zmax = GEO.shapes.HalfSpace(normal=(0., 0.,  1.), origin=(0.*zmax, 0.*zmax, zmax))  # z <= zmax

    return hs_xmin & hs_xmax & hs_ymin & hs_ymax & hs_zmin & hs_zmax


# -------------------------
# Smooth helpers (ND functions)
# -------------------------
def clamp01(q):
    return fn.misc.min(1.0, fn.misc.max(0.0, q))

def smoothstep01(t):
    return t*t*(3.0 - 2.0*t)


# -------------------------
# Velocidades (ND) para BCs y nodeSets
# -------------------------
v_slab_nd = float(GEO.nd(2.5 * u.centimeter / u.year))
v_over_nd = float(GEO.nd(-5.0 * u.centimeter / u.year))  # negativo


# -------------------------
# Geometría (dimensional)
# -------------------------
xL = X_DOMAIN[0] * u.kilometer
xB = X_BREAK      * u.kilometer
xR = X_DOMAIN[1] * u.kilometer
xO0 = 1700.0 * u.kilometer

y0 = Y_DOMAIN[0] * u.kilometer
y1 = Y_DOMAIN[1] * u.kilometer

# espesores “driving” (dimensional, positivos; z es negativo hacia abajo)
SLAB_DRV_THICK = float(SLAB_THICKNESS)  * u.kilometer
OVR_DRV_THICK  = float(OVR_TOTAL_THICK) * u.kilometer
eps = 1e-6 * u.kilometer
SLAB_DRIVER_XSPAN = 250.0 * u.kilometer
x_slab_driver_max = min(xL + SLAB_DRIVER_XSPAN, xB - eps)


# -------------------------
# nodeSets shapes (volumen superior)
# -------------------------
slab_driver_shape = cuboid_3d(
    GEO,
    xmin=xL,
    xmax=x_slab_driver_max,
    ymin=y0,
    ymax=y1,
    zmin=-SLAB_DRV_THICK,
    zmax=0.0 * u.kilometer
)

over_driver_shape = cuboid_3d(
    GEO,
    xmin=xO0 + eps,
    xmax=xR,
    ymin=y0,
    ymax=y1,
    zmin=-OVR_DRV_THICK,
    zmax=0.0 * u.kilometer
)


# -------------------------
# Taper en Y (ventana suave) — evita “clavar” todo en y=0 y y=Ymax
# -------------------------
y = Model.y  # ND


# -------------------------
# Taper en Z (profundidad) para los drivers volumétricos (nodeSets)
# -------------------------
z = Model.z  # ND, negativo hacia abajo

taperZ = 1.0


# -------------------------
# nodeSet velocity functions (ND)
# -------------------------
vx_slab_fn = v_slab_nd * taperZ
vx_over_fn = v_over_nd * taperZ


# -------------------------
# Left wall: inflow slab + outflow compensatorio en manto
# -------------------------
H_slab_nd = float(GEO.nd(SLAB_DRV_THICK))
H_ovr_nd = float(GEO.nd(OVR_DRV_THICK))
d_max_nd = float(GEO.nd(990.0 * u.kilometer))
wedge_out_nd = float(GEO.nd(40.0 * u.kilometer))

tZ_left_in = fn.branching.conditional([
    (((-z) <= H_slab_nd), 1.0),
    (True, 0.0),
])

tZ_left_mantle = fn.branching.conditional([
    (((-z) <= H_slab_nd), 0.0),
    (((-z) >= d_max_nd), 0.0),
    (True, 1.0),
])
t_up_lm = clamp01(((-z) - H_slab_nd) / max(1e-30, wedge_out_nd))
t_dn_lm = clamp01((d_max_nd - (-z)) / max(1e-30, wedge_out_nd))
tZ_left_mantle = tZ_left_mantle * smoothstep01(t_up_lm) * smoothstep01(t_dn_lm)

H_left_mantle_nd = max(float(GEO.nd(1.0 * u.kilometer)), (d_max_nd - H_slab_nd))
H_left_eff_nd = max(float(GEO.nd(1.0 * u.kilometer)), (H_left_mantle_nd - wedge_out_nd))
v_left_out_mag_nd = abs(float(v_slab_nd)) * (float(H_slab_nd) / float(H_left_eff_nd))

vx_left_fn = v_slab_nd * tZ_left_in - v_left_out_mag_nd * tZ_left_mantle


# -------------------------
# Right wall: outflow compensatorio en manto por ingreso overriding
# -------------------------
d_base_nd = H_ovr_nd
tZ_right_mantle = fn.branching.conditional([
    ((-z) <= d_base_nd, 0.0),
    ((-z) >= d_max_nd, 0.0),
    (True, 1.0),
])
t_up_rm = clamp01(((-z) - d_base_nd) / max(1e-30, wedge_out_nd))
t_dn_rm = clamp01((d_max_nd - (-z)) / max(1e-30, wedge_out_nd))
tZ_right_mantle = tZ_right_mantle * smoothstep01(t_up_rm) * smoothstep01(t_dn_rm)

H_right_mantle_nd = max(float(GEO.nd(1.0 * u.kilometer)), (d_max_nd - d_base_nd))
H_right_eff_nd = max(float(GEO.nd(1.0 * u.kilometer)), (H_right_mantle_nd - wedge_out_nd))
v_right_out_mag_nd = abs(float(v_over_nd)) * (float(H_ovr_nd) / float(H_right_eff_nd))
vx_right_fn = v_right_out_mag_nd * tZ_right_mantle


# -------------------------
# BCs globales + nodeSets
# -------------------------
Model.set_velocityBCs(
    left=[vx_left_fn,   None, None],
    right=[vx_right_fn, None, None],
    front=[None, 0.0, None],
    back=[None,  0.0, None],
    top=[None, None, 0.0],
    bottom=[None, None, 0.0],
)

# -------------------------
# Diagnóstico rápido: tamaños globales de nodeSets (útil en MPI)
# -------------------------
coords = Model.mesh.data
mask_slab = slab_driver_shape.evaluate(coords)
mask_over = over_driver_shape.evaluate(coords)

Nslab = comm.allreduce(int(np.sum(mask_slab)), op=MPI.SUM)
Nover = comm.allreduce(int(np.sum(mask_over)), op=MPI.SUM)

if uw.mpi.rank == 0:
    print("[OK] BLOCK 11: BCs only on walls set (no volumetric nodeSets).")
    print(
        f"      v_slab_nd={v_slab_nd:.4e}  v_over_nd={v_over_nd:.4e}  "
        f"v_left_out_mantle_nd={v_left_out_mag_nd:.4e}  v_right_out_mantle_nd={v_right_out_mag_nd:.4e}"
    )
    print(f"      nodeSets global counts: slab={Nslab}  over={Nover}")

## TEST 01

Model.swarm.particleEscape = True
Model.solver.set_inner_method("mumps")

Model.solver.options.scr.ksp_type = "fgmres"
Model.solver.options.scr.ksp_rtol = "1e-5"

# # ### TEST 02

# Model.swarm.particleEscape = True
# Model.solver.set_inner_method("mg")

# Model.solver.options.scr.ksp_type = "fgmres"
# Model.solver.options.A11.ksp_rtol = "1e-6"
# Model.solver.options.scr.ksp_rtol = "1e-6"


print("[OK] Boundary conditions and solver settings applied")


# ============================================================
# BLOCK 12 — Outputs and initial steady-state
# ============================================================

outputfields=['temperature',
              'pressureField',
              'strainRateField',
              'velocityField',
              'projTimeField',
              'projMaterialField',
              'projViscosityField',
              'projStressField',
              'projMeltField',
              'projPlasticStrain',
              'projDensityField',
              'projStressTensor']

GEO.rcParams['default.outputs'] = outputfields

### remove variables from output (NO CAMBIAR)
#GEO.rcParams["default.outputs"].remove("temperature")
#GEO.rcParams["default.outputs"].remove("projMeltField")
#GEO.rcParams["default.outputs"].remove("projTimeField")
#GEO.rcParams["default.outputs"].remove("projPlasticStrain")

print("[OK] Output fields registered")
print("[INFO] Starting steady-state initialisation")

# --- TIMERS (global)
t0_total = time.perf_counter()

# init_model
t_init0 = time.perf_counter()
Model.init_model(pressure="lithostatic")
init_model_s = time.perf_counter() - t_init0

print("[OK] Volumetric velocities re-imposed after init_model.")

print("[OK] Steady-state initial condition created")

if SMOKE_TEST:
    print("[SMOKE_TEST] OK: build + init_model. Corto antes de run_for.")
    import sys
    sys.exit(0)

# ============================================================
# BLOCK 13 — Run model
# ============================================================

print('Running...')

Total_Time = TOTAL_TIME_YEARS
interval   = CHECKPOINT_EVERY

t_run0 = time.perf_counter()

if RESTART:
    print(">>> Restarting model from last checkpoint")
    Model.run_for(
        Total_Time * u.years,
        checkpoint_interval=interval * u.years,
        restartStep=-1
    )
else:
    print(">>> Running model from initial conditions")
    Model.run_for(
        Total_Time * u.years,
        checkpoint_interval=interval * u.years
    )

run_for_s = time.perf_counter() - t_run0

total_runtime_s = time.perf_counter() - t0_total

#Model.run_for(nstep = 3, checkpoint_interval = 1)

print("[OK] Model run completed successfully")

end_time = time.time()
elapsed = end_time - start_time
print(f"[TOTAL RUNTIME] Total execution time: {elapsed/60:.2f} minutes ({elapsed:.1f} seconds)")

# ============================================
# SAVE MODEL CONFIGURATION TO TEXT FILE
# ============================================

config_filename = f"model_C{EXP_NAME}_config.txt"
config_path = os.path.join(Model.outputDir, config_filename)

runtime_seconds = total_runtime_s

if ROOT_RANK == 0:
    with open(config_path, "w") as f:

        f.write("=== MODEL CONFIGURATION ===\n")
        f.write(f"modelname           = {EXP_NAME}\n")
        f.write(f"domain X,Y,Z (km)     = {X_DOMAIN}, {Y_DOMAIN}, {Z_DOMAIN}\n")
        f.write(f"resolution            = {NX} x {NY} x {NZ}\n")
        f.write(f"X_BREAK, X_END (km)   = {X_BREAK}, {X_END}\n")
        f.write(f"dips (normal/flat)    = {DIP_normal}°, {DIP_wedge_flat}°\n")
        f.write(f"dip_nodes_flat        = {dip_nodes_flat}\n")
        f.write(f"dip_nodes_30curve     = {dip_nodes_30curve}\n")
        f.write(f"slab thickness (km)   = {SLAB_THICKNESS}\n")
        f.write(f"slab channel thk (km) = {SLAB_CHANNEL_THICKNESS}\n")
        f.write(f"overriding total (km) = {OVR_TOTAL_THICK} (layers {OVR_LAYER_THICK} km)\n")
        f.write(f"wedge length (km)     = {OVR_KNEE_LENGTH} (max thick {OVR_MAX_THICK_WEDGE} km)\n")

        f.write("\n=== MATERIAL RHEOLOGIES ===\n")
        for mat in Model.materials:
            name = mat.name
            rho  = mat.density
            visc = mat.viscosity

            f.write(f"\nMaterial: {name}\n")

            # --- DENSITY ---
            if hasattr(rho, "magnitude"):
                f.write(f"  density      = {rho.magnitude:.3e} {rho.units}\n")
            else:
                f.write(f"  density      = {rho}\n")

            # --- VISCOSITY ---
            # Si es constante con unidades (pint), imprimimos magnitud + unidades
            if hasattr(visc, "magnitude"):
                f.write(f"  viscosity    = {visc.magnitude:.3e} {visc.units}\n")
            # Si es un float/numérico sin unidades
            elif isinstance(visc, (int, float)):
                f.write(f"  viscosity    = {visc:.3e}\n")
            else:
                # Funciones de UW / viscosidad no constante
                f.write(f"  viscosity    = <Underworld function>\n")

            # Info extra desde el diccionario RHEOLOGY si existe
            if name in RHEOLOGY:
                f.write("  -- values from RHEOLOGY dict --\n")
                for k, v in RHEOLOGY[name].items():
                    f.write(f"     {k:12s} = {v}\n")

        f.write("\n=== RUNTIME ===\n")
        f.write(f"total_seconds        = {runtime_seconds:.2f}\n")
        f.write(f"total_minutes        = {runtime_seconds/60:.2f}\n")
        f.write(f"total_hours          = {runtime_seconds/3600:.2f}\n")

        f.write("\n===========================\n")

    print(f"[OK] Configuration saved to {config_path}")
    print(f"Total runtime: {runtime_seconds:.2f} seconds ({runtime_seconds/60:.1f} min)")


# ============================================================
# RUN SUMMARY (para comparar eficiencia entre corridas)
# ============================================================
import socket
from datetime import datetime

def _env(k, default=""):
    return os.environ.get(k, default)

def _safe_int(x, default=None):
    try:
        return int(x)
    except Exception:
        return default

def _read_slab_timing_total(path):
    """Lee TOTAL del timing_slab_build.txt si existe."""
    if not os.path.exists(path):
        return None
    total = None
    try:
        with open(path, "r") as f:
            for line in f:
                if line.strip().startswith("TOTAL"):
                    # formato: TOTAL <seconds>
                    parts = line.split()
                    total = float(parts[-1])
                    break
    except Exception:
        return None
    return total

# --- identificar script
try:
    script_name = os.path.basename(__file__)
except Exception:
    script_name = "unknown_script.py"

# --- Slurm / MPI info
slurm_job_id   = _env("SLURM_JOB_ID", "")
slurm_job_name = _env("SLURM_JOB_NAME", "")
slurm_nnodes   = _env("SLURM_NNODES", "")
slurm_ntasks   = _env("SLURM_NTASKS", "")
slurm_tpn      = _env("SLURM_TASKS_PER_NODE", "")
slurm_cpt      = _env("SLURM_CPUS_PER_TASK", "")
omp_threads    = _env("OMP_NUM_THREADS", "")

try:
    mpi_size = uw.mpi.size
except Exception:
    mpi_size = None

# --- rcParams relevantes (ojo: guardamos los valores efectivos)
ppc3d  = GEO.rcParams.get("swarm.particles.per.cell.3D", None)
pop3d  = GEO.rcParams.get("popcontrol.particles.per.cell.3D", None)

# --- grid + celdas/partículas
nCells = int(NX) * int(NY) * int(NZ)
nParts_est = None
try:
    nParts_est = int(nCells * int(ppc3d))
except Exception:
    pass

# --- tiempos (si querés separarlos, medilos con timers alrededor de init_model/run_for; ver nota abajo)
total_seconds = runtime_seconds

# --- slab timing si existe
slab_timing_path = os.path.join(Model.outputDir, "timing_slab_build.txt")
slab_total_s = _read_slab_timing_total(slab_timing_path)

# --- output summary filename
summary_name = f"RUNSUMMARY__{os.path.splitext(script_name)[0]}__{EXP_NAME}.txt"
summary_path = os.path.join(Model.outputDir, summary_name)

# --- (opcional) carpeta índice global para que después me pases solo los summaries
global_summary_dir = os.path.join(base_outdir, "_run_summaries")
if rank == 0:
    os.makedirs(global_summary_dir, exist_ok=True)
global_summary_path = os.path.join(global_summary_dir, summary_name)

if rank == 0:
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    host = socket.gethostname()

    # sizes con unidades (ya los tenés como pint)
    dx_km = dx.to(u.kilometer)
    dy_km = dy.to(u.kilometer)
    dz_km = dz.to(u.kilometer)

    lines = []
    lines.append("=== RUN SUMMARY ===")
    lines.append(f"timestamp_utc           = {now}")
    lines.append(f"host                    = {host}")
    lines.append("")
    lines.append("=== IDENTIDAD ===")
    lines.append(f"script                  = {script_name}")
    lines.append(f"EXP_NAME                = {EXP_NAME}")
    lines.append(f"outputDir               = {Model.outputDir}")
    lines.append(f"slab_geometries_path    = {slab_geometries.__file__}")
    lines.append("")
    lines.append("=== SLURM / MPI ===")
    lines.append(f"SLURM_JOB_ID            = {slurm_job_id}")
    lines.append(f"SLURM_JOB_NAME          = {slurm_job_name}")
    lines.append(f"SLURM_NNODES            = {slurm_nnodes}")
    lines.append(f"SLURM_NTASKS            = {slurm_ntasks}")
    lines.append(f"SLURM_TASKS_PER_NODE    = {slurm_tpn}")
    lines.append(f"SLURM_CPUS_PER_TASK     = {slurm_cpt}")
    lines.append(f"OMP_NUM_THREADS         = {omp_threads}")
    lines.append(f"uw_mpi_size             = {mpi_size}")
    lines.append("")
    lines.append("=== DISCRETIZACION ===")
    lines.append(f"NX,NY,NZ                = {NX},{NY},{NZ}")
    lines.append(f"dx,dy,dz (km)           = {dx_km:.4f}, {dy_km:.4f}, {dz_km:.4f}")
    lines.append(f"nCells                  = {nCells}")
    lines.append(f"ppc3D                   = {ppc3d}")
    lines.append(f"popcontrol3D            = {pop3d}")
    lines.append(f"nParticles_est          = {nParts_est}")
    lines.append("")
    lines.append("=== GEOMETRIA ===")
    lines.append(f"N_TRANS_SEGMENTS        = {N_TRANS_SEGMENTS}")
    lines.append(f"N_XSUB                  = {N_XSUB}")
    lines.append(f"X_OVERLAP (km)          = {X_OVERLAP}")
    lines.append(f"Y_OVERLAP (km)          = {Y_OVERLAP}")
    lines.append(f"SLAB_THICKNESS (km)     = {SLAB_THICKNESS}")
    lines.append(f"SLAB_CHANNEL_THK (km)   = {SLAB_CHANNEL_THICKNESS}")
    lines.append("")
    lines.append("=== TIMINGS ===")
    lines.append(f"slab_build_total_s      = {slab_total_s}")
    lines.append(f"init_model_s            = {init_model_s:.3f}")
    lines.append(f"run_for_s               = {run_for_s:.3f}")
    lines.append(f"total_runtime_s         = {total_runtime_s:.3f}")
    lines.append(f"total_runtime_min       = {total_runtime_s/60:.2f}")
    lines.append("")
    lines.append("=== NOTAS ===")
    lines.append("Si queres, puedo comparar runs usando: tiempo_total / nCells / nParticles_est y ver scaling vs NNODES.")

    content = "\n".join(lines) + "\n"

    # write in outputDir
    with open(summary_path, "w") as f:
        f.write(content)

    # write global index copy
    with open(global_summary_path, "w") as f:
        f.write(content)

    print(f"[OK] Run summary saved to: {summary_path}")
    print(f"[OK] Run summary copied to: {global_summary_path}")
