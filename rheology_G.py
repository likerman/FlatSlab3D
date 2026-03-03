# -*- coding: utf-8 -*-
"""
Rheology definitions for model C10 / C13
----------------------------------------
This file provides the RHEOLOGY dictionary used by test_C10.py / test_C13.py.

Each entry:
    "MaterialName": {
        "density":  <code in density_map>,
        "viscosity": <code in viscosity_map>,
    }

Density codes disponibles:
    0     -> sticky air (no usado acá)
    3300  -> mantle / overriding plate
    3350  -> oceanic slab

Viscosity codes disponibles:
    0.02  -> 1e19 Pa·s   (weak)
    0.1   -> 5e19 Pa·s   (weak/intermediate)
    0.2   -> 1e20 Pa·s   (upper mantle)
    10    -> 5e21 Pa·s   (lower mantle)
    100   -> 5e22 Pa·s   (lithosphere: overriding + slab)
    1000  -> sticky air muy viscoso
"""

RHEOLOGY = {
    # Mantle
    "Mantle": {
        "density":   3300,
        "viscosity": 0.2,   # ~1e20 Pa·s
    },

    "LowerMantle": {
        "density":   3300,
        "viscosity": 10,    # ~5e21 Pa·s
    },

    # Overriding plate (single-layer, toda la cuña + parte plana)
    "OverridingPlate": {
        "density":   3250,
        "viscosity": 100,   # litosfera continental
    },

    # Slab (single-layer, toda la geometría continua)
    "Slab": {
        "density":   3350,  # slab oceánico más denso
        "viscosity": 100,   # litosfera oceánica fuerte
    },
    # Canal débil en la parte superior del slab
    "UpperSlabChannel": {
        "density":   3350,
        "viscosity": 0.02,  # weak zone ~1e19 Pa·s
        "plasticity": "Huismans and Beaumont, 2007 (WeakCrust)",
    },
}
