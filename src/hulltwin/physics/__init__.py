"""Physics core: boundary-layer, friction, propeller and wave models.

All models are deliberately simple, interpretable and differentiable-friendly
(numpy-first, vectorised) — they serve both as the synthetic-data generator
and as the residual terms of the PINN loss.
"""

from hulltwin.physics.friction import cf_ittc57, cf_local, roughness_allowance
from hulltwin.physics.karman import karman_residual, solve_momentum_boundary_layer
from hulltwin.physics.propeller import (
    PropellerParams,
    kq,
    kt,
    open_water_efficiency,
    torque_residual,
)
from hulltwin.physics.waves import added_resistance_aw

__all__ = [
    "PropellerParams",
    "added_resistance_aw",
    "cf_ittc57",
    "cf_local",
    "karman_residual",
    "kt",
    "kq",
    "open_water_efficiency",
    "roughness_allowance",
    "solve_momentum_boundary_layer",
    "torque_residual",
]
