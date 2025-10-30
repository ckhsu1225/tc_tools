"""
Coordinate Transformation Module

This module provides functions to transform VVM data from Cartesian coordinates
to storm-relative cylindrical coordinates (radius, azimuth) and decompose winds
into tangential and radial components.

Main functions:
- transform_to_cylindrical: Convert to (r, azimuth) for single or multiple time steps
- compute_tangential_radial_winds: Decompose (u, v) into (v_tan, v_rad)
"""

from .coordinates import (
    transform_to_cylindrical,
)
from .winds import (
    compute_tangential_radial_winds,
)

__all__ = [
    # Coordinate transformation
    'transform_to_cylindrical',
    # Wind decomposition
    'compute_tangential_radial_winds',
]
