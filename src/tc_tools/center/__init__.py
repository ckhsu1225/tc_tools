"""
Typhoon Center Finding Module

This module provides various methods for locating tropical cyclone centers
in VVM simulation data. Each method has different strengths depending on
the storm characteristics and available data.

Available methods:
- Streamfunction: Most robust for weak vortices (recommended for idealized sims)
- Maximum vorticity: Simple and fast, but sensitive to noise
- Vorticity centroid: Weighted center of mass, more stable for asymmetric vortices

Main functions:
- find_center_by_streamfunction: Primary method using streamfunction extremum
- find_center_by_vorticity_maximum: Alternative method using vorticity maximum
- find_center_by_vorticity_centroid: Alternative method using vorticity centroid (weighted)

All functions return xr.Dataset with time dimension for easy analysis and export.
"""

from .streamfunction import (
    compute_streamfunction,
    find_center_by_streamfunction,
)
from .vorticity import (
    smooth_vorticity,
    find_center_by_vorticity_maximum,
    find_center_by_vorticity_centroid,
)

__all__ = [
    # Streamfunction method
    'compute_streamfunction',
    'find_center_by_streamfunction',
    # Vorticity methods
    'smooth_vorticity',
    'find_center_by_vorticity_maximum',
    'find_center_by_vorticity_centroid',
]
