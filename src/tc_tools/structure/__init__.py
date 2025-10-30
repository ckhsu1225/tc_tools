"""
Structure Metrics Module

This module provides functions to compute tropical cyclone size and
structural metrics based on wind fields.

Main functions:
- compute_size_metrics: Compute wind radii (R34, R50, R64) and other size metrics
- compute_wind_radius: Compute radius where wind exceeds threshold
"""

from .size import (
    compute_size_metrics,
    compute_wind_radius,
)

__all__ = [
    'compute_size_metrics',
    'compute_wind_radius',
]
