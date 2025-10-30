"""
Intensity Metrics Module

This module provides functions to compute tropical cyclone intensity metrics
based on wind fields. Since VVM uses height coordinates and doesn't output
pressure, all intensity metrics are wind-based.

Main functions:
- compute_intensity: Compute maximum wind speed and radius of maximum wind
"""

from .wind import (
    compute_intensity,
)

__all__ = [
    'compute_intensity',
]
