"""
Composite Analysis Module

This module provides functions for computing azimuthal (angular) averages
and creating composite fields for tropical cyclone structure analysis.

Main functions:
- azimuthal_mean: Compute azimuthal average to obtain axisymmetric structure
- azimuthal_anomaly: Compute deviations from azimuthal mean (asymmetries)
"""

from .azimuthal import (
    azimuthal_mean,
    azimuthal_anomaly,
)

__all__ = [
    'azimuthal_mean',
    'azimuthal_anomaly',
]
