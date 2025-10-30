"""
Utility Functions Module

This module provides common utility functions for distance calculations
and other helper operations.

Main functions:
- cartesian_distance: Compute Euclidean distance in Cartesian coordinates
- periodic_cartesian_distance: Compute distance with periodic boundary conditions
"""

from .distance import (
    cartesian_distance,
    periodic_cartesian_distance,
)

__all__ = [
    'cartesian_distance',
    'periodic_cartesian_distance',
]
