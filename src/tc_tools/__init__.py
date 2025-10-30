"""
TC Tools - Tropical Cyclone Analysis Tools for VVM Simulations

A Python package for analyzing tropical cyclone structure and evolution
in VVM (Vector Vorticity equation cloud-resolving Model) simulations.

Main modules:
- center: Typhoon center finding and tracking algorithms
- transform: Coordinate transformations and wind decomposition
- composite: Azimuthal averaging and asymmetry analysis
- intensity: Intensity metrics (wind-based)
- structure: Size and structural metrics
- utils: Utility functions and helpers

Typical workflow:
1. Load VVM data using vvm_reader
2. Find typhoon center using center finding algorithms
3. Transform to cylindrical coordinates centered on storm
4. Decompose winds into tangential and radial components
5. Compute azimuthal mean for axisymmetric structure
6. Analyze intensity and size metrics
"""

__version__ = "0.1.0"

# Import main functions for convenient access
from .center import (
    compute_streamfunction,
    find_center_by_streamfunction,
    smooth_vorticity,
    find_center_by_vorticity_maximum,
    find_center_by_vorticity_centroid,
)

from .transform import (
    transform_to_cylindrical,
    compute_tangential_radial_winds,
)

from .composite import (
    azimuthal_mean,
    azimuthal_anomaly,
)

from .intensity import (
    compute_intensity,
)

from .structure import (
    compute_size_metrics,
    compute_wind_radius,
)

from .utils import (
    cartesian_distance,
)

__all__ = [
    # Center finding
    'compute_streamfunction',
    'find_center_by_streamfunction',
    'smooth_vorticity',
    'find_center_by_vorticity_maximum',
    'find_center_by_vorticity_centroid',

    # Coordinate transformation
    'transform_to_cylindrical',
    'compute_tangential_radial_winds',

    # Composite analysis
    'azimuthal_mean',
    'azimuthal_anomaly',

    # Intensity metrics
    'compute_intensity',

    # Structure metrics
    'compute_size_metrics',
    'compute_wind_radius',

    # Utilities
    'cartesian_distance',
]
