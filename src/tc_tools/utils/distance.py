"""
Distance calculation utilities

This module provides functions for calculating distances between points
in Cartesian coordinates.
"""

from typing import Union, Tuple
import numpy as np
import numpy.typing as npt


def _periodic_displacement(
    x1: Union[float, npt.NDArray[np.floating]],
    y1: Union[float, npt.NDArray[np.floating]],
    x2: Union[float, npt.NDArray[np.floating]],
    y2: Union[float, npt.NDArray[np.floating]],
    domain_size_x: float,
    domain_size_y: float
) -> Tuple[Union[float, npt.NDArray[np.floating]], Union[float, npt.NDArray[np.floating]]]:
    """
    Calculate periodic displacement components (internal helper function).

    This function computes the shortest displacement (dx, dy) between points
    considering periodic boundary conditions. It applies the minimum image
    convention.

    Parameters
    ----------
    x1, y1 : float or array-like
        Coordinates of first point(s)
    x2, y2 : float or array-like
        Coordinates of second point(s)
    domain_size_x : float
        Size of the periodic domain in x direction
    domain_size_y : float
        Size of the periodic domain in y direction

    Returns
    -------
    dx, dy : float or array-like
        Wrapped displacement components (same units as input)
        These are in the range [-domain_size/2, domain_size/2]
    """
    # Calculate direct displacement
    dx = x2 - x1
    dy = y2 - y1

    # Apply periodic boundary conditions (minimum image convention)
    # This wraps displacements to the range [-domain_size/2, domain_size/2]
    dx = (dx + domain_size_x/2) % domain_size_x - domain_size_x/2
    dy = (dy + domain_size_y/2) % domain_size_y - domain_size_y/2

    return dx, dy


def cartesian_distance(
    x1: Union[float, npt.NDArray[np.floating]],
    y1: Union[float, npt.NDArray[np.floating]],
    x2: Union[float, npt.NDArray[np.floating]],
    y2: Union[float, npt.NDArray[np.floating]]
) -> Union[float, npt.NDArray[np.floating]]:
    """
    Calculate Euclidean distance in Cartesian coordinates.

    Parameters
    ----------
    x1, y1 : float or array-like
        Coordinates of first point(s)
    x2, y2 : float or array-like
        Coordinates of second point(s)

    Returns
    -------
    distance : float or array-like
        Euclidean distance (same units as input coordinates)

    Examples
    --------
    >>> from tc_tools.utils import cartesian_distance
    >>>
    >>> # Distance in Cartesian coordinates
    >>> d = cartesian_distance(0.0, 0.0, 3.0, 4.0)
    >>> print(f"Distance: {d:.1f}")  # Should be 5.0
    >>>
    >>> # Vectorized
    >>> import numpy as np
    >>> x = np.array([0, 1, 2])
    >>> y = np.array([0, 1, 2])
    >>> d = cartesian_distance(0, 0, x, y)

    Notes
    -----
    - Simple Pythagorean distance: sqrt((x2-x1)^2 + (y2-y1)^2)
    - Works for any Cartesian coordinate system (km, meters, etc.)
    - Input and output units are the same
    """
    dx = x2 - x1
    dy = y2 - y1
    distance = np.sqrt(dx**2 + dy**2)
    return distance


def periodic_cartesian_distance(
    x1: Union[float, npt.NDArray[np.floating]],
    y1: Union[float, npt.NDArray[np.floating]],
    x2: Union[float, npt.NDArray[np.floating]],
    y2: Union[float, npt.NDArray[np.floating]],
    domain_size_x: float,
    domain_size_y: float
) -> Union[float, npt.NDArray[np.floating]]:
    """
    Calculate distance in Cartesian coordinates with periodic boundaries.

    This function computes the shortest distance between points considering
    that the domain wraps around at its boundaries (periodic boundary conditions).
    This is essential for models like VVM that use doubly-periodic domains.

    Parameters
    ----------
    x1, y1 : float or array-like
        Coordinates of first point(s)
    x2, y2 : float or array-like
        Coordinates of second point(s)
    domain_size_x : float
        Size of the periodic domain in x direction
    domain_size_y : float
        Size of the periodic domain in y direction

    Returns
    -------
    distance : float or array-like
        Shortest distance considering periodic boundaries (same units as input)

    Examples
    --------
    >>> from tc_tools.utils import periodic_cartesian_distance
    >>> import numpy as np
    >>>
    >>> # Domain size: 200 km x 200 km
    >>> domain_x = 200000.0  # meters
    >>> domain_y = 200000.0  # meters
    >>>
    >>> # Point near left boundary
    >>> x1, y1 = 10000.0, 100000.0
    >>> # Point near right boundary (would be far without wrapping)
    >>> x2, y2 = 195000.0, 100000.0
    >>>
    >>> # Without periodic boundaries, distance would be ~185 km
    >>> d_regular = np.sqrt((x2-x1)**2 + (y2-y1)**2) / 1000
    >>> print(f"Regular distance: {d_regular:.1f} km")  # 185 km
    >>>
    >>> # With periodic boundaries, it wraps around to ~15 km
    >>> d_periodic = periodic_cartesian_distance(
    ...     x1, y1, x2, y2, domain_x, domain_y
    ... ) / 1000
    >>> print(f"Periodic distance: {d_periodic:.1f} km")  # 15 km

    Notes
    -----
    - Uses the minimum image convention: finds the shortest distance
      considering all periodic images of the points
    - The distance formula applies periodic wrapping:
      dx = (dx + L/2) mod L - L/2, where L is the domain size
    - Essential for VVM simulations with double periodic boundaries
    - Input and output units are the same (typically meters)

    See Also
    --------
    cartesian_distance : Regular (non-periodic) Cartesian distance
    """
    # Get wrapped displacement components using helper function
    dx, dy = _periodic_displacement(x1, y1, x2, y2, domain_size_x, domain_size_y)

    # Calculate distance from wrapped displacements
    distance = np.sqrt(dx**2 + dy**2)
    return distance


__all__ = [
    'cartesian_distance',
    'periodic_cartesian_distance',
]
