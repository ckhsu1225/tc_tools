"""
Storm-relative coordinate transformations

This module provides functions to transform data from Cartesian (x, y) or (lon, lat)
coordinates to cylindrical coordinates (r, azimuth) centered on the storm center.

The cylindrical coordinate system is essential for:
- Computing tangential and radial winds
- Azimuthal averaging to obtain axisymmetric structure
- Analyzing storm structure in a natural coordinate system

Coordinate definitions:
- r (radius): Distance from storm center (km or degrees)
- azimuth: Angle from north, measured clockwise (degrees, 0-360)
  * 0° = North, 90° = East, 180° = South, 270° = West
"""

import numpy as np
import xarray as xr


def _get_domain_size(ds: xr.Dataset) -> tuple[float, float]:
    """
    Get domain size from dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        Input dataset with 'xc' and 'yc' coordinates

    Returns
    -------
    domain_size_x : float
        Domain size in x direction (meters)
    domain_size_y : float
        Domain size in y direction (meters)
    """
    xc = ds.xc.values
    yc = ds.yc.values

    # Calculate domain size (assuming uniform spacing)
    dx = xc[1] - xc[0]
    dy = yc[1] - yc[0]
    domain_size_x = len(xc) * dx
    domain_size_y = len(yc) * dy

    return domain_size_x, domain_size_y


def transform_to_cylindrical(
    ds: xr.Dataset,
    centers: xr.Dataset
) -> xr.Dataset:
    """
    Transform dataset to cylindrical coordinates centered on storm.

    This function computes radius and azimuth for each time step based on
    the corresponding storm center. Works for both single and multiple time steps.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset with dimensions (time, ..., lat, lon).
        Must contain Cartesian coordinates 'xc', 'yc'.
    centers : xr.Dataset
        Storm centers from find_center_* functions.
        Must contain:
        - center_xc (time,) : x-coordinates in meters
        - center_yc (time,) : y-coordinates in meters
        - center_lon (time,) : longitudes
        - center_lat (time,) : latitudes

    Returns
    -------
    ds_cylindrical : xr.Dataset
        Dataset with added data variables:
        - r (time, lat, lon) : Distance from storm center in km
        - azimuth (time, lat, lon) : Angle from north (0-360°)

    Examples
    --------
    >>> import tc_tools as tct
    >>> import vvm_reader as vvm
    >>>
    >>> # Load data and find centers
    >>> ds = vvm.open_vvm_dataset('/path/to/sim', variables=['u', 'v', 'zeta'])
    >>> ds_1km = ds.sel(lev=1000, method='nearest')
    >>> centers = tct.find_center_by_streamfunction(ds_1km)
    >>>
    >>> # Transform to cylindrical coordinates (all times)
    >>> ds_cyl = tct.transform_to_cylindrical(ds, centers)
    >>>
    >>> # r and azimuth vary with time as storm moves
    >>> print(ds_cyl.r)  # (time, lat, lon)
    >>> print(ds_cyl.azimuth)  # (time, lat, lon)
    >>>
    >>> # Can select specific time
    >>> ds_t0 = ds_cyl.isel(time=0)
    >>> print(ds_t0.r)  # (lat, lon) for time=0
    >>>
    >>> # Also works for single time step
    >>> centers_t0 = tct.find_center_by_streamfunction(ds_1km.isel(time=0))
    >>> ds_cyl_t0 = tct.transform_to_cylindrical(ds.isel(time=0), centers_t0)

    Notes
    -----
    - r and azimuth are data variables (not coordinates) because they vary with time
    - Uses periodic boundary conditions for distance calculation (VVM convention)
    - Fully vectorized: no loops over time steps for maximum performance
    - Works for both single and multiple time steps
    """
    # Validate inputs
    if 'xc' not in ds.coords or 'yc' not in ds.coords:
        raise ValueError("Dataset must contain 'xc' and 'yc' Cartesian coordinates.")

    required_vars = ['center_xc', 'center_yc', 'center_lon', 'center_lat']
    for var in required_vars:
        if var not in centers:
            raise ValueError(f"centers Dataset must contain '{var}' variable.")

    if 'time' not in ds.dims:
        raise ValueError("Dataset must have 'time' dimension.")

    if 'time' not in centers.dims:
        raise ValueError("centers Dataset must have 'time' dimension.")

    # Check time dimensions match
    if not np.array_equal(ds.time.values, centers.time.values):
        raise ValueError("Time coordinates of ds and centers must match.")

    # Get domain size
    domain_size_x, domain_size_y = _get_domain_size(ds)

    # Get coordinate arrays
    xc = ds.xc.values
    yc = ds.yc.values

    # Create 2D spatial grids (lat, lon)
    XC, YC = np.meshgrid(xc, yc)

    # Get center arrays (time,)
    center_xc = centers.center_xc.values
    center_yc = centers.center_yc.values

    n_times = len(center_xc)
    n_lat, n_lon = XC.shape

    # Preallocate output arrays (time, lat, lon)
    r_all = np.zeros((n_times, n_lat, n_lon))
    azimuth_all = np.zeros((n_times, n_lat, n_lon))

    # Vectorized computation over time
    # Expand center arrays to (time, lat, lon) for broadcasting
    center_xc_3d = center_xc[:, np.newaxis, np.newaxis]
    center_yc_3d = center_yc[:, np.newaxis, np.newaxis]

    # Expand spatial grids to (time, lat, lon)
    XC_3d = np.broadcast_to(XC[np.newaxis, :, :], (n_times, n_lat, n_lon))
    YC_3d = np.broadcast_to(YC[np.newaxis, :, :], (n_times, n_lat, n_lon))

    # Compute periodic displacements for all times at once
    dx_m = XC_3d - center_xc_3d
    dy_m = YC_3d - center_yc_3d

    # Apply periodic wrapping
    dx_m = dx_m - domain_size_x * np.round(dx_m / domain_size_x)
    dy_m = dy_m - domain_size_y * np.round(dy_m / domain_size_y)

    # Convert to km
    dx_km = dx_m / 1000.0
    dy_km = dy_m / 1000.0

    # Compute radius (time, lat, lon)
    r_all = np.sqrt(dx_km**2 + dy_km**2)

    # Compute azimuth (time, lat, lon)
    azimuth_all = (90 - np.degrees(np.arctan2(dy_km, dx_km))) % 360

    # Create output dataset (copy input)
    ds_out = ds.copy()

    # Add r and azimuth as data variables
    ds_out['r'] = (('time', 'lat', 'lon'), r_all)
    ds_out['azimuth'] = (('time', 'lat', 'lon'), azimuth_all)

    # Add metadata
    ds_out['r'].attrs = {
        'long_name': 'radius from storm center',
        'units': 'km',
        'domain_size_x': domain_size_x,
        'domain_size_y': domain_size_y,
        'note': 'Computed with periodic boundary conditions. Varies with time as storm moves.'
    }

    ds_out['azimuth'].attrs = {
        'long_name': 'azimuthal angle from north',
        'units': 'degrees',
        'convention': 'meteorological (0=North, clockwise positive)',
        'valid_range': (0, 360),
        'note': 'Varies with time as storm moves.'
    }

    # Store center information as attributes (for reference)
    ds_out.attrs.update({
        'cylindrical_coordinates': 'time-varying',
        'center_finding_method': centers.attrs.get('method', 'unknown')
    })

    return ds_out


__all__ = [
    'transform_to_cylindrical',
]
