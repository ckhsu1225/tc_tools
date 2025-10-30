"""
Wind decomposition into tangential and radial components

This module provides functions to decompose Cartesian wind components (u, v)
into cylindrical components (tangential, radial) relative to the storm center.

Wind component definitions:
- vt (tangential): Wind component perpendicular to radius, positive = cyclonic
  * NH: Positive = counterclockwise circulation
  * SH: Positive = clockwise circulation
- vr (radial): Wind component along radius, positive = outward from center

For tropical cyclones:
- vt > 0: Cyclonic circulation
- vr < 0: Inflow toward center (boundary layer)
- vr > 0: Outflow from center (upper levels)
"""

from typing import Tuple
import numpy as np
import numpy.typing as npt
import xarray as xr


def compute_tangential_radial_winds(
    ds: xr.Dataset,
    u_var: str = 'u',
    v_var: str = 'v'
) -> xr.Dataset:
    """
    Compute tangential and radial wind components from Cartesian winds.

    This function requires the dataset to have 'r' and 'azimuth' data variables,
    which should be added first using transform_to_cylindrical().

    Parameters
    ----------
    ds : xarray.Dataset
        Input dataset with 'r' and 'azimuth' data variables and
        Cartesian wind components (u, v)
    u_var : str, optional
        Name of zonal wind variable. Default: 'u'
    v_var : str, optional
        Name of meridional wind variable. Default: 'v'

    Returns
    -------
    ds_out : xarray.Dataset
        Dataset with added 'vt' and 'vr' variables
        - vt: Tangential wind (positive = cyclonic)
        - vr: Radial wind (positive = outward)

    Raises
    ------
    ValueError
        If 'r' or 'azimuth' data variables not found (must transform first)
        If u_var or v_var not found in dataset

    Examples
    --------
    >>> import tc_tools as tct
    >>> import vvm_reader as vvm
    >>>
    >>> # Load data
    >>> ds = vvm.open_vvm_dataset('/path/to/sim', variables=['u', 'v', 'zeta'])
    >>>
    >>> # Find centers and transform to cylindrical coordinates
    >>> ds_1km = ds.sel(lev=1000, method='nearest')
    >>> centers = tct.find_center_by_streamfunction(ds_1km)
    >>> ds_cyl = tct.transform_to_cylindrical(ds, centers)
    >>>
    >>> # Compute tangential and radial winds
    >>> ds_winds = tct.compute_tangential_radial_winds(ds_cyl)
    >>>
    >>> # Now ds_winds has 'vt' and 'vr' with same dimensions as r/azimuth
    >>> print(ds_winds['vt'])  # (time, lev, lat, lon)
    >>> print(ds_winds['vr'])  # (time, lev, lat, lon)

    Notes
    -----
    - Dataset must already have 'r' and 'azimuth' data variables
    - Works with time-varying cylindrical coordinates (r and azimuth with time dimension)
    - Tangential wind convention: positive = cyclonic (counterclockwise in NH)
    - Radial wind convention: positive = outward from center
    - The transformation is:
        vt = -u * cos(θ) + v * sin(θ)
        vr =  u * sin(θ) + v * cos(θ)
      where θ is azimuth measured clockwise from north
    """
    # Validate inputs
    if 'r' not in ds or 'azimuth' not in ds:
        raise ValueError(
            "Dataset must have 'r' and 'azimuth' data variables. "
            "Use transform_to_cylindrical() first."
        )

    if u_var not in ds:
        raise ValueError(f"Variable '{u_var}' not found in dataset")
    if v_var not in ds:
        raise ValueError(f"Variable '{v_var}' not found in dataset")

    # Get wind components
    u = ds[u_var]
    v = ds[v_var]

    # Get azimuth angle (in degrees)
    azimuth = ds['azimuth']

    # Convert azimuth to radians
    azimuth_rad = np.deg2rad(azimuth)

    # Compute tangential and radial components
    # Meteorological azimuth: 0° = North, clockwise positive
    # Standard math angle: 0° = East, counterclockwise positive
    # Conversion: math_angle = 90° - meteor_azimuth
    #
    # For vector transformation:
    # vt = -u * cos(θ) + v * sin(θ)  (positive = counterclockwise)
    # vr =  u * sin(θ) + v * cos(θ)  (positive = outward)
    # where θ is meteorological azimuth

    vt = -u * np.cos(azimuth_rad) + v * np.sin(azimuth_rad)
    vr = u * np.sin(azimuth_rad) + v * np.cos(azimuth_rad)

    # Create output dataset
    ds_out = ds.copy()
    ds_out['vt'] = vt
    ds_out['vr'] = vr

    # Add metadata
    ds_out['vt'].attrs = {
        'long_name': 'tangential wind',
        'units': u.attrs.get('units', 'm/s'),
        'description': 'Wind component perpendicular to radius (positive = cyclonic)',
        'convention': 'positive = counterclockwise in NH, clockwise in SH'
    }
    ds_out['vr'].attrs = {
        'long_name': 'radial wind',
        'units': u.attrs.get('units', 'm/s'),
        'description': 'Wind component along radius (positive = outward from center)',
        'convention': 'positive = outward, negative = inward'
    }

    return ds_out


def decompose_wind_vector(
    u: npt.NDArray[np.floating],
    v: npt.NDArray[np.floating],
    azimuth: npt.NDArray[np.floating]
) -> Tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """
    Decompose wind vector into tangential and radial components.

    This is a lower-level function that operates on numpy arrays.
    For xarray datasets, use compute_tangential_radial_winds() instead.

    Parameters
    ----------
    u : numpy.ndarray
        Zonal wind component (m/s)
    v : numpy.ndarray
        Meridional wind component (m/s)
    azimuth : numpy.ndarray
        Azimuthal angle from north, clockwise (degrees)

    Returns
    -------
    vt : numpy.ndarray
        Tangential wind component (positive = cyclonic)
    vr : numpy.ndarray
        Radial wind component (positive = outward)

    Examples
    --------
    >>> import numpy as np
    >>> from tc_tools.transform import decompose_wind_vector
    >>>
    >>> # Example: Pure eastward wind at north of center
    >>> u = np.array([10.0])  # 10 m/s eastward
    >>> v = np.array([0.0])   # no meridional component
    >>> azimuth = np.array([0.0])  # north of center
    >>>
    >>> vt, vr = decompose_wind_vector(u, v, azimuth)
    >>> print(f"Tangential: {vt[0]:.1f} m/s")  # Should be -10 (anticyclonic)
    >>> print(f"Radial: {vr[0]:.1f} m/s")      # Should be 0 (outward)
    """
    # Convert azimuth to radians
    azimuth_rad = np.deg2rad(azimuth)

    # Compute components
    vt = -u * np.cos(azimuth_rad) + v * np.sin(azimuth_rad)
    vr = u * np.sin(azimuth_rad) + v * np.cos(azimuth_rad)

    return vt, vr


__all__ = [
    'compute_tangential_radial_winds',
    'decompose_wind_vector',
]
