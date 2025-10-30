"""
Tropical cyclone size metrics

This module provides functions to compute size metrics based on wind fields.
Common size metrics include wind radii (R34, R50, R64) which represent the
radii where winds exceed certain thresholds.

Wind radius definitions (operational standards):
- R34: Radius of 34 kt (17.5 m/s) winds - outer size
- R50: Radius of 50 kt (25.7 m/s) winds - inner core size
- R64: Radius of 64 kt (32.9 m/s) winds - hurricane-force wind radius

For idealized simulations, custom thresholds can be used based on
typical wind speeds in the simulation.
"""

from typing import Optional, Sequence
import xarray as xr


def compute_size_metrics(
    ds_axi: xr.Dataset,
    wind_var: Optional[str] = None,
    thresholds: Optional[Sequence[float]] = None
) -> xr.Dataset:
    """
    Compute tropical cyclone size metrics based on wind radii.

    This function computes wind radii for all time steps and vertical levels
    in the input dataset. Wind radii represent the outermost radius where
    wind speed exceeds specified thresholds.

    Parameters
    ----------
    ds_axi : xarray.Dataset
        Axisymmetric dataset (from azimuthal_mean) with dimensions (time, lev, rbin)
        Must contain wind speed variable
    wind_var : str, optional
        Wind variable name to use. If None, automatically searches for:
        1. 'ws' (total wind speed) - preferred
        2. 'vt' (tangential wind) - fallback
        Raises error if neither found.
        Default: None
    thresholds : list of float, optional
        Wind speed thresholds (kt) for computing radii.
        Default: [34, 50, 64] (corresponding to 17.5, 25.7, 32.9 m/s)

    Returns
    -------
    size_metrics : xarray.Dataset
        Dataset with dimensions (time, lev) containing wind radii variables:
        - 'r34': Radius of 34 kt winds (km)
        - 'r50': Radius of 50 kt winds (km)
        - 'r64': Radius of 64 kt winds (km)
        Or custom names based on thresholds. Values are NaN if threshold not reached.

    Raises
    ------
    ValueError
        If wind variable not found in dataset or required dimensions missing

    Examples
    --------
    >>> import tc_tools as tct
    >>>
    >>> # After getting axisymmetric data with dimensions (time, lev, rbin)
    >>> ds_axi = tct.azimuthal_mean(ds_winds, r_bins=50, r_max=200)
    >>>
    >>> # Compute size metrics for all times and levels
    >>> size = tct.compute_size_metrics(ds_axi)
    >>> print(size)
    >>> # <xarray.Dataset>
    >>> # Dimensions:  (time: 72, lev: 30)
    >>> # Data variables:
    >>> #     r34      (time, lev) float64 ...
    >>> #     r50      (time, lev) float64 ...
    >>> #     r64      (time, lev) float64 ...
    >>>
    >>> # Get R34 at 1 km height for all times
    >>> r34_1km = size.r34.sel(lev=1000, method='nearest')
    >>>
    >>> # Use custom thresholds
    >>> size = tct.compute_size_metrics(
    ...     ds_axi,
    ...     thresholds=[20, 50, 100]  # kt
    ... )
    >>> print(size.data_vars)  # Will have r20, r50, r100

    Notes
    -----
    - Wind radii are computed as maximum radius where wind exceeds threshold
    - If wind never reaches threshold, returns NaN for that time/level
    - Standard thresholds: 34, 50, 64 kt → 17.5, 25.7, 32.9 m/s
    - Total wind speed (ws) is preferred as it represents full 3D wind
    """
    # Validate required dimension
    if 'rbin' not in ds_axi.dims:
        raise ValueError(
            "Dataset must have 'rbin' dimension. "
            "Use azimuthal_mean() first to create axisymmetric dataset."
        )

    # Determine wind variable to use
    if wind_var is None:
        if 'ws' in ds_axi:
            wind_var = 'ws'
        elif 'vt' in ds_axi:
            wind_var = 'vt'
        else:
            raise ValueError(
                "No suitable wind variable found. "
                "Dataset must contain 'ws' or 'vt'. "
                "Available variables: " + ", ".join(ds_axi.data_vars)
            )
    else:
        if wind_var not in ds_axi:
            raise ValueError(
                f"Variable '{wind_var}' not found in dataset. "
                f"Available variables: {', '.join(ds_axi.data_vars)}"
            )

    # Default thresholds (34, 50, 64 kt in m/s)
    if thresholds is None:
        thresholds = [34, 50, 64]
    threshold_names = [f'r{int(t)}' for t in thresholds]

    # Get wind field
    wind_field = ds_axi[wind_var]

    # Compute wind radius for each threshold
    size_data = {}
    for threshold, name in zip(thresholds, threshold_names):
        radius = compute_wind_radius(wind_field, ds_axi['rbin'], threshold*0.51444)
        size_data[name] = radius

    # Create output dataset
    ds_out = xr.Dataset(size_data)

    # Add metadata for each variable
    for threshold, name in zip(thresholds, threshold_names):
        ds_out[name].attrs = {
            'long_name': f'radius of {threshold} kt winds',
            'units': 'km',
            'description': f'Outermost radius where {wind_var} >= {threshold} kt',
            'threshold': threshold,
            'source_variable': wind_var
        }

    return ds_out


def compute_wind_radius(
    wind_field: xr.DataArray,
    r_bin: xr.DataArray,
    threshold: float
) -> xr.DataArray:
    """
    Compute radius where wind exceeds threshold.

    This function finds the outermost radius where the wind field
    exceeds the specified threshold, working on all time and level dimensions.

    Parameters
    ----------
    wind_field : xarray.DataArray
        Wind field with dimensions (time, lev, rbin) or subset
    r_bin : xarray.DataArray
        Radial coordinate (km)
    threshold : float
        Wind speed threshold (m/s)

    Returns
    -------
    radius : xarray.DataArray
        Maximum radius where wind >= threshold (km), with dimensions (time, lev)
        or subset. NaN where threshold is never reached.

    Examples
    --------
    >>> # Find radius of 20 m/s winds for all times and levels
    >>> r20 = compute_wind_radius(ds_axi.ws, ds_axi.rbin, 20.0)
    >>> print(r20.dims)  # (time, lev)
    >>>
    >>> # Get time series at specific level
    >>> r20_1km = r20.sel(lev=1000, method='nearest')

    Notes
    -----
    - Returns outermost radius (maximum r where wind >= threshold)
    - Returns NaN if wind never reaches threshold at that time/level
    - Uses xarray's where() for efficient vectorized computation
    """
    # Find where wind exceeds threshold
    mask = wind_field >= threshold

    # Get radius values where condition is met, NaN elsewhere
    r_exceeds = r_bin.where(mask)

    # Find maximum radius along radial dimension
    # This gives the outermost radius where wind >= threshold
    radius = r_exceeds.max(dim='rbin')

    return radius


__all__ = [
    'compute_size_metrics',
    'compute_wind_radius',
]
