"""
Wind-based intensity metrics

This module provides functions to compute tropical cyclone intensity metrics
from wind fields, including maximum wind speed and radius of maximum wind (RMW).

Since VVM simulations use height coordinates without pressure output, all
intensity metrics are based on wind speed at specific height levels.

Common intensity levels:
- Surface (10 m): Operational intensity standard
- 1 km height: Common choice for idealized simulations
- Flight level (~700 hPa): Aircraft reconnaissance level
"""

from typing import Optional
import xarray as xr


def compute_intensity(
    ds_axi: xr.Dataset,
    wind_var: Optional[str] = None
) -> xr.Dataset:
    """
    Compute tropical cyclone intensity metrics from axisymmetric wind field.

    This function computes maximum wind speed (vmax) and radius of maximum wind
    (RMW) for all time steps and vertical levels in the input dataset.

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

    Returns
    -------
    intensity : xarray.Dataset
        Dataset with dimensions (time, lev) containing:
        - 'vmax': Maximum wind speed (m/s)
        - 'rmw': Radius of maximum wind (km)

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
    >>> # Compute intensity for all times and levels
    >>> intensity = tct.compute_intensity(ds_axi)
    >>> print(intensity)
    >>> # <xarray.Dataset>
    >>> # Dimensions:  (time: 72, lev: 30)
    >>> # Data variables:
    >>> #     vmax     (time, lev) float64 ...
    >>> #     rmw      (time, lev) float64 ...
    >>>
    >>> # Get intensity at 1 km height for all times
    >>> intensity_1km = intensity.sel(lev=1000, method='nearest')
    >>> print(f"Vmax at 1 km: {intensity_1km.vmax.values} m/s")
    >>>
    >>> # Get time series of surface vmax
    >>> vmax_surface = intensity.isel(lev=0).vmax
    >>>
    >>> # Plot vertical profile at specific time
    >>> import matplotlib.pyplot as plt
    >>> intensity.isel(time=50).vmax.plot(y='lev')
    >>> plt.xlabel('Vmax (m/s)')
    >>> plt.ylabel('Height (m)')

    Notes
    -----
    - Input should be azimuthally-averaged (axisymmetric) data
    - Total wind speed (ws) is preferred as it represents full 3D wind intensity
    - Tangential wind (vt) can be used for purely rotational intensity
    - RMW is a key structural metric related to storm maturity
    - For datasets without time or lev dimensions, function still works on available dimensions
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

    # Get wind field
    wind_field = ds_axi[wind_var].dropna(dim='lev', how='all')

    # Find maximum wind and its location along radial dimension
    vmax = wind_field.max(dim='rbin').compute()
    vmax_idx = wind_field.argmax(dim='rbin').compute()

    # Get radius at maximum wind
    rmw = ds_axi['rbin'].isel(rbin=vmax_idx)

    # Create output dataset
    ds_out = xr.Dataset({
        'vmax': vmax,
        'rmw': rmw
    })

    # Add metadata
    ds_out['vmax'].attrs = {
        'long_name': 'maximum wind speed',
        'units': wind_field.attrs.get('units', 'm/s'),
        'description': f'Maximum {wind_var} at each level and time',
        'source_variable': wind_var
    }

    ds_out['rmw'].attrs = {
        'long_name': 'radius of maximum wind',
        'units': 'km',
        'description': f'Radius of maximum {wind_var} at each level and time',
        'source_variable': wind_var
    }

    return ds_out


__all__ = [
    'compute_intensity',
]
