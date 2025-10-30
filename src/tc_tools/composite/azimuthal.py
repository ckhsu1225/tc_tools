"""
Azimuthal averaging and asymmetry analysis

This module provides functions to compute azimuthal (angular) averages of
tropical cyclone fields, producing axisymmetric structure as a function of
radius and height.

The azimuthal mean is fundamental for:
- Analyzing mean TC structure (eyewall, rainbands)
- Computing intensity metrics (maximum tangential wind)
- Studying vertical structure
- Identifying asymmetric features by computing anomalies

Mathematical definition:
    field_axi(r, z) = (1/2π) ∫[0 to 2π] field(r, θ, z) dθ
"""

from typing import Optional, Union, Sequence
import numpy as np
import xarray as xr


def azimuthal_mean(
    ds: xr.Dataset,
    r_bins: Optional[Union[int, Sequence[float]]] = None,
    r_max: Optional[float] = None,
    variables: Optional[Sequence[str]] = None
) -> xr.Dataset:
    """
    Compute azimuthal mean to obtain axisymmetric structure.

    This function averages all variables in the dataset over the azimuthal
    direction, producing fields as a function of radius and other dimensions
    (time, height, etc.).

    Parameters
    ----------
    ds : xarray.Dataset
        Input dataset with cylindrical coordinates ('r', 'azimuth')
        Must be created using transform_to_cylindrical() first
    r_bins : int or array-like, optional
        Radial bins for averaging:
        - If int: number of equally-spaced bins from 0 to r_max
        - If array: explicit bin edges (km)
        - If None: uses all unique r values (no binning)
        Default: None
    r_max : float, optional
        Maximum radius to include (km). If None, uses maximum r in dataset.
        Default: None
    variables : list of str, optional
        List of variable names to average. If None, averages all data variables.
        Default: None

    Returns
    -------
    ds_axi : xarray.Dataset
        Axisymmetric dataset with 'rbin' dimension replacing spatial dimensions
        Variables have shape (time, lev, rbin) or subset depending on input

    Raises
    ------
    ValueError
        If 'r' coordinate not found (must transform to cylindrical first)

    Examples
    --------
    >>> import tc_tools as tct
    >>> import vvm_reader as vvm
    >>>
    >>> # Load data and transform to cylindrical coordinates
    >>> ds = vvm.open_vvm_dataset('/path/to/sim', variables=['u', 'v', 'zeta'])
    >>> ds_1km = ds.sel(lev=1000, method='nearest')
    >>> centers = tct.find_center_by_streamfunction(ds_1km)
    >>> ds_cyl = tct.transform_to_cylindrical(ds, centers)
    >>> ds_winds = tct.compute_tangential_radial_winds(ds_cyl)
    >>>
    >>> # Compute azimuthal mean with 50 radial bins out to 200 km
    >>> ds_axi = tct.azimuthal_mean(ds_winds, r_bins=50, r_max=200)
    >>>
    >>> # Result has dimensions (time, lev, rbin)
    >>> print(ds_axi.vt.shape)  # e.g., (72, 30, 50)
    >>>
    >>> # Can now easily find maximum tangential wind
    >>> vt = ds_axi.vt.max(dim='rbin')

    Notes
    -----
    - Dataset must have 'r' variable (use transform_to_cylindrical first)
    - Averaging is done by flattening the 2D spatial grid and grouping by radius
    - Radial binning reduces noise and creates regular radial grid
    - Output has 'rbin' dimension with bin center coordinates
    """
    # Validate inputs
    if 'r' not in ds:
        raise ValueError(
            "Dataset must have 'r' variable. "
            "Use transform_to_cylindrical() first."
        )

    # Determine which variables to average
    if variables is None:
        variables = [var for var in ds.data_vars if var not in ['r', 'azimuth']]

    # Get radius values
    r = ds['r']

    # Determine r_max
    if r_max is None:
        r_max = float(r.max())

    # Stack spatial dimensions into single 'point' dimension
    ds_stacked = ds.stack(point=('lat', 'lon') if 'lat' in ds.dims else ('yc', 'xc'), create_index=False)

    # Filter by r_max
    mask = ds_stacked['r'] <= r_max
    ds_filtered = ds_stacked.where(mask, drop=True)

    # Perform azimuthal averaging
    if r_bins is not None:
        if isinstance(r_bins, int):
            # Create equally-spaced bins
            r_edges = np.linspace(0, r_max, r_bins + 1)
        else:
            # Use provided bin edges
            r_edges = np.asarray(r_bins)

        # Use xarray's groupby_bins for cleaner binning
        ds_axi = ds_filtered[variables].groupby_bins(
            ds_filtered['r'],
            bins=r_edges,
            labels=(r_edges[:-1] + r_edges[1:]) / 2  # Use bin centers as labels
        ).mean(dim='point', engine='flox', fill_value=np.nan)

        # Rename the dimension created by groupby_bins
        ds_axi = ds_axi.rename({'r_bins': 'rbin'})

    else:
        # Average over all points at each unique radius
        # (This can be memory-intensive for large datasets)
        ds_axi = ds_filtered[variables].groupby(ds_filtered['r']).mean(dim='point')
        ds_axi = ds_axi.rename({'r': 'rbin'})

    # Add metadata
    ds_axi['rbin'].attrs = {
        'long_name': 'radial bin center',
        'units': 'km',
        'description': 'Distance from storm center'
    }

    # Copy attributes from original variables
    for var in variables:
        if var in ds:
            ds_axi[var].attrs = ds[var].attrs.copy()
            ds_axi[var].attrs['azimuthal_averaging'] = 'mean over all azimuths'

    return ds_axi


def azimuthal_anomaly(
    ds: xr.Dataset,
    ds_axi: xr.Dataset,
    variables: Optional[Sequence[str]] = None
) -> xr.Dataset:
    """
    Compute azimuthal anomalies (deviations from azimuthal mean).

    This function computes the asymmetric component of fields by subtracting
    the azimuthal mean from the full field.

    Parameters
    ----------
    ds : xarray.Dataset
        Full dataset with cylindrical coordinates
    ds_axi : xarray.Dataset
        Azimuthal mean dataset (from azimuthal_mean)
    variables : list of str, optional
        Variables to compute anomalies for. If None, uses all common variables.
        Default: None

    Returns
    -------
    ds_anom : xarray.Dataset
        Dataset with anomaly fields (field' = field - field_axi)

    Examples
    --------
    >>> # Compute azimuthal mean
    >>> ds_axi = tct.azimuthal_mean(ds_winds, r_bins=50, r_max=200)
    >>>
    >>> # Compute asymmetric component
    >>> ds_anom = tct.azimuthal_anomaly(ds_winds, ds_axi)
    >>>
    >>> # Now ds_anom contains deviations from axisymmetric structure
    >>> # Useful for studying asymmetries (wavenumber-1, environmental interaction)

    Notes
    -----
    - Both datasets must have compatible dimensions (time, lev)
    - The azimuthal mean is interpolated to the full grid before subtraction
    - Large asymmetries may indicate environmental influence or tilt
    """
    # Determine which variables to process
    if variables is None:
        variables = [v for v in ds.data_vars if v in ds_axi.data_vars]

    # Create output dataset
    ds_anom = ds[variables].copy()

    # Compute anomalies for each variable
    for var in variables:
        # Interpolate axisymmetric field back to full grid
        # Match r values from full dataset to rbin from azimuthal mean
        field_axi_interp = ds_axi[var].interp(rbin=ds['r'], method='nearest')

        # Compute anomaly
        ds_anom[var] = ds[var] - field_axi_interp

        # Update metadata
        ds_anom[var].attrs = ds[var].attrs.copy()
        ds_anom[var].attrs['long_name'] = ds[var].attrs.get('long_name', var) + ' anomaly'
        ds_anom[var].attrs['description'] = 'Deviation from azimuthal mean'

    return ds_anom


__all__ = [
    'azimuthal_mean',
    'azimuthal_anomaly',
]
