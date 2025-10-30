"""
Vorticity-based typhoon center finding

This module provides functions to locate typhoon centers using the maximum
vorticity method. This is a simple and fast approach but can be sensitive
to small-scale noise in the vorticity field.

For weak vortices in idealized simulations, the streamfunction method is
generally more robust. Use this method when:
- The vortex is well-defined with strong vorticity
- You need fast computation
- You want a simple, interpretable method
"""

import numpy as np
import xarray as xr
from typing import Literal

try:
    from astropy.convolution import convolve_fft, Gaussian2DKernel
    HAS_ASTROPY = True
except ImportError:
    HAS_ASTROPY = False
    raise ImportError(
        "astropy is required for vorticity-based center finding. "
        "VVM simulations include terrain (NaN values) which require NaN-aware convolution. "
        "Please install: pip install astropy"
    )


def smooth_vorticity(
    vort: xr.DataArray,
    smooth_sigma: float = 5,
    parallel_method: Literal['auto', 'joblib', 'serial'] = 'auto',
    n_jobs: int = -1
) -> xr.DataArray:
    """
    Smooth vorticity field using FFT-based Gaussian convolution.

    This function handles multi-dimensional vorticity data (with time dimension)
    and applies NaN-aware smoothing suitable for VVM terrain.

    Parameters
    ----------
    vort : xarray.DataArray
        Vorticity field to smooth. Should have dimensions (time, lat, lon).
        For single time step, use vort.isel(time=0) to preserve dimension.
    smooth_sigma : float, optional
        Gaussian smoothing parameter (standard deviation in grid points).
        Larger values = more smoothing. Set to 0 for no smoothing.
        Default: 5
    parallel_method : {'auto', 'joblib', 'serial'}, optional
        Parallelization strategy:
        - 'auto': Automatically choose based on data size
                  (serial for <=10 time steps, joblib for >10 steps)
        - 'joblib': Use joblib for parallel processing (fastest for most cases)
        - 'serial': Simple for loop (good for small datasets or debugging)
        Default: 'auto'
    n_jobs : int, optional
        Number of parallel jobs for joblib method. -1 uses all CPU cores.
        Ignored when parallel_method='serial'. Default: -1

    Returns
    -------
    smoothed : xarray.DataArray
        Smoothed vorticity field with same dimensions as input

    Notes
    -----
    - Uses FFT-based convolution for efficiency with large kernels
    - Assumes periodic boundary conditions (consistent with VVM model)
    - NaN-aware: preserves NaN locations (terrain) after smoothing
    - For most datasets, 'joblib' method is fastest (often 2-4x speedup)
    - Serial method is useful for small datasets where parallel overhead dominates

    Examples
    --------
    >>> import tc_tools as tct
    >>> import vvm_reader as vvm
    >>>
    >>> # Load vorticity data
    >>> ds = vvm.open_vvm_dataset('/path/to/sim', variables=['zeta'])
    >>> zeta = ds['zeta'].sel(lev=1000, method='nearest')
    >>>
    >>> # Smooth with auto parallelization
    >>> zeta_smooth = tct.smooth_vorticity(zeta, smooth_sigma=5)
    >>>
    >>> # Force serial processing
    >>> zeta_smooth = tct.smooth_vorticity(zeta, smooth_sigma=10, parallel_method='serial')
    >>>
    >>> # Use joblib with 4 cores
    >>> zeta_smooth = tct.smooth_vorticity(zeta, smooth_sigma=5,
    ...                                      parallel_method='joblib', n_jobs=4)
    """
    if smooth_sigma <= 0:
        return vort

    # Validate parallel_method
    valid_methods = {'auto', 'joblib', 'serial'}
    if parallel_method not in valid_methods:
        raise ValueError(f"parallel_method must be one of {valid_methods}, got {parallel_method!r}")

    def _smooth_2d(vort_data: np.ndarray) -> np.ndarray:
        """Smooth single 2D vorticity slice"""
        kernel = Gaussian2DKernel(x_stddev=smooth_sigma)
        return convolve_fft(
            vort_data,
            kernel=kernel,
            boundary='wrap',  # VVM uses periodic boundaries
            nan_treatment='interpolate',  # Interpolate over NaN (terrain)
            preserve_nan=True,  # Keep original NaN locations
            allow_huge=True,
            normalize_kernel=True
        )

    # Ensure time dimension exists
    if 'time' not in vort.dims:
        raise ValueError("Input must have 'time' dimension. "
                        "For single time step, use vort.isel(time=[0]) to preserve dimension.")

    n_times = vort.sizes['time']

    # Compute if dask array (joblib works better with in-memory data)
    vort_computed = vort.compute().values if hasattr(vort.data, 'compute') else vort.values

    # Auto-select method based on data size
    if parallel_method == 'auto':
        parallel_method = 'serial' if n_times <= 10 else 'joblib'

    # Serial processing
    if parallel_method == 'serial':
        results = [
            _smooth_2d(vort_computed[t])
            for t in range(n_times)
        ]
        smoothed_data = np.stack(results, axis=0)

    # Joblib parallel processing
    elif parallel_method == 'joblib':
        from joblib import Parallel, delayed
        import os

        # Disable BLAS/FFT threading to avoid over-subscription
        # When joblib creates multiple processes, each process should use single-threaded FFT
        # to maximize overall throughput and avoid CPU contention
        old_omp = os.environ.get('OMP_NUM_THREADS')
        old_mkl = os.environ.get('MKL_NUM_THREADS')
        old_openblas = os.environ.get('OPENBLAS_NUM_THREADS')

        try:
            os.environ['OMP_NUM_THREADS'] = '1'
            os.environ['MKL_NUM_THREADS'] = '1'
            os.environ['OPENBLAS_NUM_THREADS'] = '1'

            results = Parallel(n_jobs=n_jobs)(
                delayed(_smooth_2d)(vort_computed[t])
                for t in range(n_times)
            )
            smoothed_data = np.stack(results, axis=0)

        finally:
            # Restore original threading settings
            if old_omp is not None:
                os.environ['OMP_NUM_THREADS'] = old_omp
            elif 'OMP_NUM_THREADS' in os.environ:
                del os.environ['OMP_NUM_THREADS']

            if old_mkl is not None:
                os.environ['MKL_NUM_THREADS'] = old_mkl
            elif 'MKL_NUM_THREADS' in os.environ:
                del os.environ['MKL_NUM_THREADS']

            if old_openblas is not None:
                os.environ['OPENBLAS_NUM_THREADS'] = old_openblas
            elif 'OPENBLAS_NUM_THREADS' in os.environ:
                del os.environ['OPENBLAS_NUM_THREADS']

    # Reconstruct xarray with original coordinates
    smoothed = xr.DataArray(
        smoothed_data,
        coords=vort.coords,
        dims=vort.dims,
        attrs=vort.attrs
    )
    smoothed.attrs['smoothing_sigma'] = smooth_sigma
    smoothed.attrs['smoothing_method'] = parallel_method

    return smoothed


def find_center_by_vorticity_maximum(
    vort: xr.DataArray,
    smooth_sigma: float | None = 5,
) -> xr.Dataset:
    """
    Find typhoon centers by maximum vorticity method.

    The center is identified as the location of maximum vertical vorticity.
    Gaussian smoothing reduces sensitivity to small-scale noise.

    Parameters
    ----------
    vort : xr.DataArray
        Vorticity field with dimensions (time, lat, lon).
        User should select vertical level before calling this function.
        Coordinates 'xc', 'yc', 'lon', 'lat' must be present.
        Can be raw or pre-smoothed vorticity.
    smooth_sigma : float or None, optional
        Gaussian smoothing parameter for vorticity field (in grid points).
        - If float: Apply smoothing with this sigma value
        - If None: Skip smoothing (use input as-is)
        Larger values = more smoothing. Default: 5
        Set to None if you have already smoothed the data.

    Returns
    -------
    centers : xr.Dataset
        Dataset with time dimension containing:
        - 'center_xc' (time,) : Center x-coordinates in meters
        - 'center_yc' (time,) : Center y-coordinates in meters
        - 'center_lon' (time,) : Center longitudes
        - 'center_lat' (time,) : Center latitudes
        - 'center_vorticity' (time,) : Maximum vorticity values at centers

    Raises
    ------
    ValueError
        If required coordinates not found in DataArray

    Examples
    --------
    >>> import tc_tools as tct
    >>> import vvm_reader as vvm
    >>>
    >>> # Load VVM data
    >>> ds = vvm.open_vvm_dataset('/path/to/sim', variables=['zeta'])
    >>> vort = ds['zeta'].sel(lev=1000, method='nearest')
    >>>
    >>> # Find centers with auto smoothing (recommended)
    >>> centers = tct.find_center_by_vorticity_maximum(vort)
    >>> print(centers.center_lon.values)  # Array of longitudes
    >>>
    >>> # Use larger smoothing for noisy data
    >>> centers = tct.find_center_by_vorticity_maximum(vort, smooth_sigma=20)
    >>>
    >>> # Skip smoothing if already smoothed
    >>> vort_smooth = tct.smooth_vorticity(vort, smooth_sigma=10)
    >>> centers = tct.find_center_by_vorticity_maximum(vort_smooth, smooth_sigma=None)
    >>>
    >>> # Use vertical average
    >>> vort_avg = ds['zeta'].sel(lev=slice(500, 2000)).mean('lev')
    >>> centers = tct.find_center_by_vorticity_maximum(vort_avg)

    Notes
    -----
    - Uses FFT-based convolution for efficient smoothing with large kernels
    - Assumes periodic boundary conditions (consistent with VVM model setup)
    - Smoothing is highly recommended to reduce sensitivity to grid-scale noise
    - For weak or poorly-defined vortices, consider using streamfunction method
    - The method assumes cyclonic vorticity is positive (Northern Hemisphere convention)
    - NaN-aware: properly handles terrain in VVM simulations
    - Requires astropy: pip install astropy
    - Processes each time step independently (no temporal smoothing)
    - Designed for single-typhoon VVM simulations (searches entire domain)
    """
    # Validate input
    if 'xc' not in vort.coords or 'yc' not in vort.coords:
        raise ValueError("DataArray must contain 'xc' and 'yc' Cartesian coordinates (in meters). "
                        "VVM uses Cartesian grid; lat/lon are derived coordinates.")

    # Ensure time dimension exists (even if single time step)
    if 'time' not in vort.dims:
        raise ValueError("DataArray must have 'time' dimension. "
                        "For single time step, use vort.isel(time=[0]) to preserve dimension.")

    # Apply smoothing if requested
    if smooth_sigma is not None:
        vort_smoothed = smooth_vorticity(vort, smooth_sigma)
    else:
        vort_smoothed = vort

    # Vectorized center finding for all time steps
    n_times = vort_smoothed.sizes['time']
    xc = vort_smoothed.xc.values
    yc = vort_smoothed.yc.values
    lon = vort_smoothed.lon.values
    lat = vort_smoothed.lat.values

    # Get all vorticity data at once
    vort_data = vort_smoothed.values  # (time, lat, lon)

    # Reshape to (time, lat*lon) for vectorized argmax
    ny, nx = vort_data.shape[1], vort_data.shape[2]
    vort_flat = vort_data.reshape(n_times, -1)

    # Find maximum indices for all time steps at once
    max_flat_indices = np.nanargmax(vort_flat, axis=1)  # (time,)

    # Convert flat indices back to 2D indices
    i_indices = max_flat_indices // nx  # row indices
    j_indices = max_flat_indices % nx   # col indices

    # Get vorticity values at maximum locations
    center_vorts = vort_data[np.arange(n_times), i_indices, j_indices]

    # Get coordinates at maximum locations
    center_xcs = xc[j_indices]
    center_ycs = yc[i_indices]
    center_lons = lon[j_indices]
    center_lats = lat[i_indices]

    # Handle all-NaN time steps
    all_nan_mask = np.all(np.isnan(vort_flat), axis=1)
    center_xcs[all_nan_mask] = np.nan
    center_ycs[all_nan_mask] = np.nan
    center_lons[all_nan_mask] = np.nan
    center_lats[all_nan_mask] = np.nan
    center_vorts[all_nan_mask] = np.nan

    # Return as xarray Dataset
    center_ds = xr.Dataset(
        {
            'center_xc': (['time'], center_xcs),
            'center_yc': (['time'], center_ycs),
            'center_lon': (['time'], center_lons),
            'center_lat': (['time'], center_lats),
            'center_vorticity': (['time'], center_vorts),
        },
        coords={'time': vort_smoothed.time.values},
        attrs={
            'method': 'vorticity_maximum',
            'smooth_sigma': smooth_sigma,
            'description': 'Typhoon center locations found by maximum vorticity method'
        }
    )
    return center_ds


def find_center_by_vorticity_centroid(
    vort: xr.DataArray,
    smooth_sigma: float | None = 5,
    search_radius: float = 100_000,
    threshold_percentile: float = 90,
    parallel_center: str | bool = 'auto',
    n_jobs: int = -1,
) -> xr.Dataset:
    """
    Find typhoon centers using local vorticity-weighted centroid method.

    This method addresses the issue of global centroid being influenced by
    distant signals. It first finds the vorticity maximum, then computes
    the centroid only within a local region around that maximum. This is
    more robust for cases with multiple vorticity centers or asymmetric patterns.

    The method properly handles periodic boundaries (VVM's doubly periodic domain)
    by shifting the domain so the search region is continuous.

    Parameters
    ----------
    vort : xr.DataArray
        Vorticity field with dimensions (time, lat, lon).
        User should select vertical level before calling this function.
        Coordinates 'xc', 'yc', 'lon', 'lat' must be present.
        Can be raw or pre-smoothed vorticity.
    smooth_sigma : float or None, optional
        Gaussian smoothing parameter for vorticity field (in grid points).
        - If float: Apply smoothing with this sigma value
        - If None: Skip smoothing (use input as-is)
        Larger values = more smoothing. Default: 5
        Set to None if you have already smoothed the data.
    search_radius : float, optional
        Radius (in meters) around vorticity maximum to search for centroid.
        Only vorticity within this radius contributes to centroid calculation.
        Default: 100,000 (100 km)
    threshold_percentile : float, optional
        Percentile threshold for vorticity values within the search region.
        Only vorticity values above this percentile are weighted in the centroid.
        Default: 90 (use top 10% of vorticity values within search region)
        Range: 0-100
    parallel_center : {'auto', True, False}, optional
        Whether to use parallel processing for center finding (across time steps).
        - 'auto': Automatically decide based on grid size and number of time steps
                  (parallel for grids >500×500, otherwise serial)
        - True: Force parallel processing
        - False: Force serial processing
        Default: 'auto' (recommended - optimal for most cases)
    n_jobs : int, optional
        Number of parallel jobs when parallel_center=True. -1 uses all CPU cores.
        Default: -1

    Returns
    -------
    centers : xr.Dataset
        Dataset with time dimension containing:
        - 'center_xc' (time,) : Center x-coordinates in meters (weighted centroid)
        - 'center_yc' (time,) : Center y-coordinates in meters (weighted centroid)
        - 'center_lon' (time,) : Center longitudes
        - 'center_lat' (time,) : Center latitudes
        - 'center_vorticity' (time,) : Vorticity values at centroids
        - 'center_max_vorticity' (time,) : Maximum vorticity values
        - 'max_xc' (time,) : X-coordinates of vorticity maximum
        - 'max_yc' (time,) : Y-coordinates of vorticity maximum

    Raises
    ------
    ValueError
        If required coordinates not found in DataArray or invalid parameters

    Examples
    --------
    >>> import tc_tools as tct
    >>> import vvm_reader as vvm
    >>>
    >>> # Load VVM data
    >>> ds = vvm.open_vvm_dataset('/path/to/sim', variables=['zeta'])
    >>> vort = ds['zeta'].sel(lev=1000, method='nearest')
    >>>
    >>> # Find centers using local centroid method (recommended)
    >>> centers = tct.find_center_by_vorticity_centroid(vort)
    >>>
    >>> # Use larger search radius (200 km)
    >>> centers = tct.find_center_by_vorticity_centroid(
    ...     vort,
    ...     search_radius=200_000,
    ...     threshold_percentile=95
    ... )
    >>>
    >>> # Skip smoothing if already smoothed
    >>> vort_smooth = tct.smooth_vorticity(vort, smooth_sigma=10)
    >>> centers = tct.find_center_by_vorticity_centroid(
    ...     vort_smooth,
    ...     smooth_sigma=None,
    ...     search_radius=150_000
    ... )

    Notes
    -----
    - More robust than global centroid when there are distant signals
    - Properly handles VVM's doubly periodic boundary conditions
    - The search_radius should be large enough to capture the vortex core
    - Smaller search_radius focuses more on the immediate core
    - This method is recommended for asymmetric or noisy vorticity fields
    - center_xc/yc represent the weighted centroid position
    - max_xc/yc represent the absolute vorticity maximum position
    - The distance between center and max can indicate asymmetry
    """
    # Validate input
    if 'xc' not in vort.coords or 'yc' not in vort.coords:
        raise ValueError("DataArray must contain 'xc' and 'yc' Cartesian coordinates (in meters). "
                        "VVM uses Cartesian grid; lat/lon are derived coordinates.")

    if not 0 <= threshold_percentile <= 100:
        raise ValueError(f"threshold_percentile must be between 0 and 100, got {threshold_percentile}")

    if search_radius <= 0:
        raise ValueError(f"search_radius must be positive, got {search_radius}")

    # Ensure time dimension exists
    if 'time' not in vort.dims:
        raise ValueError("DataArray must have 'time' dimension. "
                        "For single time step, use vort.isel(time=[0]) to preserve dimension.")

    # Apply smoothing if requested
    if smooth_sigma is not None:
        vort_smoothed = smooth_vorticity(vort, smooth_sigma)
    else:
        vort_smoothed = vort

    # Get coordinates
    xc = vort_smoothed.xc.values
    yc = vort_smoothed.yc.values
    lon = vort_smoothed.lon.values
    lat = vort_smoothed.lat.values
    dx = float(xc[1] - xc[0])
    dy = float(yc[1] - yc[0])

    # Create coordinate grids
    xc_grid, yc_grid = np.meshgrid(xc, yc)

    def _compute_local_centroid_2d(vort_data, xc, yc, lon, lat, search_radius, threshold_pct):
        """Compute local centroid around vorticity maximum for 2D field"""
        if np.isnan(vort_data).all():
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

        # Find global maximum location
        max_idx = np.unravel_index(np.nanargmax(vort_data), vort_data.shape)
        i_max, j_max = max_idx
        max_xc = xc[j_max]
        max_yc = yc[i_max]
        max_vort = vort_data[i_max, j_max]

        # Shift coordinates to handle periodic boundaries
        # Make maximum location the "center" by using modulo arithmetic
        dx_grid = xc_grid - max_xc
        dy_grid = yc_grid - max_yc

        # Handle periodicity: shift coordinates to [-L/2, L/2] range
        Lx = xc[-1] - xc[0] + dx
        Ly = yc[-1] - yc[0] + dy
        dx_grid = np.where(dx_grid > Lx/2, dx_grid - Lx, dx_grid)
        dx_grid = np.where(dx_grid < -Lx/2, dx_grid + Lx, dx_grid)
        dy_grid = np.where(dy_grid > Ly/2, dy_grid - Ly, dy_grid)
        dy_grid = np.where(dy_grid < -Ly/2, dy_grid + Ly, dy_grid)

        # Compute distance from maximum
        dist = np.sqrt(dx_grid**2 + dy_grid**2)

        # Create local mask: within search_radius and above threshold
        local_mask = dist <= search_radius
        if not local_mask.any():
            # Fallback if search radius too small
            return max_xc, max_yc, lon[j_max], lat[i_max], max_vort, max_vort, max_xc, max_yc

        # Get vorticity values in local region
        vort_local = np.where(local_mask, vort_data, np.nan)
        threshold_value = np.nanpercentile(vort_local, threshold_pct)

        # Final mask: local region AND above threshold
        mask = (vort_local >= threshold_value) & ~np.isnan(vort_local)

        if not mask.any():
            # Fallback if no points above threshold
            return max_xc, max_yc, lon[j_max], lat[i_max], max_vort, max_vort, max_xc, max_yc

        # Calculate weighted centroid using SHIFTED coordinates
        vort_weights = np.where(mask, vort_data, 0)
        total_weight = np.nansum(vort_weights)

        if total_weight == 0:
            return max_xc, max_yc, lon[j_max], lat[i_max], max_vort, max_vort, max_xc, max_yc

        # Centroid in shifted coordinate system
        dx_centroid = np.nansum(dx_grid * vort_weights) / total_weight
        dy_centroid = np.nansum(dy_grid * vort_weights) / total_weight

        # Convert back to absolute coordinates with periodic wrapping
        center_xc = (max_xc + dx_centroid) % Lx
        center_yc = (max_yc + dy_centroid) % Ly

        # Find nearest grid point for lon/lat and vorticity value
        i_center = np.argmin(np.abs(yc - center_yc))
        j_center = np.argmin(np.abs(xc - center_xc))

        center_lat = lat[i_center]
        center_lon = lon[j_center]
        vort_at_center = vort_data[i_center, j_center]

        return (center_xc, center_yc, center_lon, center_lat,
                vort_at_center, max_vort, max_xc, max_yc)

    # Process all time steps
    n_times = vort_smoothed.sizes['time']

    # Auto-detect parallel strategy based on grid size
    if parallel_center == 'auto':
        ny, nx = vort_smoothed.shape[1], vort_smoothed.shape[2]
        grid_size = ny * nx
        # Use parallel for large grids (>500×500 = 250k points)
        use_parallel = grid_size > 250000
    else:
        use_parallel = parallel_center

    if use_parallel and n_times > 1:
        # Parallel processing
        from joblib import Parallel, delayed
        import os

        # Get all vorticity data at once
        vort_all_data = vort_smoothed.values

        # Disable BLAS threading to avoid over-subscription
        old_omp = os.environ.get('OMP_NUM_THREADS')
        old_mkl = os.environ.get('MKL_NUM_THREADS')
        old_openblas = os.environ.get('OPENBLAS_NUM_THREADS')

        try:
            os.environ['OMP_NUM_THREADS'] = '1'
            os.environ['MKL_NUM_THREADS'] = '1'
            os.environ['OPENBLAS_NUM_THREADS'] = '1'

            results = Parallel(n_jobs=n_jobs)(
                delayed(_compute_local_centroid_2d)(
                    vort_all_data[t], xc, yc, lon, lat, search_radius, threshold_percentile
                )
                for t in range(n_times)
            )

        finally:
            # Restore original threading settings
            if old_omp is not None:
                os.environ['OMP_NUM_THREADS'] = old_omp
            elif 'OMP_NUM_THREADS' in os.environ:
                del os.environ['OMP_NUM_THREADS']

            if old_mkl is not None:
                os.environ['MKL_NUM_THREADS'] = old_mkl
            elif 'MKL_NUM_THREADS' in os.environ:
                del os.environ['MKL_NUM_THREADS']

            if old_openblas is not None:
                os.environ['OPENBLAS_NUM_THREADS'] = old_openblas
            elif 'OPENBLAS_NUM_THREADS' in os.environ:
                del os.environ['OPENBLAS_NUM_THREADS']

        # Unpack results
        results_array = np.array(results)
        center_xcs = results_array[:, 0]
        center_ycs = results_array[:, 1]
        center_lons = results_array[:, 2]
        center_lats = results_array[:, 3]
        center_vorts = results_array[:, 4]
        max_vorts = results_array[:, 5]
        max_xcs = results_array[:, 6]
        max_ycs = results_array[:, 7]

    else:
        # Serial processing
        center_xcs = np.zeros(n_times)
        center_ycs = np.zeros(n_times)
        center_lons = np.zeros(n_times)
        center_lats = np.zeros(n_times)
        center_vorts = np.zeros(n_times)
        max_vorts = np.zeros(n_times)
        max_xcs = np.zeros(n_times)
        max_ycs = np.zeros(n_times)

        for t in range(n_times):
            vort_data = vort_smoothed.isel(time=t).values
            results = _compute_local_centroid_2d(
                vort_data, xc, yc, lon, lat, search_radius, threshold_percentile
            )
            (center_xcs[t], center_ycs[t], center_lons[t], center_lats[t],
             center_vorts[t], max_vorts[t], max_xcs[t], max_ycs[t]) = results

    # Return as xarray Dataset
    center_ds = xr.Dataset(
        {
            'center_xc': (['time'], center_xcs),
            'center_yc': (['time'], center_ycs),
            'center_lon': (['time'], center_lons),
            'center_lat': (['time'], center_lats),
            'center_vorticity': (['time'], center_vorts),
            'center_max_vorticity': (['time'], max_vorts),
            'max_xc': (['time'], max_xcs),
            'max_yc': (['time'], max_ycs),
        },
        coords={'time': vort_smoothed.time.values},
        attrs={
            'method': 'vorticity_centroid_local',
            'smooth_sigma': smooth_sigma,
            'search_radius': search_radius,
            'threshold_percentile': threshold_percentile,
            'description': 'Typhoon center locations found by local vorticity centroid method'
        }
    )
    return center_ds


__all__ = [
    'smooth_vorticity',
    'find_center_by_vorticity_maximum',
    'find_center_by_vorticity_centroid',
]
