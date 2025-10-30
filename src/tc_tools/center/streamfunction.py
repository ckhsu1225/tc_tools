"""
Streamfunction-based typhoon center finding

This module provides functions to locate typhoon centers using the streamfunction
method, which is more robust than direct vorticity maximization, especially for
weak vortices in idealized simulations.

Theory:
-------
The streamfunction ψ is defined by:
    u = -∂ψ/∂y
    v = ∂ψ/∂x
    ∇²ψ = ζ  (vorticity)

For a cyclonic vortex (NH), the streamfunction has a minimum at the center.
For an anticyclonic vortex (SH), the streamfunction has a maximum at the center.

The streamfunction is computed by solving the Poisson equation ∇²ψ = ζ using
spectral methods (FFT) which is fast and accurate for periodic domains.
"""

import warnings
import numpy as np
import xarray as xr
import numpy.typing as npt
from scipy import fft as sfft
from typing import Literal, Optional
from scipy.sparse.linalg import LinearOperator, cg

ArrayF = npt.NDArray[np.floating]
BoolA = npt.NDArray[np.bool_]


def compute_streamfunction_2d(
    vorticity: ArrayF,
    dx: float,
    dy: float,
    method: Literal['spectral', 'pcg'] = 'spectral',
    fluid_mask: Optional[BoolA] = None,
    *,
    tol: float = 1e-6,
    maxiter: int = 500,
    fft_workers: int = -1,
    dtype: Literal['float64', 'float32'] = 'float64',
) -> ArrayF:
    """
    Compute the 2-D streamfunction ψ from vorticity ζ by solving the Poisson equation:

        ∇²ψ = ζ

    Parameters
    ----------
    vorticity : (ny, nx) array_like
        Vorticity field ζ at scalar (cell-center) points.
    dx, dy : float
        Grid spacing [m].
    method : {'spectral', 'iterative', 'pcg'}, optional
        - 'spectral': Periodic FFT solver (fast; *no* terrain support).
        - 'pcg': Matrix-free PCG with FFT preconditioner (handles terrain via mask).
    fluid_mask : (ny, nx) bool array, optional
        True for fluid cells, False for terrain/solid. Required for 'pcg'.
    tol : float, optional
        Target relative tolerance for the linear solve (PCG / SOR).
    maxiter : int, optional
        Maximum iterations for the linear solve (PCG / SOR).
    fft_workers : int, optional
        Number of worker threads for SciPy FFT (e.g., -1 to use all cores).
    dtype : {'float64','float32'}, optional
        Internal compute dtype. Use 'float32' to reduce memory/boost speed if accuracy allows.

    Returns
    -------
    psi : (ny, nx) ndarray
        Streamfunction field.

    Notes
    -----
    - 'spectral' assumes fully periodic domain *without* internal terrain.
    - 'pcg' enforces Dirichlet ψ=0 inside terrain cells and periodic on the outer box.
      It uses an FFT-based inverse Poisson as right-preconditioner and an FFT-based
      defect-correction pass to reduce boundary residuals.
    - For performance, pass plain NumPy arrays (not xarray) and prefer SciPy FFT with workers.
    """
    zeta = np.asarray(vorticity, dtype=np.float32 if dtype == 'float32' else np.float64, order='C')

    if method == 'spectral':
        return _solve_poisson_spectral(zeta, dx, dy, fft_workers=fft_workers)
    elif method == 'pcg':
        if fluid_mask is None:
            raise ValueError("method='pcg' requires a boolean fluid_mask (True=fluid, False=terrain).")
        mask = np.asarray(fluid_mask, dtype=bool, order='C')
        return _solve_poisson_pcg(
            zeta, mask, dx, dy, tol=tol, maxiter=maxiter, fft_workers=fft_workers, dtype=zeta.dtype
        )
    else:
        raise ValueError(f"Unknown method: {method!r}")


# ---------------------------- Periodic FFT solver ---------------------------- #

def _solve_poisson_spectral(zeta: ArrayF, dx: float, dy: float, *, fft_workers: int = -1) -> ArrayF:
    """Fast periodic Poisson solver using rFFT: -(kx^2+ky^2) ψ̂ = ζ̂ (zero-mean RHS)."""
    ny, nx = zeta.shape
    rhs = zeta - zeta.mean()

    # Real FFT plan & wavenumbers
    kx = 2*np.pi*sfft.fftfreq(nx, d=dx)
    ky = 2*np.pi*sfft.fftfreq(ny, d=dy)
    K2 = (ky[:, None]**2) + (kx[None, :nx//2 + 1]**2)  # rfft2 shape (ny, nx//2+1)

    rhs_hat = sfft.rfft2(rhs, workers=fft_workers)
    mask = K2 != 0.0
    rhs_hat[~mask] = 0.0
    rhs_hat[mask] *= -1.0 / K2[mask]
    psi = sfft.irfft2(rhs_hat, s=(ny, nx), workers=fft_workers)
    return psi

# ----------------------- Matrix-free PCG with FFT-PC ------------------------ #

def _laplacian_periodic(f: ArrayF, dx: float, dy: float) -> ArrayF:
    """Five-point Laplacian with periodic wraps."""
    return ((np.roll(f, -1, axis=1) - 2*f + np.roll(f, 1, axis=1))/dx**2 +
            (np.roll(f, -1, axis=0) - 2*f + np.roll(f, 1, axis=0))/dy**2)


def _fft_poisson(rhs: ArrayF, dx: float, dy: float, *, fft_workers: int = -1) -> ArrayF:
    """Periodic inverse-Poisson used as preconditioner (rFFT, zero-mean RHS)."""
    ny, nx = rhs.shape
    rhs = rhs - rhs.mean()
    kx = 2*np.pi*sfft.fftfreq(nx, d=dx)
    ky = 2*np.pi*sfft.fftfreq(ny, d=dy)
    K2 = (ky[:, None]**2) + (kx[None, :nx//2 + 1]**2)

    rhat = sfft.rfft2(rhs, workers=fft_workers)
    mask = K2 != 0.0
    rhat[~mask] = 0.0
    rhat[mask] *= -1.0 / K2[mask]
    return sfft.irfft2(rhat, s=(ny, nx), workers=fft_workers)


def _apply_A_vec(
    psi_vec: ArrayF, fluid_mask: BoolA, dx: float, dy: float, nx: int, ny: int
) -> ArrayF:
    """
    Matrix-free application of Aψ:
        Aψ = ∇²ψ  (periodic)  on fluid cells
        Dirichlet ψ=0 inside terrain cells
    Returned on all cells, but values on terrain are zeroed (no equation there).
    """
    psi = psi_vec.reshape(ny, nx)
    psi_bc = np.where(fluid_mask, psi, 0.0)          # enforce Dirichlet in terrain
    Ap = _laplacian_periodic(psi_bc, dx, dy)
    Ap = np.where(fluid_mask, Ap, 0.0)
    return Ap.ravel()


def _solve_poisson_pcg(
    zeta: ArrayF,
    fluid_mask: BoolA,
    dx: float,
    dy: float,
    *,
    tol: float = 1e-6,
    maxiter: int = 500,
    fft_workers: int = -1,
    dtype=np.float64,
) -> ArrayF:
    """
    Solve ∇²ψ = ζ with Dirichlet ψ=0 in terrain, periodic elsewhere,
    via Preconditioned Conjugate Gradient (matrix-free A, FFT preconditioner).

    - Initial guess x0 uses periodic FFT solution ignoring terrain to reduce iterations.
    - After PCG, apply one FFT-based defect-correction pass to reduce boundary residuals.
    """
    ny, nx = zeta.shape
    b = np.where(fluid_mask, zeta, 0.0)

    # Initial guess: periodic solution (ignoring terrain)
    x0 = _fft_poisson(b, dx, dy, fft_workers=fft_workers)

    # Linear operator A and right-preconditioner M^{-1}
    A = LinearOperator(
        (b.size, b.size),
        matvec=lambda x: _apply_A_vec(x, fluid_mask, dx, dy, nx, ny),
        dtype=dtype,
    )
    M = LinearOperator(
        (b.size, b.size),
        matvec=lambda x: _fft_poisson(x.reshape(ny, nx), dx, dy, fft_workers=fft_workers).ravel(),
        dtype=dtype,
    )

    its = {'k': 0}
    def _cb(_): its['k'] += 1

    psi_vec, info = cg(A, b.ravel(), x0=x0.ravel(), rtol=tol, maxiter=maxiter, M=M, callback=_cb)
    if info > 0:
        warnings.warn(f"PCG reached max iterations ({info}); residual may remain {tol:g} level.")
    elif info < 0:
        warnings.warn("PCG failed to converge (numerical breakdown).")

    psi = psi_vec.reshape(ny, nx)
    psi[~fluid_mask] = 0.0

    # One defect-correction sweep to clean boundary residuals
    r = np.where(fluid_mask, zeta - _laplacian_periodic(psi, dx, dy), 0.0)
    psi += _fft_poisson(r, dx, dy, fft_workers=fft_workers)
    psi[~fluid_mask] = 0.0

    # Optional: print iterations used
    # print(f"PCG iterations: {its['k']}")

    return psi.astype(dtype, copy=False)


def compute_streamfunction(
    vort: xr.DataArray,
    poisson_method: Literal['spectral', 'pcg', 'auto'] = 'auto',
    parallel_method: Literal['auto', 'joblib', 'serial'] = 'auto',
    n_jobs: int = -1
) -> xr.DataArray:
    """
    Compute streamfunction from vorticity field for xarray DataArrays.

    This is a convenient wrapper around compute_streamfunction_2d that handles
    xarray DataArrays with time dimension and supports parallel processing.

    Parameters
    ----------
    vort : xr.DataArray
        Vorticity field with dimensions (time, lat, lon).
        Coordinates 'xc' and 'yc' must be present for grid spacing calculation.
    poisson_method : {'spectral', 'pcg', 'auto'}, optional
        Method for solving Poisson equation:
        - 'spectral': Fast FFT method (assumes no terrain/NaN)
        - 'pcg': Preconditioned conjugate gradient (handles terrain/NaN)
        - 'auto': Automatically detect terrain and choose method
        Default: 'auto'
    parallel_method : {'auto', 'joblib', 'serial'}, optional
        Parallelization strategy for processing multiple time steps:
        - 'auto': Automatically choose (serial for <=10 steps, joblib for >10)
        - 'joblib': Use joblib parallel processing (fastest for most cases)
        - 'serial': Simple for loop (good for small datasets or debugging)
        Default: 'auto'
    n_jobs : int, optional
        Number of parallel jobs for joblib. -1 uses all CPU cores.
        Ignored when parallel_method='serial'. Default: -1

    Returns
    -------
    psi : xr.DataArray
        Streamfunction field with same dimensions and coordinates as input.

    Raises
    ------
    ValueError
        If required coordinates not found or invalid method specified.

    Examples
    --------
    >>> import tc_tools as tct
    >>> import vvm_reader as vvm
    >>>
    >>> # Load vorticity data
    >>> ds = vvm.open_vvm_dataset('/path/to/sim', variables=['zeta'])
    >>> zeta = ds['zeta'].sel(lev=1000, method='nearest')
    >>>
    >>> # Compute streamfunction with auto parallelization
    >>> psi = tct.compute_streamfunction(zeta)
    >>>
    >>> # Force serial processing
    >>> psi = tct.compute_streamfunction(zeta, parallel_method='serial')
    >>>
    >>> # Use PCG method for terrain
    >>> psi = tct.compute_streamfunction(zeta, poisson_method='pcg')
    >>>
    >>> # Use joblib with 4 cores
    >>> psi = tct.compute_streamfunction(zeta, poisson_method='spectral',
    ...                                   parallel_method='joblib', n_jobs=4)

    Notes
    -----
    - For VVM simulations with periodic boundaries, 'spectral' method is fastest
    - If terrain (NaN values) detected with method='auto', switches to PCG
    - Joblib parallelization typically provides 2-4x speedup on multi-core systems
    - For datasets already in memory, joblib is usually faster than dask
    """
    # Validate coordinates
    if 'xc' not in vort.coords or 'yc' not in vort.coords:
        raise ValueError("DataArray must contain 'xc' and 'yc' Cartesian coordinates (in meters). "
                        "VVM uses Cartesian grid; lat/lon are derived coordinates.")

    # Validate poisson_method
    valid_methods = {'spectral', 'pcg', 'auto'}
    if poisson_method not in valid_methods:
        raise ValueError(f"method must be one of {valid_methods}, got {poisson_method!r}")

    # Validate parallel_method
    valid_parallel = {'auto', 'joblib', 'serial'}
    if parallel_method not in valid_parallel:
        raise ValueError(f"parallel_method must be one of {valid_parallel}, got {parallel_method!r}")

    # Ensure time dimension exists
    if 'time' not in vort.dims:
        raise ValueError("Input must have 'time' dimension. "
                        "For single time step, use vort.isel(time=[0]) to preserve dimension.")

    # Compute grid spacing
    xc = vort.xc.values
    yc = vort.yc.values
    dx = float(xc[1] - xc[0])
    dy = float(yc[1] - yc[0])

    # Detect terrain and choose poisson_method
    has_terrain = np.isnan(vort.values).any()
    if poisson_method == 'auto':
        poisson_method = 'pcg' if has_terrain else 'spectral'
        if has_terrain:
            warnings.warn(
                "Detected NaN values (terrain). Using 'pcg' method for terrain handling.",
                UserWarning
            )
    elif poisson_method == 'spectral' and has_terrain:
        raise ValueError("Cannot use 'spectral' method with terrain (NaN values). Use poisson_method='pcg' or 'auto'.")

    # Define 2D computation function
    def _compute_psi_2d(vort_2d):
        """Compute streamfunction for single 2D field"""
        fluid_mask = ~np.isnan(vort_2d) if poisson_method == 'pcg' else None
        return compute_streamfunction_2d(vort_2d, dx, dy, method=poisson_method, fluid_mask=fluid_mask)

    n_times = vort.sizes['time']

    # Compute if dask array (joblib works better with in-memory data)
    vort_computed = vort.compute().values if hasattr(vort.data, 'compute') else vort.values

    # Auto-select parallel method
    if parallel_method == 'auto':
        parallel_method = 'serial' if n_times <= 10 else 'joblib'

    # Serial processing
    if parallel_method == 'serial':
        results = [
            _compute_psi_2d(vort_computed[t])
            for t in range(n_times)
        ]
        psi_data = np.stack(results, axis=0)

    # Joblib parallel processing
    elif parallel_method == 'joblib':
        from joblib import Parallel, delayed
        import os

        # Disable BLAS/FFT threading to avoid over-subscription
        old_omp = os.environ.get('OMP_NUM_THREADS')
        old_mkl = os.environ.get('MKL_NUM_THREADS')
        old_openblas = os.environ.get('OPENBLAS_NUM_THREADS')

        try:
            os.environ['OMP_NUM_THREADS'] = '1'
            os.environ['MKL_NUM_THREADS'] = '1'
            os.environ['OPENBLAS_NUM_THREADS'] = '1'

            results = Parallel(n_jobs=n_jobs)(
                delayed(_compute_psi_2d)(vort_computed[t])
                for t in range(n_times)
            )
            psi_data = np.stack(results, axis=0)

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
    psi = xr.DataArray(
        psi_data,
        coords=vort.coords,
        dims=vort.dims,
        attrs=vort.attrs.copy()
    )
    psi.attrs['long_name'] = 'Streamfunction'
    psi.attrs['units'] = 'm²/s'
    psi.attrs['poisson_method'] = poisson_method
    psi.attrs['parallel_method'] = parallel_method

    return psi


def find_center_by_streamfunction(
    vort: xr.DataArray,
    method: Literal['spectral', 'pcg', 'auto'] = 'auto',
) -> xr.Dataset:
    """
    Find typhoon centers using streamfunction method.

    The center is identified as the minimum of the streamfunction field
    computed from vorticity (Northern Hemisphere convention).

    Parameters
    ----------
    vort : xr.DataArray
        Vorticity field with dimensions (time, lat, lon).
        User should select vertical level before calling this function.
        Coordinates 'xc', 'yc', 'lon', 'lat' must be present.
        Can be raw or pre-smoothed vorticity.
    method : {'spectral', 'pcg', 'auto'}, optional
        Method for solving Poisson equation:
        - 'spectral': Fast FFT method (assumes no terrain)
        - 'pcg': Preconditioned conjugate gradient (handles terrain/NaN)
        - 'auto': Automatically detect terrain and choose method
        Default: 'auto'

    Returns
    -------
    centers : xr.Dataset
        Dataset with time dimension containing:
        - 'center_xc' (time,) : Center x-coordinates in meters
        - 'center_yc' (time,) : Center y-coordinates in meters
        - 'center_lon' (time,) : Center longitudes
        - 'center_lat' (time,) : Center latitudes
        - 'center_streamfunction' (time,) : Streamfunction values at centers
        - 'center_vorticity' (time,) : Vorticity values at centers

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
    >>>
    >>> # Select vertical level (1 km height)
    >>> vort = ds['zeta'].sel(lev=1000, method='nearest')
    >>>
    >>> # Find centers for all time steps
    >>> centers = tct.find_center_by_streamfunction(vort)
    >>> print(centers.center_lon.values)  # Array of longitudes
    >>>
    >>> # Save to CSV
    >>> centers.to_dataframe().to_csv('centers.csv')
    >>>
    >>> # Explicitly use PCG method for terrain
    >>> centers = tct.find_center_by_streamfunction(vort, method='pcg')
    >>>
    >>> # Use vertical average
    >>> vort_avg = ds['zeta'].sel(lev=slice(500, 2000)).mean('lev')
    >>> centers = tct.find_center_by_streamfunction(vort_avg)

    Notes
    -----
    - For VVM simulations with periodic boundaries, this method is optimal
    - The streamfunction is smoother than vorticity, making it more robust for weak vortices
    - VVM simulations are assumed to be in Northern Hemisphere convention
    - Typhoon center is identified as streamfunction minimum
    - If terrain (NaN values) detected with method='auto', automatically switches to PCG
    - PCG method is slower but handles terrain correctly
    - Designed for single-typhoon VVM simulations (searches entire domain)
    """
    # Validate input
    if 'xc' not in vort.coords or 'yc' not in vort.coords:
        raise ValueError("DataArray must contain 'xc' and 'yc' Cartesian coordinates (in meters). "
                        "VVM uses Cartesian grid; lat/lon are derived coordinates.")

    # Ensure time dimension exists
    if 'time' not in vort.dims:
        raise ValueError("DataArray must have 'time' dimension. "
                        "For single time step, use vort.isel(time=[0]) to preserve dimension.")

    # Compute streamfunction using the existing function
    # This handles terrain detection, parallelization, and all edge cases
    psi = compute_streamfunction(vort, poisson_method=method)

    # Vectorized center finding for all time steps
    n_times = psi.sizes['time']
    xc = vort.xc.values
    yc = vort.yc.values
    lon = vort.lon.values
    lat = vort.lat.values

    # Get all data at once (time, lat, lon)
    psi_data = psi.values
    #vort_data = vort.values

    # Reshape to (time, lat*lon) for vectorized argmin
    ny, nx = psi_data.shape[1], psi_data.shape[2]
    psi_flat = psi_data.reshape(n_times, -1)

    # Find minimum indices for all time steps at once
    # For streamfunction, we find MINIMUM (not maximum like vorticity)
    min_flat_indices = np.nanargmin(psi_flat, axis=1)  # (time,)

    # Convert flat indices back to 2D indices
    i_indices = min_flat_indices // nx  # row indices (lat)
    j_indices = min_flat_indices % nx   # col indices (lon)

    # Get values at minimum locations using advanced indexing
    center_psis = psi_data[np.arange(n_times), i_indices, j_indices]
    #center_vorts = vort_data[np.arange(n_times), i_indices, j_indices]

    # Get coordinates at minimum locations
    center_xcs = xc[j_indices]
    center_ycs = yc[i_indices]
    center_lons = lon[j_indices]
    center_lats = lat[i_indices]

    # Handle all-NaN time steps (if any)
    all_nan_mask = np.all(np.isnan(psi_flat), axis=1)
    center_xcs[all_nan_mask] = np.nan
    center_ycs[all_nan_mask] = np.nan
    center_lons[all_nan_mask] = np.nan
    center_lats[all_nan_mask] = np.nan
    center_psis[all_nan_mask] = np.nan
    #center_vorts[all_nan_mask] = np.nan

    # Return as xarray Dataset
    center_ds = xr.Dataset(
        {
            'center_xc': (['time'], center_xcs),
            'center_yc': (['time'], center_ycs),
            'center_lon': (['time'], center_lons),
            'center_lat': (['time'], center_lats),
            'center_streamfunction': (['time'], center_psis),
            #'center_vorticity': (['time'], center_vorts),
        },
        coords={'time': psi.time.values},
        attrs={
            'method': 'streamfunction',
            'solver': psi.attrs.get('poisson_method', method),  # Get actual method used
            'description': 'Typhoon center locations found by streamfunction method'
        }
    )
    return center_ds


__all__ = [
    'compute_streamfunction_2d',
    'compute_streamfunction',
    'find_center_by_streamfunction',
]
