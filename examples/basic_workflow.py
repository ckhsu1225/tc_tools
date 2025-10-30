"""
Basic TC Tools Workflow Example

This script demonstrates the complete workflow for analyzing tropical cyclone
structure in VVM simulations using tc_tools.

Workflow:
1. Load VVM data with vvm_reader
2. Find typhoon center
3. Transform to cylindrical coordinates
4. Decompose winds into tangential/radial components
5. Compute azimuthal mean (axisymmetric structure)
6. Analyze intensity and size metrics
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import vvm_reader as vvm
import tc_tools as tct

# =============================================================================
# Configuration
# =============================================================================

# Path to VVM simulation directory
SIM_DIR = "/path/to/your/simulation"

# Analysis parameters
TIME_INDEX = 0  # First time step
ANALYSIS_LEVEL = 1000  # 1 km height
R_MAX = 200  # Maximum radius for analysis (km)
N_R_BINS = 50  # Number of radial bins

# =============================================================================
# Step 1: Load VVM Data
# =============================================================================

print("Loading VVM data...")
ds = vvm.open_vvm_dataset(
    SIM_DIR,
    variables=['u', 'v', 'w', 'th', 'qv'],
    time_selection=vvm.TimeSelection(time_index_range=(TIME_INDEX, TIME_INDEX+1)),
    vertical_selection=vvm.VerticalSelection(height_range=(0, 15000)),
    auto_compute_diagnostics=True  # Compute T, theta_v, etc.
)

print(f"Loaded dataset with dimensions: {dict(ds.dims)}")

# =============================================================================
# Step 2: Compute Vorticity (if not already present)
# =============================================================================

if 'zeta' not in ds:
    print("Computing vorticity...")
    # Note: You need to implement this function based on your grid
    # For Arakawa-C staggered grid:
    # vorticity = (dv/dx - du/dy) on the scalar grid
    # This is left as an exercise - vvm_reader should ideally provide this
    raise NotImplementedError("Vorticity computation needs to be implemented")

# =============================================================================
# Step 3: Find Typhoon Center
# =============================================================================

print("Finding typhoon center using streamfunction method...")
# Select vertical level first, then pass vorticity DataArray
vort = ds['zeta'].sel(lev=ANALYSIS_LEVEL, method='nearest')
centers = tct.find_center_by_streamfunction(vort)

print(f"Center found at:")
print(f"  Cartesian: ({float(centers.center_xc.values[0]):.0f} m, {float(centers.center_yc.values[0]):.0f} m)")
print(f"  Geographic: ({float(centers.center_lon.values[0]):.2f}, {float(centers.center_lat.values[0]):.2f})")

# =============================================================================
# Step 4: Transform to Cylindrical Coordinates
# =============================================================================

print("Transforming to cylindrical coordinates...")
ds_cyl = tct.transform_to_cylindrical(ds.isel(time=0), centers)

print(f"Added 'r' and 'azimuth' data variables")
print(f"Radius range: {float(ds_cyl['r'].min()):.1f} - {float(ds_cyl['r'].max()):.1f} km")

# =============================================================================
# Step 5: Compute Tangential and Radial Winds
# =============================================================================

print("Computing tangential and radial winds...")
ds_winds = tct.compute_tangential_radial_winds(ds_cyl)

print(f"Added 'vt' (tangential) and 'vr' (radial) variables")

# =============================================================================
# Step 6: Compute Azimuthal Mean (Axisymmetric Structure)
# =============================================================================

print(f"Computing azimuthal mean with {N_R_BINS} radial bins...")
ds_axi = tct.azimuthal_mean(
    ds_winds,
    r_bins=N_R_BINS,
    r_max=R_MAX,
    variables=['vt', 'vr', 'w', 'T', 'qv']
)

print(f"Axisymmetric dataset shape: {dict(ds_axi.dims)}")

# =============================================================================
# Step 7: Compute Intensity Metrics
# =============================================================================

print("Computing intensity metrics...")
# compute_intensity returns Dataset with (time, lev) dimensions
intensity = tct.compute_intensity(ds_axi)

# Select analysis level
intensity_1km = intensity.sel(lev=ANALYSIS_LEVEL, method='nearest')

print(f"\nIntensity Metrics at {ANALYSIS_LEVEL} m:")
print(f"  Vmax: {float(intensity_1km['vmax'].values[0]):.1f} m/s ({float(intensity_1km['vmax'].values[0])*1.944:.1f} kt)")
print(f"  RMW:  {float(intensity_1km['rmw'].values[0]):.1f} km")

# Get vertical profile of Vmax
vmax_profile = intensity.isel(time=0).vmax  # Select first time step
print(f"  Vmax at lowest level: {float(vmax_profile.isel(lev=0)):.1f} m/s")
print(f"  Vmax at highest level: {float(vmax_profile.isel(lev=-1)):.1f} m/s")

# =============================================================================
# Step 8: Compute Size Metrics
# =============================================================================

print("\nComputing size metrics...")
# compute_size_metrics returns Dataset with (time, lev) dimensions
size = tct.compute_size_metrics(ds_axi)

# Select analysis level
size_1km = size.sel(lev=ANALYSIS_LEVEL, method='nearest').isel(time=0)

print(f"Size Metrics at {ANALYSIS_LEVEL} m:")
for var_name in ['r34', 'r50', 'r64']:
    if var_name in size_1km:
        value = float(size_1km[var_name].values)
        if np.isnan(value):
            print(f"  {var_name}: Not reached")
        else:
            print(f"  {var_name}: {value:.1f} km")

# =============================================================================
# Step 9: Visualization
# =============================================================================

print("\nCreating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Panel 1: Tangential wind profile
ax = axes[0, 0]
vt_1km = ds_axi.vt.sel(lev=ANALYSIS_LEVEL, method='nearest')
rmw_val = float(intensity_1km['rmw'].values[0])
vmax_val = float(intensity_1km['vmax'].values[0])
ax.plot(ds_axi.rbin, vt_1km, 'b-', linewidth=2)
ax.axvline(rmw_val, color='r', linestyle='--', label=f"RMW = {rmw_val:.1f} km")
ax.axhline(vmax_val, color='r', linestyle='--', alpha=0.5)
ax.set_xlabel('Radius (km)')
ax.set_ylabel('Tangential Wind (m/s)')
ax.set_title(f'Tangential Wind at {ANALYSIS_LEVEL} m')
ax.grid(True, alpha=0.3)
ax.legend()

# Panel 2: Radial wind profile
ax = axes[0, 1]
vr_1km = ds_axi.vr.sel(lev=ANALYSIS_LEVEL, method='nearest')
ax.plot(ds_axi.rbin, vr_1km, 'g-', linewidth=2)
ax.axhline(0, color='k', linestyle='-', linewidth=0.5)
ax.axvline(rmw_val, color='r', linestyle='--', alpha=0.5)
ax.set_xlabel('Radius (km)')
ax.set_ylabel('Radial Wind (m/s)')
ax.set_title(f'Radial Wind at {ANALYSIS_LEVEL} m')
ax.grid(True, alpha=0.3)

# Panel 3: Vertical structure (radius-height cross-section of vt)
ax = axes[1, 0]
vt_rz = ds_axi.vt.isel(time=0) if 'time' in ds_axi.dims else ds_axi.vt
contour = ax.contourf(ds_axi.rbin, ds_axi.lev/1000, vt_rz, levels=20, cmap='RdYlBu_r')
ax.contour(ds_axi.rbin, ds_axi.lev/1000, vt_rz, levels=10, colors='k', linewidths=0.5, alpha=0.3)
plt.colorbar(contour, ax=ax, label='Tangential Wind (m/s)')
ax.set_xlabel('Radius (km)')
ax.set_ylabel('Height (km)')
ax.set_title('Tangential Wind: Radius-Height Cross-Section')

# Panel 4: Vmax vertical profile
ax = axes[1, 1]
ax.plot(vmax_profile, ds_axi.lev/1000, 'b-', linewidth=2)
ax.set_xlabel('Vmax (m/s)')
ax.set_ylabel('Height (km)')
ax.set_title('Vertical Profile of Maximum Tangential Wind')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('tc_structure_analysis.png', dpi=150, bbox_inches='tight')
print("Saved figure: tc_structure_analysis.png")

plt.show()

print("\nAnalysis complete!")
