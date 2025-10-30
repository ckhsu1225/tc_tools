# TC Tools

Tropical Cyclone Analysis Tools for VVM Simulations

## Overview

`tc_tools` is a Python package for analyzing tropical cyclone structure and evolution in VVM (Vector Vorticity equation cloud-resolving Model) simulations. It provides a comprehensive set of tools for:

- **Center Finding**: Locate typhoon centers using streamfunction or vorticity methods
- **Coordinate Transformation**: Transform to storm-relative cylindrical coordinates
- **Wind Decomposition**: Compute tangential and radial wind components
- **Azimuthal Averaging**: Obtain axisymmetric structure
- **Intensity Analysis**: Compute maximum wind speed, RMW, and vertical profiles
- **Size Metrics**: Calculate wind radii (R34, R50, R64) and other size metrics

## Installation

### Option 1: Using uv (Recommended)

```bash
# Install in development mode with uv
uv pip install -e .

# Or use uv sync (creates/updates .venv automatically)
uv sync
```

### Option 2: Using pip

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in development mode
pip install -e .
```

## Quick Start

```python
import vvm_reader as vvm
import tc_tools as tct

# 1. Load VVM data
ds = vvm.open_vvm_dataset('/path/to/sim', variables=['u', 'v', 'w', 'zeta'])

# 2. Find typhoon center
vort = ds['zeta'].sel(lev=1000, method='nearest')  # Select level first
centers = tct.find_center_by_streamfunction(vort)

# 3. Transform to cylindrical coordinates
ds_cyl = tct.transform_to_cylindrical(ds, centers)

# 4. Compute tangential and radial winds
ds_winds = tct.compute_tangential_radial_winds(ds_cyl)

# 5. Azimuthal average
ds_axi = tct.azimuthal_mean(ds_winds, r_bins=50, r_max=200)

# 6. Compute intensity
intensity = tct.compute_intensity(ds_axi, level=1000)
print(f"Vmax: {intensity['vmax']:.1f} m/s, RMW: {intensity['rmw']:.1f} km")
```

## Package Structure

```
tc_tools/
├── center/              # Center finding
│   ├── streamfunction.py   - Streamfunction method (recommended)
│   └── vorticity.py        - Vorticity-based methods (maximum, centroid)
├── transform/           # Coordinate transformations
│   ├── coordinates.py      - Cylindrical coordinates
│   └── winds.py            - Wind decomposition
├── composite/           # Azimuthal averaging
│   └── azimuthal.py        - Azimuthal mean and anomaly
├── intensity/           # Intensity metrics
│   └── wind.py             - Wind-based intensity metrics
├── structure/           # Size metrics
│   └── size.py             - Wind radii and size metrics
└── utils/               # Utilities
    └── distance.py         - Distance calculations
```

## Documentation

See `examples/basic_workflow.py` for a complete workflow example.

All functions have comprehensive docstrings with parameter descriptions and usage examples.

## Requirements

- Python >= 3.10
- numpy, xarray, scipy, pandas
- vvm-reader (https://github.com/ckhsu1225/vvm_reader)

