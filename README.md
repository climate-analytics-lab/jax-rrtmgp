# JAX-RRTMGP: JAX-based RRTMGP Radiative Transfer

JAX-RRTMGP is a JAX-based implementation of the RRTMGP (Rapid Radiative Transfer Model for General circulation models - Parallel) radiative transfer scheme. This package provides fast, differentiable radiative transfer calculations for atmospheric modeling applications.

RRTMGP is a correlated k-distribution model for computing optical depths, source functions, and fluxes for longwave and shortwave radiation in planetary atmospheres. This JAX implementation enables automatic differentiation and efficient execution on GPUs and TPUs.

## Features

- **JAX-native implementation**: Full compatibility with JAX transformations (jit, grad, vmap, pmap)
- **GPU/TPU acceleration**: Efficient execution on modern accelerators
- **Automatic differentiation**: Enable gradient-based optimization and sensitivity analysis [IN PROGRESS]
- **Longwave and shortwave radiation**: Complete radiative transfer calculations
- **Gas and cloud optics**: Support for molecular absorption and cloud scattering/absorption
- **RRTMGP data files included**: Pre-computed optical property lookup tables
- **Comprehensive testing**: Extensive test suite with reference data

## Installation

It is recommended to install in a virtual environment.

```shell
git clone https://github.com/climate-analytics-lab/jax-rrtmgp.git
python3 -m pip install -e jax-rrtmgp
```

## Basic Usage

```python
import jax
import jax.numpy as jnp
from rrtmgp import rrtmgp

# Set up atmospheric state
temperature = jnp.array([...])  # Temperature profile [K]
pressure = jnp.array([...])     # Pressure profile [Pa] 
vmr_h2o = jnp.array([...])      # Water vapor volume mixing ratio

# Initialize RRTMGP
config = rrtmgp.get_default_config()
rrtmgp_state = rrtmgp.initialize(config, pressure, temperature, vmr_h2o)

# Compute radiative fluxes
fluxes_lw = rrtmgp.compute_longwave_fluxes(rrtmgp_state, ...)
fluxes_sw = rrtmgp.compute_shortwave_fluxes(rrtmgp_state, ...)
```

## Package Structure

- `rrtmgp`: Main RRTMGP interface
- `rrtmgp.optics`: Gas and cloud optical property calculations
- `rrtmgp.rte`: Radiative transfer equation solvers
- `rrtmgp.optics.rrtmgp_data`: Pre-computed lookup tables

## Dependencies

- JAX: Automatic differentiation and JIT compilation
- NumPy: Numerical computing
- NetCDF4: Reading RRTMGP data files

## Data cache

The netCDF data files are copied out of the package into a local cache before
they are read. The cache is `/tmp/netcdf/data` by default; set the environment
variable `RRTMGP_NETCDF_CACHE_DIR` before importing `rrtmgp` to put it elsewhere
(for example on a host where `/tmp/netcdf` belongs to another user).

## Testing

Run the test suite with:

```bash
pytest rrtmgp/
```

Or run individual test modules:

```bash
python rrtmgp/optics/gas_optics_test.py
python rrtmgp/rte/two_stream_test.py
```

## License

Licensed under the Apache License, Version 2.0. This implementation is based on the JAX port of the original RRTMGP Fortran code by Eli Mlawer and Robert Pincus, originally included in [swirl-jatmos](https://github.com/google-research/swirl-jatmos).

## Citation

If you use this software, please cite:

1. The original RRTMGP paper: Pincus, R., Mlawer, E. J., and Delamere, J. S.: Balancing accuracy, efficiency, and flexibility in radiation calculations for dynamical models, J. Adv. Model. Earth Syst., 11, 3074-3089, 2019.

2. This JAX implementation: [Add citation when published]