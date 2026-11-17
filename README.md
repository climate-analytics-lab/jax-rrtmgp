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

## Performance: placements, layouts and backends

None of the options below changes the physics; each changes the compiled
program, and the fastest one depends on the hardware.

- `gas_optics_mode`, `use_scan` and `gpt_chunk` (arguments of
  `RRTMGP.compute_heating_rate`, `two_stream.solve_lw` and `solve_sw`) choose
  where the gas optics is evaluated, whether the vertical recurrences are
  scanned or unrolled, and how many g-points the spectral loop solves per
  iteration. The library defaults (`'per_gpt'`, unrolled, one g-point) keep
  the historical program.
- With `gas_optics_mode='all_gpts'`, `use_scan=True` and `gpt_chunk > 1`, a 3D
  atmosphere is solved in a **column-minor layout** on every backend: levels
  leading, columns minor, so the columns are the vectorised axis through the
  gas optics, the spectral loop and the vertical scans, with no transposes.
  Measured, one longwave + shortwave call with clear-sky fluxes and McICA
  clouds at T63L47 (18432 columns): 108 ms on an A100 against 243 ms for the
  level-minor solve of the same placement, and 1.2 s against 4.1 s on a
  64-core CPU at 2048 columns.
- `RRTMGP.compute_heating_rate_zc` takes the column-minor layout directly,
  `(n_levels + 2, n_columns)` profiles, for host models that keep their
  columns contiguous (no transposes in or out, and McICA as one-byte masks of
  the bulk cloud paths). See `two_stream.solve_lw_zc` for the layout of every
  input and output.

**TPU.** A gather runs serially on a TPU's vector unit, so on a TPU the table
lookups become contractions on the matrix unit, the vertical scans step a
block of levels at a time, and the g-point chunk of the column-minor solve is
capped so a chunk's arrays fit in vector memory
(`two_stream.TPU_CHUNK_VMEM_BYTES`). These TPU kernels are chosen
automatically from the default device; set `RRTMGP_TPU_KERNELS=1` (or `0`), or
use `rrtmgp.backend.tpu_kernels(True)`, where that is not the device the
computation runs on (e.g. lowering for a TPU from a CPU host). On a TPU v5e
use the column-minor solve on blocks of about 9216 columns, with `gpt_chunk=8`:
a T63L47 call as two such blocks measured 222 ms (265 ms at `gpt_chunk=16`).
Do not `vmap` the solve: its data-dependent branches become selects that run
both sides (measured 3.5x the cost of the same solves one after another).

**Table lookups.** `RRTMGP_LOOKUP_IMPL=gather|matmul` (or
`optics_utils.set_lookup_impl`) selects how the per-g-point gas-optics tables
are read: by direct indexing (the default) or by one-hot contractions. Making
either choice explicitly also switches off the TPU kernels' contraction of the
1D cloud and Planck tables.

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