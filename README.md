# Signaloid Data Toolkit

The Signaloid Data Toolkit provides tools for interacting with
applications that utilize Signaloid's UxHw® technology for distributional
arithmetic. Use the library to analyze Ux Data values from the application.

The tools for benchmarking applications against equivalent Monte Carlo
methods build on this library and live in the separate
[signaloid-benchmarking](https://github.com/signaloid/signaloid-benchmarking)
repository.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/signaloid/signaloid-data-toolkit/main/images/signaloid-external-illustration-signaloid-data-toolkit-diagram-flat-dark-withCR.png">
  <img alt="signaloid-data-toolkit diagram" src="https://raw.githubusercontent.com/signaloid/signaloid-data-toolkit/main/images/signaloid-external-illustration-signaloid-data-toolkit-diagram-flat-light-withCR.png">
</picture>

## Requirements

The Signaloid Data Toolkit requires Python 3.11 or later. See
`pyproject.toml` for the full list of dependencies.

## Installation
Install the `signaloid` package via pip (recommended):
```bash
python -m pip install signaloid
```


Install the latest version from the GitHub repository:
```bash
python -m pip install git+https://github.com/signaloid/signaloid-data-toolkit
```

Alternatively, clone this repository and install from source with:
```bash
python -m pip install .
```

## Usage

### Parsing Ux Data
Construct `DistributionalValue` Python objects by parsing
[Ux Data](https://docs.signaloid.io/docs/uxhw-api/ux-data-format/) in
Ux-string or Ux Binary format.

```python
from signaloid.distributional.distributional import DistributionalValue

# Intermediate code which writes to ux_string and ux_binary_buffer
# ...

# Parse a Ux-string
dist_value = DistributionalValue.parse(ux_string)

# Parse a Ux Binary buffer
dist_value = DistributionalValue.parse(ux_binary_buffer)
```

### Create Distribution Plots
Create plots to visualize distributional information by using the
[`plot` function](https://github.com/signaloid/signaloid-data-toolkit/blob/main/src/signaloid/distributional_information_plotting/plot_wrapper.py)
with a `PlotData` object built from a `DistributionalValue` containing Ux Data.
The `plot` function is a wrapper function for the `PlotHistogramDiracDeltas`
class for plotting a distributional value as a histogram with variable bin
widths.

```python
from signaloid.distributional.distributional import DistributionalValue
from signaloid.distributional_information_plotting.plot_histogram_dirac_deltas import PlotData
from signaloid.distributional_information_plotting.plot_wrapper import plot

# Intermediate code which writes to ux_string
# ...

# Create distributional value object from Ux-string
dist_value = DistributionalValue.parse(ux_string)
plot(PlotData(dist_value))
```

For plotting from raw samples, saving to a file, and the other `plot` options,
see the package [README.md](https://github.com/signaloid/signaloid-data-toolkit/blob/main/src/signaloid/distributional_information_plotting/README.md).

### Sample from Ux Data
Draw random samples from a distributional value with the
[`sample_generator` function](https://github.com/signaloid/signaloid-data-toolkit/blob/main/src/signaloid/distributional_information_plotting/sample_generator.py).
Samples of the finite part of the distribution are drawn by inverse transform
sampling of the binned distribution. Distributions that also carry non-finite
mass (`NaN`, `-Inf`, `+Inf`) are sampled as a mixture, with each sample drawn
from the finite or the non-finite part in proportion to their masses.

```python
from signaloid.distributional_information_plotting.sample_generator import sample_generator

# Intermediate code which writes to ux_string
# ...

samples = sample_generator(ux_string, n_samples=1000)
```

To sample from a `DistributionalValue` that is already parsed, use
`sample_from_distributional_value` from the same module.
