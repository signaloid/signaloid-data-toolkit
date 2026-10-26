# Distributional Information Plotting

Tools for visualizing and sampling from Signaloid distributional data.

## Plotting a Ux-string

Parse a Ux-encoded string into a `DistributionalValue`, build a `PlotData` object, and pass it to `plot()`:

```python
from signaloid.distributional.distributional import DistributionalValue
from signaloid.distributional_information_plotting.plot_histogram_dirac_deltas import PlotData
from signaloid.distributional_information_plotting.plot_wrapper import plot

ux_string = "0.40007Ux0000000000000000013FD99AC12423C7C7000000013FD99AC12423C7C78000000000000000"

dist_value = DistributionalValue.parse(ux_string)
if dist_value is None:
    raise ValueError(f"Failed to parse Ux-string: {ux_string}")

plot_data = PlotData(dist_value)

# Display interactively
plot(plot_data)

# Or save to a file
plot(plot_data, path="output.png", save=True)
```

## Plotting from raw float samples

If you already have an array of float samples (e.g. from Monte Carlo simulation), use `DistributionalValue.from_samples()` to build a distributional value and then pass it to `PlotData`:

```python
import numpy as np
from signaloid.distributional.distributional import DistributionalValue
from signaloid.distributional_information_plotting.plot_histogram_dirac_deltas import PlotData
from signaloid.distributional_information_plotting.plot_wrapper import plot

samples = np.random.normal(0, 1, 10_000)

dist_value = DistributionalValue.from_samples(samples)
plot_data = PlotData(dist_value)
plot(plot_data, path="output.png", save=True)
```

Non-finite values (`NaN`, `-Inf`, `+Inf`) in the samples array are automatically separated and displayed in a dedicated special-values panel alongside the main histogram.

## Customizing the plot

The `plot()` function accepts several optional parameters:

```python
plot(
    plot_data,
    path="output.png",                    # Output file path
    save=True,                            # Save to file (False = show interactively)
    plot_expected_value_line=True,        # Vertical line at the mean
    x_lim=(-5, 5),                        # Custom x-axis limits
    y_lim=(0, 0.5),                       # Custom y-axis limits
    x_label="My Variable",                # Custom x-axis label
    x_tick_label_rotation=45,             # Rotate x-axis tick labels
    font_size=20,                         # Font size for labels
    matplotlib_rc_params_override={...},  # Custom matplotlib rc params
)
```

## Sampling from a Ux-string

Generate random samples from a Ux-encoded distribution:

```python
from signaloid.distributional_information_plotting.sample_generator import sample_generator

ux_string = "0.40007Ux0000000000000000013FD99AC12423C7C7000000013FD99AC12423C7C78000000000000000"

samples = sample_generator(ux_string, n_samples=1000)
```

Distributions that contain non-finite Dirac deltas (`NaN`, `-Inf`, `+Inf`) are handled via mixture sampling: each sample is drawn from either the finite part (via inverse CDF) or the non-finite part (categorically), proportional to their respective masses.

## CLI usage

These tools are also available via the `signaloid-uxdata-toolkit` command-line interface:

```bash
# Plot a distribution
signaloid-uxdata-toolkit plot --ux-data=0.40007Ux0000000000000000013FD99AC12423C7C7000000013FD99AC12423C7C78000000000000000

# Save plot to file
signaloid-uxdata-toolkit plot -o output.png --ux-data=0.40007Ux...

# Generate samples
signaloid-uxdata-toolkit sample --ux-data=0.40007Ux... --num-samples 100

# Save samples to file
signaloid-uxdata-toolkit sample -o samples.txt --ux-data=0.40007Ux... --num-samples 100
```

> **Note:** Use `=` syntax (`--ux-data=...`) for Ux-strings that start with `-`, otherwise the shell may interpret them as flags.
