# Distributional Distance

Distance utilities for comparing Signaloid `DistributionalValue` instances.


## Wasserstein-p distance

Compute the Wasserstein-p distance between two `DistributionalValue`s. `p` is an
integer ≥ 1. `p=1` and `p=2` (the canonical cases) have ergonomic shortcuts.

```python
import numpy as np
from signaloid.distributional.distributional import DistributionalValue
from signaloid.distributional_distance import (
    wasserstein_p_uxhw_wrapper,
    wasserstein_1_uxhw_wrapper,
    wasserstein_2_uxhw_wrapper,
)

rng = np.random.default_rng(seed=42)
dist_u = DistributionalValue.from_samples(rng.standard_normal(1000))
dist_v = DistributionalValue.from_samples(rng.standard_normal(10000))

# Generic, explicit p
print(wasserstein_p_uxhw_wrapper(dist_u, dist_v, p=1))
print(wasserstein_p_uxhw_wrapper(dist_u, dist_v, p=2))

# Ergonomic shortcuts for the canonical cases
print(wasserstein_1_uxhw_wrapper(dist_u, dist_v))
print(wasserstein_2_uxhw_wrapper(dist_u, dist_v))
```

## Normalized Wasserstein-p

Compute Wp after rescaling positions onto the ground-truth support `[0, 1]`. Same shortcut pattern: a generic `_p_` wrapper plus `_1_` / `_2_` shortcuts.

```python
from signaloid.distributional_distance import (
    normalized_wasserstein_p_uxhw_wrapper,
    normalized_wasserstein_1_uxhw_wrapper,
    normalized_wasserstein_2_uxhw_wrapper,
)

print(normalized_wasserstein_p_uxhw_wrapper(test_dist, ground_truth_dist, p=2))
print(normalized_wasserstein_1_uxhw_wrapper(test_dist, ground_truth_dist))
print(normalized_wasserstein_2_uxhw_wrapper(test_dist, ground_truth_dist))
```

## Kolmogorov-Smirnov distance

Sup-norm between two empirical step-function CDFs.

```python
from signaloid.distributional_distance import kolmogorov_smirnov_distance_uxhw_wrapper

print(kolmogorov_smirnov_distance_uxhw_wrapper(dist_u, dist_v))
```

## Scalar comparisons

For single-Dirac `DistributionalValue`s, three error variants are available. Each input must hold exactly one finite Dirac — non-scalar inputs are rejected to avoid silently using only `positions[0]`.

```python
from signaloid.distributional_distance import (
    relative_error_uxhw_wrapper,
    absolute_error_uxhw_wrapper,
    signed_error_uxhw_wrapper,
)

# |test − gt| / |gt| — scale-invariant
print(relative_error_uxhw_wrapper(test_dist, ground_truth_dist))

# |test − gt| — same units as inputs, non-negative
print(absolute_error_uxhw_wrapper(test_dist, ground_truth_dist))

# test − gt — same units, preserves sign
print(signed_error_uxhw_wrapper(test_dist, ground_truth_dist))
```

## Binned Wasserstein-1

Asymmetric distance: bins the first argument and treats the second as raw weighted samples. Use the `_uxhw_` form for two `DistributionalValue`s, the `_ux_string_` form to skip parsing, or the array entry point if you already have histogram data.

```python
from signaloid.distributional_distance import (
    binned_wasserstein_1_uxhw_wrapper,
    binned_wasserstein_1_ux_string_wrapper,
    wasserstein_1_between_distribution_and_samples,
)

# From DistributionalValues
print(binned_wasserstein_1_uxhw_wrapper(binned_dist, ground_truth_dist))

# From Ux-strings
print(binned_wasserstein_1_ux_string_wrapper(distribution_ux, ground_truth_ux))

# From pre-binned histogram data
distance = wasserstein_1_between_distribution_and_samples(
    bin_boundaries=[0.0, 1.0, 2.0, 3.0],
    bin_heights=[0.25, 0.5, 0.25],
    bin_widths=[1.0, 1.0, 1.0],
    sample_positions=[0.3, 1.2, 1.8, 2.4],
)
```

## Lower-level array entry points

For callers that have pre-sorted sample arrays and want to skip the `DistributionalValue` wrapping:

```python
from signaloid.distributional_distance import (
    wasserstein_1_distance,
    wasserstein_1_distance_with_weights,
)

# W1 between two equally-weighted, pre-sorted sample sets
# (drop-in for scipy.stats.wasserstein_distance)
distance = wasserstein_1_distance(u_values, v_values, all_values)

# W1 between an unweighted u and a weighted v
distance = wasserstein_1_distance_with_weights(
    u_values, v_values, v_cum_weights, all_values
)
```

## CLI usage

`wasserstein.py`, `binned_wasserstein.py`, and `scalar.py` each expose a `python -m …` entry point that parses two Ux-strings, computes the distance, and exits non-zero if the result exceeds the supplied tolerance.

```bash
# Wasserstein-p (default p=1)
python -m signaloid.distributional_distance.wasserstein \
    <dist_u_ux> <dist_v_ux> <tolerance> [--p N]

# Binned Wasserstein-1 (asymmetric — first arg is binned)
python -m signaloid.distributional_distance.binned_wasserstein \
    <distribution_ux> <ground_truth_ux> <tolerance>

# Scalar error (single-Dirac inputs; default metric: relative)
python -m signaloid.distributional_distance.scalar \
    <test_dist_ux> <ground_truth_dist_ux> <tolerance> \
    [--metric relative|absolute|signed]
```

> **Note:** Ux-strings with a leading `-` (e.g. `-0.0Ux…`) get parsed by `argparse` as flags. Place any options (`--p N`, `--metric M`) *before* the positional arguments, or use the `--` separator before the leading-minus Ux-string.
