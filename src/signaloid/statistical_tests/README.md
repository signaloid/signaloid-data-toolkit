# Statistical Tests

Hypothesis tests for comparing Signaloid `DistributionalValue` instances.


## Bootstrapped Kolmogorov-Smirnov test

Bootstrapped KS goodness-of-fit test. Draws `bootstrap_sample_size` samples from
the discrete representation, runs a one-sample KS test against the true CDF,
repeats `number_of_bootstraps` times, and combines the per-bootstrap p-values
with Fisher's method. Returns `(accept_hypothesis, combined_p_value)`, where the
hypothesis is accepted when the combined p-value is at least
`significance_level`.

The `_wrapper` form takes two `DistributionalValue`s and uses the reference's
empirical `.cdf`. The core form takes any callable CDF (e.g. an analytic
`scipy.stats` CDF). Pass a seeded `rng` for reproducibility.

```python
import numpy as np
from scipy.stats import norm
from signaloid.distributional.distributional import DistributionalValue
from signaloid.statistical_tests import (
    bootstrapped_kolmogorov_smirnov,
    bootstrapped_kolmogorov_smirnov_wrapper,
)

rng = np.random.default_rng(seed=42)
discrete = DistributionalValue.from_samples(rng.standard_normal(1000))
reference = DistributionalValue.from_samples(rng.standard_normal(10000))

# Against another DistributionalValue's empirical CDF
accept, p_value = bootstrapped_kolmogorov_smirnov_wrapper(
    discrete,
    reference,
    bootstrap_sample_size=128,
    number_of_bootstraps=50,
    significance_level=0.05,
    rng=rng,
)

# Against an analytic CDF callable
accept, p_value = bootstrapped_kolmogorov_smirnov(
    discrete,
    norm.cdf,
    bootstrap_sample_size=128,
    number_of_bootstraps=50,
    significance_level=0.05,
    rng=rng,
)
```

## One-sample Kolmogorov-Smirnov test

Test whether an observed `sample` is consistent with a reference `DistributionalValue`'s empirical CDF at the given significance level. Accepts a `list[float]` or an `np.ndarray`. Returns `True` when the p-value is at least `significance_level`.

```python
from signaloid.statistical_tests import kolmogorov_smirnov_wrapper

accept = kolmogorov_smirnov_wrapper(
    sample=[0.1, 0.3, 0.5, 0.7, 0.9],
    true_distribution=reference,
    significance_level=0.05,
)
```

## CLI usage

`ks_hypothesis.py` exposes a `python -m …` entry point that parses two Ux-strings, runs a KS hypothesis test at the given significance level, prints the verdict, and exits non-zero when the null hypothesis is rejected.

```bash
# Bootstrapped KS test (default)
python -m signaloid.statistical_tests.ks_hypothesis \
    <dist_ux> <reference_ux> <significance_level> \
    [--bootstrap-sample-size N] [--number-of-bootstraps M] [--seed S]

# One-sample KS test (uses the first distribution's support positions
# as the observed sample)
python -m signaloid.statistical_tests.ks_hypothesis \
    <dist_ux> <reference_ux> <significance_level> --test one-sample
```

> **Note:** Ux-strings with a leading `-` (e.g. `-0.0Ux…`) get parsed by `argparse` as flags. Place any options (`--test`, `--seed`, …) *before* the positional arguments, or use the `--` separator before the leading-minus Ux-string.

> **Note:** The Kolmogorov-Smirnov *distance* (sup-norm between two empirical CDFs) is a distance metric rather than a hypothesis test, and lives in `signaloid.distributional_distance.ks_distance`.
