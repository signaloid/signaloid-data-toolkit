#   Copyright (c) 2026, Signaloid.
#
#   Permission is hereby granted, free of charge, to any person obtaining a copy
#   of this software and associated documentation files (the "Software"), to
#   deal in the Software without restriction, including without limitation the
#   rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
#   sell copies of the Software, and to permit persons to whom the Software is
#   furnished to do so, subject to the following conditions:
#
#   The above copyright notice and this permission notice shall be included in
#   all copies or substantial portions of the Software.
#
#   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
#   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
#   DEALINGS IN THE SOFTWARE.

"""NumPy and SciPy distance utilities for DistributionalValue comparisons.

Modules:
- wasserstein: Wasserstein-1 / Wasserstein-2 (+ normalised variants),
  including UxHw wrappers.
- binned_wasserstein: CDF-based binned-Wasserstein-1 stack.
- ks_distance: Kolmogorov-Smirnov distance (sup norm between empirical
  CDFs). The hypothesis-test variants live elsewhere.
- scalar: scalar-distance utility.
"""

from signaloid.distributional_distance.binned_wasserstein import (
    binned_wasserstein_1_ux_string_wrapper,
    binned_wasserstein_1_uxhw_wrapper,
    wasserstein_1_between_distribution_and_samples,
)
from signaloid.distributional_distance.ks_distance import (
    kolmogorov_smirnov_distance_uxhw_wrapper,
)
from signaloid.distributional_distance.scalar import relative_error_uxhw_wrapper
from signaloid.distributional_distance.wasserstein import (
    normalized_wasserstein_1_uxhw_wrapper,
    normalized_wasserstein_2_uxhw_wrapper,
    normalized_wasserstein_p_uxhw_wrapper,
    wasserstein_1_distance,
    wasserstein_1_distance_with_weights,
    wasserstein_1_uxhw_wrapper,
    wasserstein_2_uxhw_wrapper,
    wasserstein_p_distance,
    wasserstein_p_uxhw_wrapper,
)

__all__ = [
    "binned_wasserstein_1_ux_string_wrapper",
    "binned_wasserstein_1_uxhw_wrapper",
    "kolmogorov_smirnov_distance_uxhw_wrapper",
    "normalized_wasserstein_1_uxhw_wrapper",
    "normalized_wasserstein_2_uxhw_wrapper",
    "normalized_wasserstein_p_uxhw_wrapper",
    "relative_error_uxhw_wrapper",
    "wasserstein_1_between_distribution_and_samples",
    "wasserstein_1_distance",
    "wasserstein_1_distance_with_weights",
    "wasserstein_1_uxhw_wrapper",
    "wasserstein_2_uxhw_wrapper",
    "wasserstein_p_distance",
    "wasserstein_p_uxhw_wrapper",
]
