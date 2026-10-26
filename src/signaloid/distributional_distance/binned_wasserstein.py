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

import argparse
import sys
import warnings

import numpy as np

from signaloid.distributional_information_plotting.plot_histogram_dirac_deltas import (
    PlotData,
)
from .wasserstein import (
    wasserstein_1_uxhw_wrapper,
)
from signaloid.distributional.distributional import DistributionalValue
from signaloid.distributional_distance._validators import (
    _require_distributional_pair,
)


def normalized_uxhw_cdf_heights(
    bin_widths: np.ndarray, bin_heights: np.ndarray
) -> np.ndarray:
    """CDF heights for the binned representation of a UxHw distribution.

    Returns the cumulative bin areas (heights * widths) after
    normalising the total mass to 1 if it has drifted from 1 by more
    than 1e-12.

    Args:
        bin_widths: Width of each bin.
        bin_heights: PDF height of each bin.

    Returns:
        Cumulative bin areas (length = len(bin_widths)).
    """
    single_bin_areas: np.ndarray = bin_heights * bin_widths

    uxhw_cdf_total_mass: float = np.sum(single_bin_areas)
    if uxhw_cdf_total_mass <= 0.0:
        raise ValueError(
            "Binned UxHw distribution has non-positive total mass "
            f"({uxhw_cdf_total_mass}); cannot normalise."
        )
    if abs(uxhw_cdf_total_mass - 1.0) >= 1e-12:
        warnings.warn(
            "Binned uxhw distribution was not normalized. Normalizing cdf.",
            stacklevel=2,
        )
        single_bin_areas = single_bin_areas / uxhw_cdf_total_mass

    return np.cumsum(single_bin_areas)


def normalized_sample_cdf_heights(
    sample_positions: np.ndarray, sample_weights: np.ndarray
) -> np.ndarray:
    """Cumulative heights of the empirical (step-wise) CDF.

    Normalises the total weight to 1 if it has drifted from 1 by more
    than 1e-12.

    Args:
        sample_positions: Sorted array of sample positions. Currently
            unused — kept for symmetry with `normalized_uxhw_cdf_heights`
            and to document the calling contract (positions and weights
            must already be sorted by position so that `cumsum(weights)`
            gives the empirical CDF heights).
        sample_weights: Weight for each sample (must be aligned with
            `sample_positions`).

    Returns:
        Cumulative weights, length = len(sample_weights).
    """
    single_step_areas: np.ndarray = sample_weights

    sample_cdf_total_mass: float = np.sum(single_step_areas)
    if sample_cdf_total_mass <= 0.0:
        raise ValueError(
            "Empirical sample distribution has non-positive total mass "
            f"({sample_cdf_total_mass}); cannot normalise."
        )
    if abs(sample_cdf_total_mass - 1.0) >= 1e-12:
        warnings.warn(
            "Sample empirical distribution was not normalized. Normalizing cdf.",
            stacklevel=2,
        )
        single_step_areas = single_step_areas / sample_cdf_total_mass

    return np.cumsum(single_step_areas)


def wasserstein_1_core(
    uxhw_cdf_boundaries: np.ndarray,
    uxhw_cdf_widths: np.ndarray,
    uxhw_cdf_heights: np.ndarray,
    sample_cdf_positions: np.ndarray,
    sample_cdf_heights: np.ndarray,
) -> float:
    """Wasserstein-1 distance between a binned UxHw CDF and an empirical
    sample CDF.

    Merges the bin boundaries and sample positions into one sorted array,
    evaluates both CDFs at every merged position and integrates the
    absolute difference piecewise. Between two consecutive merged
    positions the UxHw CDF is linear and the empirical CDF is constant,
    so their difference is linear there and its absolute value
    integrates exactly as a trapezoid, or as two triangles where the
    difference changes sign.

    Caller-validates contract: this is the inner algorithmic core and
    does not validate its inputs. The only intended caller is
    `wasserstein_1_between_distribution_and_samples`, which validates
    upstream via `_validate_binned_lengths` and
    `_validate_binned_semantics`. Callers must ensure shapes are
    length-consistent and values are finite.

    Args:
        uxhw_cdf_boundaries: Sorted bin boundaries for the UxHw CDF
            (length N+1).
        uxhw_cdf_widths: Bin widths (length N).
        uxhw_cdf_heights: Cumulative CDF heights at each boundary
            (length N+1, starting at 0).
        sample_cdf_positions: Sorted positions of the empirical samples
            (length M).
        sample_cdf_heights: Cumulative empirical-CDF heights at each
            sample position (length M+1, starting at 0).

    Returns:
        Wasserstein-1 distance between the two CDFs.
    """
    num_boundaries = len(uxhw_cdf_boundaries)
    num_bins = num_boundaries - 1

    # Merge the boundaries and sample positions, tagging each entry with
    # its origin: 0 for a bin boundary, 1 for a sample. Sort with `types`
    # as the secondary key so that at tied positions (boundary coincident
    # with a sample) the boundary comes first. The index bookkeeping
    # below relies on this ordering. `np.argsort(positions)` alone would
    # not be stable on all numpy versions/platforms.
    positions = np.concatenate((uxhw_cdf_boundaries, sample_cdf_positions))
    types = np.concatenate(
        (
            np.zeros(num_boundaries, dtype=np.int8),
            np.ones(len(sample_cdf_positions), dtype=np.int8),
        )
    )
    order = np.lexsort((types, positions))
    positions = positions[order]
    is_sample = types[order] == 1

    # `uxhw_cdf_idx[j]` is the linear segment of the UxHw CDF that
    # positions[j] lives in: -1 before the first boundary, num_bins at
    # and after the last. `sample_cdf_idx[j]` is the number of samples
    # at or before positions[j], i.e. the constant segment of the
    # empirical CDF.
    uxhw_cdf_idx = np.cumsum(~is_sample) - 1
    sample_cdf_idx = np.cumsum(is_sample)

    # Empirical CDF at each merged position. `sample_cdf_heights[0]` is
    # 0, so points before the first sample evaluate to 0.
    sample_cdf_vals = sample_cdf_heights[sample_cdf_idx]

    # UxHw CDF at each merged position: linear interpolation within the
    # segment, clamped to 0 before the first boundary and to 1 at and
    # after the last. The segment index is clipped so the interpolation
    # is evaluated on a real segment everywhere, then the clamped ends
    # are overwritten.
    segment = np.clip(uxhw_cdf_idx, 0, num_bins - 1)
    slopes = np.diff(uxhw_cdf_heights) / uxhw_cdf_widths
    uxhw_cdf_vals = uxhw_cdf_heights[segment] + slopes[segment] * (
        positions - uxhw_cdf_boundaries[segment]
    )
    uxhw_cdf_vals[uxhw_cdf_idx < 0] = 0.0
    uxhw_cdf_vals[uxhw_cdf_idx >= num_bins] = 1.0

    # Signed gap between the CDFs at the left and right end of every
    # interval between consecutive merged positions. The empirical CDF is
    # right-continuous, so inside the interval it keeps the value it has
    # at the left end: both gaps use `sample_cdf_vals[:-1]`.
    delta_left = uxhw_cdf_vals[:-1] - sample_cdf_vals[:-1]
    delta_right = uxhw_cdf_vals[1:] - sample_cdf_vals[:-1]
    left_pos = positions[:-1]
    right_pos = positions[1:]

    # Each interval contributes a trapezoid of the absolute gap. Where
    # the gap changes sign it crosses zero at `mid`, and the interval
    # contributes two triangles instead.
    interval_areas = (
        0.5 * (np.abs(delta_left) + np.abs(delta_right)) * (right_pos - left_pos)
    )
    crossing = delta_left * delta_right < 0.0
    y0 = delta_left[crossing]
    y1 = delta_right[crossing]
    x0 = left_pos[crossing]
    x1 = right_pos[crossing]
    mid = x0 + y0 / (y0 - y1) * (x1 - x0)
    interval_areas[crossing] = 0.5 * np.abs(y0) * (mid - x0) + 0.5 * np.abs(y1) * (
        x1 - mid
    )

    return float(np.sum(interval_areas))


def _validate_binned_lengths(
    bin_boundaries: np.ndarray,
    bin_heights: np.ndarray,
    bin_widths: np.ndarray,
    sample_positions: np.ndarray,
    sample_weights: np.ndarray | None,
) -> None:
    """Structural length / non-empty checks for the binned wrapper."""
    if len(bin_widths) == 0:
        raise ValueError("bin_widths must be non-empty.")
    if len(bin_heights) != len(bin_widths):
        raise ValueError("bin_widths and bin_heights must have the same length.")
    if len(bin_boundaries) != len(bin_widths) + 1:
        raise ValueError(
            "bin_boundaries must contain exactly one more entry than bin_widths."
        )
    if len(sample_positions) == 0:
        raise ValueError("sample_positions must not be empty.")
    if sample_weights is not None and len(sample_weights) != len(sample_positions):
        raise ValueError(
            "sample_positions and sample_weights must have the same length."
        )


def _validate_binned_semantics(
    bin_boundaries: np.ndarray,
    bin_heights: np.ndarray,
    bin_widths: np.ndarray,
    sample_positions: np.ndarray,
    sample_weights: np.ndarray | None,
) -> None:
    """Finite + non-negativity / positivity checks on the raw inputs.

    Checks raw inputs (not post-cumsum CDF arrays). The cumulative
    arrays can hide negative entries — e.g. weights=[0.5, -0.1, 0.6]
    cumsums to [0.5, 0.4, 1.0], all non-negative, so a downstream
    `np.any(cdf < 0)` check would miss the negative.
    """
    bin_boundaries_arr = np.asarray(bin_boundaries, dtype=np.float64)
    bin_heights_arr = np.asarray(bin_heights, dtype=np.float64)
    bin_widths_arr = np.asarray(bin_widths, dtype=np.float64)
    finite_checks = (
        ("bin_boundaries", bin_boundaries_arr),
        ("bin_heights", bin_heights_arr),
        ("bin_widths", bin_widths_arr),
        ("sample_positions", np.asarray(sample_positions, dtype=np.float64)),
    )
    for name, arr in finite_checks:
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} must contain only finite values.")

    if np.any(bin_widths_arr <= 0.0):
        raise ValueError("All bin_widths must be strictly positive.")
    if np.any(bin_heights_arr < 0.0) or not np.any(bin_heights_arr > 0.0):
        raise ValueError(
            "bin_heights must be non-negative with at least one "
            "strictly positive entry."
        )

    # `wasserstein_1_core` assumes bin_boundaries is strictly
    # increasing and that bin_widths[i] == boundaries[i+1] - boundaries[i].
    # Non-monotonic boundaries would produce negative segment lengths
    # and silently-wrong distances. width/boundary disagreement breaks
    # the CDF-merge integration.
    boundary_gaps = np.diff(bin_boundaries_arr)
    if np.any(boundary_gaps <= 0.0):
        raise ValueError("bin_boundaries must be strictly increasing.")
    if not np.allclose(boundary_gaps, bin_widths_arr, rtol=1e-9, atol=1e-12):
        raise ValueError(
            "bin_widths must match the implied widths "
            "np.diff(bin_boundaries) within tolerance."
        )

    if sample_weights is None:
        return
    sample_weights_arr = np.asarray(sample_weights, dtype=np.float64)
    if not np.all(np.isfinite(sample_weights_arr)):
        raise ValueError("sample_weights must contain only finite values.")
    if np.any(sample_weights_arr < 0.0) or not np.any(sample_weights_arr > 0.0):
        raise ValueError(
            "sample_weights must be non-negative with at least one "
            "strictly positive entry."
        )


def _validate_binned_inputs(
    bin_boundaries: np.ndarray,
    bin_heights: np.ndarray,
    bin_widths: np.ndarray,
    sample_positions: np.ndarray,
    sample_weights: np.ndarray | None,
) -> None:
    """Full validation for `wasserstein_1_between_distribution_and_samples`."""
    _validate_binned_lengths(
        bin_boundaries, bin_heights, bin_widths, sample_positions, sample_weights
    )
    _validate_binned_semantics(
        bin_boundaries, bin_heights, bin_widths, sample_positions, sample_weights
    )


def wasserstein_1_between_distribution_and_samples(
    bin_boundaries: list[float] | np.ndarray,
    bin_heights: list[float] | np.ndarray,
    bin_widths: list[float] | np.ndarray,
    sample_positions: list[float] | np.ndarray,
    sample_weights: list[float] | np.ndarray | None = None,
) -> float:
    """Wasserstein-1 distance between a binned distribution and samples.

    Wraps the core CDF-merge algorithm: converts the inputs to arrays,
    validates them, builds normalised cumulative CDFs for both sides,
    sorts the sample side by position so the empirical CDF is
    well-defined, and delegates to `wasserstein_1_core`. Inputs may be
    lists or numpy arrays.

    Args:
        bin_boundaries: Distribution bin boundaries, length N+1, sorted.
        bin_heights: PDF heights of each bin (length N, non-negative).
        bin_widths: Width of each bin (length N, strictly positive).
        sample_positions: Sample positions, any order.
        sample_weights: Optional sample weights aligned with
            `sample_positions`. Defaults to uniform `1 / len(samples)`.

    Returns:
        Wasserstein-1 distance between the binned distribution and the
        empirical sample distribution.

    Raises:
        ValueError: On any structural mismatch (lengths, empty inputs),
            non-finite values, non-positive widths, or zero total mass
            on either side. See `_validate_binned_lengths` and
            `_validate_binned_semantics` for the full check list.
    """
    # Convert to np arrays once. Validation and the core both work on
    # these, so list inputs are converted a single time and array inputs
    # are used as they are.
    uxhw_cdf_boundaries: np.ndarray = np.asarray(bin_boundaries, dtype=np.float64)
    uxhw_bin_heights: np.ndarray = np.asarray(bin_heights, dtype=np.float64)
    uxhw_cdf_widths: np.ndarray = np.asarray(bin_widths, dtype=np.float64)
    sample_cdf_positions: np.ndarray = np.asarray(sample_positions, dtype=np.float64)
    sample_cdf_weights: np.ndarray | None = (
        None if sample_weights is None else np.asarray(sample_weights, dtype=np.float64)
    )

    _validate_binned_inputs(
        uxhw_cdf_boundaries,
        uxhw_bin_heights,
        uxhw_cdf_widths,
        sample_cdf_positions,
        sample_cdf_weights,
    )

    if sample_cdf_weights is None:
        num_samples = len(sample_cdf_positions)
        sample_cdf_weights = np.full(num_samples, 1.0 / num_samples)

    uxhw_cdf_heights: np.ndarray = normalized_uxhw_cdf_heights(
        uxhw_cdf_widths, uxhw_bin_heights
    )
    uxhw_cdf_heights = np.append([0.0], uxhw_cdf_heights)

    # The empirical CDF is cumulative-in-position-order, so positions
    # (and their matching weights) must be sorted by position before
    # cumsum. Callers may pass any ordering. Sort here.
    sample_sort_order: np.ndarray = np.argsort(sample_cdf_positions)
    sample_cdf_positions = sample_cdf_positions[sample_sort_order]
    sample_cdf_weights = sample_cdf_weights[sample_sort_order]
    sample_cdf_heights: np.ndarray = normalized_sample_cdf_heights(
        sample_cdf_positions, sample_cdf_weights
    )
    sample_cdf_heights = np.append([0.0], sample_cdf_heights)

    return wasserstein_1_core(
        uxhw_cdf_boundaries,
        uxhw_cdf_widths,
        uxhw_cdf_heights,
        sample_cdf_positions,
        sample_cdf_heights,
    )


def binned_wasserstein_1_uxhw_wrapper(
    binned_dist: DistributionalValue,
    ground_truth_dist: DistributionalValue,
) -> float:
    """Binned Wasserstein-1 distance between a UxHw distribution and a
    ground-truth distribution.

    Bins the UxHw distribution via `PlotData` (PDF binning, then TTR
    rebinning), then compares the resulting bins to the ground truth's
    weighted samples. Short-circuits to the non-binned W1 wrapper when
    `binned_dist` collapses to a single Dirac delta (binning is ill-defined
    for a single point).

    Args:
        binned_dist: Distribution under test. Sorted in place as a side effect.
        ground_truth_dist: Reference distribution treated as weighted samples
            (positions + masses).

    Returns:
        Binned Wasserstein-1 distance.

    Raises:
        ValueError: If either argument is not a `DistributionalValue`,
            or if `PlotData` fails to resolve a `plotting_ttr_order`
            for the input UxHw distribution.
    """
    _require_distributional_pair(binned_dist, ground_truth_dist)

    # Binning is ill-defined for distributions with ≤1 *finite* Dirac
    # delta — delegate to the non-binned W1 wrapper. Checking
    # `finite_dirac_deltas` (rather than `positions`) is important
    # because special-value Diracs (NaN / ±Inf) with non-zero mass
    # appear in `positions` after sorting but cannot be binned.
    # `finite_dirac_deltas` internally calls `binned_dist.sort()` as a side
    # effect (mutates the caller's distribution in place).
    if len(binned_dist.finite_dirac_deltas) <= 1:
        return wasserstein_1_uxhw_wrapper(dist_u=binned_dist, dist_v=ground_truth_dist)

    # Use PlotData to compute the binning (binned_dist is already sorted via
    # the finite_dirac_deltas access above).
    plot_interface = PlotData(binned_dist)

    if plot_interface.plotting_ttr_order is None:
        raise ValueError(
            "PlotData failed to resolve plotting_ttr_order for the input "
            "UxHw distribution (got None); cannot compute binned "
            "Wasserstein-1."
        )

    # `PlotData` has already selected the binning that suits this input:
    # the TTR binning route when the value is a full valid TTR, and the
    # non-uniform binning (one bin per Dirac delta, boundaries following the
    # Dirac spacing) when it is not. Recomputing the TTR route here
    # unconditionally would bin a non-TTR value as though it were a TTR,
    # which is exactly what `_construct_plot_data` avoids, and would leave
    # the measured distance disagreeing with the plotted distribution.
    return wasserstein_1_between_distribution_and_samples(
        bin_boundaries=plot_interface.positions,
        bin_heights=plot_interface.masses,
        bin_widths=plot_interface.widths,
        sample_positions=ground_truth_dist.positions,
        sample_weights=ground_truth_dist.masses,
    )


def binned_wasserstein_1_ux_string_wrapper(
    distribution_ux: str, ground_truth_ux: str
) -> float:
    """Binned Wasserstein-1 distance from Ux-string inputs.

    Parses both arguments into `DistributionalValue` instances and
    delegates to `binned_wasserstein_1_uxhw_wrapper`.

    Args:
        distribution_ux: Ux-string for the distribution under test.
        ground_truth_ux: Ux-string for the reference distribution.

    Returns:
        Binned Wasserstein-1 distance.

    Raises:
        ValueError: If either Ux-string fails to parse, or if the
            downstream binned-W1 computation rejects its inputs.
    """
    distribution = DistributionalValue.parse(distribution_ux)
    if distribution is None:
        raise ValueError(
            f"Could not parse distribution from ux string {distribution_ux}"
        )

    ground_truth = DistributionalValue.parse(ground_truth_ux)
    if ground_truth is None:
        raise ValueError(
            f"Could not parse ground truth from ux string {ground_truth_ux}"
        )

    return binned_wasserstein_1_uxhw_wrapper(distribution, ground_truth)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="python -m signaloid.distributional_distance.binned_wasserstein",
        description=(
            "Compute binned Wasserstein-1 distance between two ux strings. "
            "Asymmetric: the first argument is binned, the second is "
            "treated as raw weighted samples."
        ),
    )
    parser.add_argument("distribution_ux", help="ux string to be binned")
    parser.add_argument("ground_truth_ux", help="ux string treated as weighted samples")
    parser.add_argument("tolerance", type=float)
    args = parser.parse_args()

    distance: float = binned_wasserstein_1_ux_string_wrapper(
        args.distribution_ux, args.ground_truth_ux
    )
    if distance <= args.tolerance:
        print(
            f"[SUCCESS] Binned Wasserstein-1 distance within "
            f"{args.tolerance} tolerance. Distance: {distance}."
        )
    else:
        print(
            f"[FAILURE] Binned Wasserstein-1 distance NOT within "
            f"{args.tolerance} tolerance. Distance: {distance}."
        )
    sys.exit(0 if distance <= args.tolerance else 1)
