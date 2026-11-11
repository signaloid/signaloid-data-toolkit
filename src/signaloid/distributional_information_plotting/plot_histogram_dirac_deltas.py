#   Copyright (c) 2021, Signaloid.
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


import os
import sys
import math

if sys.implementation.name != "circuitpython":
    # Use numpy for accelerated computing
    import numpy as np
else:
    # Use the extended version of ulab's numpy when running on CircuitPython
    from signaloid.circuitpython.extended_ulab_numpy import np  # type: ignore[no-redef]

from signaloid.distributional.dirac_delta import DiracDelta
from signaloid.distributional.distributional import (
    TTR_NATIVE_UR_TYPES,
    DistributionalValue,
)


class PlotData:
    def __init__(
        self,
        dist: DistributionalValue,
        plotting_resolution: int | None = None,
    ) -> None:
        if dist.mean is None or dist.UR_order == 0:
            raise ValueError("Failed to load data")

        self.dist = dist
        self.plotting_resolution: int | None = plotting_resolution
        self.plotting_ttr_order: int | None = None

        self._positions: np.ndarray = np.array([], dtype=np.float64)
        self._masses: np.ndarray = np.array([], dtype=np.float64)
        self._widths: np.ndarray = np.array([], dtype=np.float64)
        self._max_value: float | None = None

        self._construct_plot_data()

    # Maximum number of bins for plotting
    MAX_BINS: int = 1024

    # Closest an internal bin boundary of a non-uniform binning may sit to the Dirac
    # delta it is placed against, as a fraction of the interval between that Dirac
    # delta and its neighbour. Only keeps the extremal bins from collapsing to zero
    # width, so it is deliberately tiny: see `create_non_uniform_binning`.
    MIN_INTERIOR_WEIGHT: float = 1e-6

    # Fewest Dirac deltas for which the extremal bins follow the detected shape of the
    # tail (see `_tail_shape`). With fewer, the second threshold the shape is read from
    # lies too close to the middle of the distribution to say anything about the tail.
    MIN_DIRAC_DELTAS_FOR_TAIL_DETECTION: int = 8

    # Width of the outer bin of an extremal Dirac delta, as a multiple of the width of
    # its inner bin, against the shape xi of a generalized Pareto tail. Each entry is a
    # pair (xi, multiple), sorted by xi. Each multiple is the one that minimises the
    # Wasserstein-1 distance between the two bins and the tail they hold, found
    # numerically. A shape of -1 is a flat density, where the outer bin is as wide as
    # the inner one. A shape of 0 is an exponential tail. A positive shape is a
    # polynomial tail with index 1 / xi, e.g., a Student's t with 1 / xi degrees of
    # freedom. The entries are sample points, not the only shapes handled.
    # `_outer_width_ratio` interpolates between the two nearest entries, linearly in
    # the logarithm of the multiple. Shapes below the first entry use the first
    # multiple, and shapes above the last entry use the last multiple.
    TAIL_SHAPE_TO_OUTER_WIDTH_RATIO: tuple[tuple[float, float], ...] = (
        (-3.0, 0.44733198080314746),
        (-2.0, 0.6215164160019556),
        (-1.5, 0.768398464210621),
        (-1.0, 1.0),
        (-0.75, 1.1730897237926152),
        (-0.5, 1.4130182735448464),
        (-0.25, 1.7653697505547048),
        (0.0, 2.32677724368277),
        (0.1, 2.6546890960368854),
        (0.2, 3.0792850281864066),
        (0.3333333333333333, 3.8830289986968785),
        (0.4, 4.4467760808977825),
        (0.5, 5.634715822694777),
        (0.6, 7.561898433223356),
        (0.6666666666666666, 9.65402816971081),
        (0.75, 14.305687112969494),
        (0.8, 19.531687030587193),
    )

    @classmethod
    def from_samples(
        cls,
        samples: np.ndarray | list[float],
        plotting_resolution: int | None = None,
    ) -> "PlotData":
        """
        Construct a PlotData from an array of float samples.

        Creates a `DistributionalValue` from the samples (each sample
        becomes an equal-weight Dirac delta) and then builds the plot
        data through the standard TTR binning pipeline.

        Args:
            samples: 1-D array of float samples (may contain NaN/Inf).
            plotting_resolution: Number of bins for the plot (must be a
                power of 2). If `None`, automatically determined from
                the data.

        Returns:
            A PlotData instance ready to be passed to plot().
        """
        dist = DistributionalValue.from_samples(samples)
        return cls(dist, plotting_resolution=plotting_resolution)

    @classmethod
    def from_data_out(
        cls,
        path: str | os.PathLike[str],
        plotting_resolution: int | None = None,
    ) -> "PlotData":
        """
        Construct a PlotData from a Monte Carlo `data.out` file.

        Reads the samples with `DistributionalValue.from_data_out()`, which
        documents the file format, and then builds the plot data as
        `from_samples()` does.

        Args:
            path: Path to the `data.out` file.
            plotting_resolution: Number of bins for the plot (must be a
                power of 2). If `None`, automatically determined from
                the data.

        Returns:
            A PlotData instance ready to be passed to plot().

        Raises:
            OSError: If the file cannot be read.
            ValueError: If the file does not hold valid samples.
        """
        dist = DistributionalValue.from_data_out(path)
        return cls(dist, plotting_resolution=plotting_resolution)

    @property
    def positions(self) -> np.ndarray:
        """The boundary positions list.

        :return: The boundary positions list.
        :rtype: np.ndarray
        """
        return self._positions

    @positions.setter
    def positions(self, positions: np.ndarray) -> None:
        """Sets the boundary positions list, resetting the widths to avoid faulty
        values.

        :param positions: The boundary positions list to use
        :type positions: np.ndarray
        """
        self._positions = positions
        self._widths = np.array([], dtype=np.float64)

    @property
    def masses(self) -> np.ndarray:
        """The bin heights list.

        :return: The bin heights list.
        :rtype: np.ndarray
        """
        return self._masses

    @masses.setter
    def masses(self, masses: np.ndarray) -> None:
        """Sets the bin heights list, resetting the max value to avoid faulty value.

        :param masses: The bin heights list to use.
        :type masses: np.ndarray
        """
        self._masses = masses
        self._max_value = None

    @property
    def min_range(self) -> float:
        """The minimum position.

        :return: The minimum position.
        :rtype: float
        """
        if len(self.positions) == 1:
            return float(self.positions[0] - 0.5)
        return float(self.positions[0])

    @property
    def max_range(self) -> float:
        """The maximum position.

        :return: The maximum position.
        :rtype: float
        """
        if len(self.positions) == 1:
            return float(self.positions[-1] + 0.5)
        return float(self.positions[-1])

    @property
    def total_range(self) -> float:
        """The total range of positions, i.e. the width between the minimum and
        maximum position.

        :return: The total range of positions.
        :rtype: float
        """
        if len(self.positions) == 1:
            return 1.0
        return float(self.positions[-1] - self.positions[0])

    @property
    def max_value(self) -> float:
        """The maximum bin height.

        :return: The maximum bin height.
        :rtype: float
        """
        if self._max_value is None:
            self._max_value = float(max(self._masses))

        return self._max_value

    @property
    def widths(self) -> np.ndarray:
        """The widths list between each pair of positions.

        :return: The widths list.
        :rtype: np.ndarray
        """
        if not self._widths.size > 0:
            self._widths = self.positions[1:] - self.positions[:-1]

        return self._widths

    @staticmethod
    def _determine_boundary_positions(
        positions: np.ndarray,
        masses: np.ndarray,
        exponent: int,
        use_ttr_binning: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        If `use_ttr_binning` is true:
        Determines the internal boundary positions (and probabilities) using the
        TTR binning method.
        If `use_ttr_binning` is false:
        Determines the internal boundary positions (and probabilities) by only
        looking at the adjacent Dirac deltas.

        Args:
            positions: The finite and sorted positions of the input Dirac deltas.
                At least two are required, since the extremal boundaries are
                placed relative to an adjacent Dirac delta.
            masses: The masses of the input Dirac deltas.
            exponent: The TTR order, i.e., the base-2 logarithm of the number of
                Dirac deltas in the TTR. The number of bins in the output binning
                is twice the number of Dirac deltas in the TTR.
            use_ttr_binning: Flag specifying whether to use the TTR binning method.
        Returns:
            (boundary_positions, boundary_probabilities): The internal boundary positions
                and boundary probabilities that are intermediaries to get a binning.
                The odd entries are the positions and masses of the Dirac deltas.
        Raises:
            ValueError: When given fewer than two Dirac deltas.
        """

        number_of_finite_dirac_deltas = len(positions)
        if number_of_finite_dirac_deltas < 2:
            raise ValueError(
                "plot_histogram_dirac_deltas: _determine_boundary_positions requires "
                f"at least two Dirac deltas, got {number_of_finite_dirac_deltas}"
            )

        number_of_boundaries = 2 * number_of_finite_dirac_deltas + 1
        boundary_positions = np.full(number_of_boundaries, np.nan)
        boundary_probabilities = np.full(number_of_boundaries, np.nan)
        boundary_positions[1::2] = positions
        boundary_probabilities[1::2] = masses

        if not use_ttr_binning:
            # Determine the 'NaN'-valued boundary points from adjacent Dirac deltas.
            # Even indices (2, 4, ...) are the NaN-valued boundaries between
            # odd-indexed Dirac delta positions.
            even = slice(2, number_of_boundaries - 1, 2)
            left_prob = boundary_probabilities[1 : number_of_boundaries - 2 : 2]
            right_prob = boundary_probabilities[3:number_of_boundaries:2]
            left_pos = boundary_positions[1 : number_of_boundaries - 2 : 2]
            right_pos = boundary_positions[3:number_of_boundaries:2]
            nan_mask = np.isnan(boundary_positions[even])
            weighted_avg = (left_prob * left_pos + right_prob * right_pos) / (
                left_prob + right_prob
            )
            boundary_positions[even] = np.where(
                nan_mask, weighted_avg, boundary_positions[even]
            )

            return (boundary_positions, boundary_probabilities)

        # First handle internal boundary positions.
        # Strided slices select the boundaries, because the ulab NumPy of
        # CircuitPython does not support integer-array indexing.
        for n in range(exponent):
            step = 2**n
            start = 2 ** (n + 1)
            stop = number_of_boundaries - 1
            stride = 2 ** (n + 2)
            if start >= stop:
                continue
            indices = slice(start, stop, stride)
            left = slice(start - step, stop - step, stride)
            right = slice(start + step, stop + step, stride)
            boundary_probabilities[indices] = (
                boundary_probabilities[left] + boundary_probabilities[right]
            )
            boundary_positions[indices] = (
                boundary_probabilities[left] * boundary_positions[left]
                + boundary_probabilities[right] * boundary_positions[right]
            ) / boundary_probabilities[indices]

        # Above process might not produce a strictly increasing sequence of
        # positions if not a valid TTR, and it will leave 'NaN'-valued
        # boundary points if the number of Dirac deltas is not a power of 2.
        # Handle both cases by sweeping over the boundary positions.
        # Note: this fixup must remain sequential because each corrected
        # position feeds into the next check.
        for i in range(2, number_of_boundaries - 1, 2):
            if (
                np.isnan(boundary_positions[i])
                or boundary_positions[i] <= boundary_positions[i - 1]
                or boundary_positions[i] >= boundary_positions[i + 1]
            ):
                boundary_positions[i] = (
                    boundary_probabilities[i - 1] * boundary_positions[i - 1]
                    + boundary_probabilities[i + 1] * boundary_positions[i + 1]
                ) / (boundary_probabilities[i - 1] + boundary_probabilities[i + 1])

        return (boundary_positions, boundary_probabilities)

    @staticmethod
    def _handle_extremal_bins(
        boundary_positions: np.ndarray,
        boundary_probabilities: np.ndarray,
        bin_widths: np.ndarray,
        bin_heights: np.ndarray,
        left: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Places the outer boundary of an extremal Dirac delta's two bins, keeping its
        mass and its mean. From `MIN_DIRAC_DELTAS_FOR_TAIL_DETECTION` Dirac deltas
        up, the outer bin follows the shape of the tail detected by `_tail_shape`.
        With fewer, the (d/dx)^2 = 0 boundary condition is tried. Otherwise, it falls
        back to the boundary condition d/dx = 0, i.e., an outer bin as wide as the
        inner one.

        Args:
            boundary_positions: The internal boundary positions that are
                intermediaries to get a binning, from `_determine_boundary_positions`.
                Their odd entries are the positions of the Dirac deltas.
            boundary_probabilities: The internal boundary probabilities that are
                intermediaries to get a binning. Their odd entries are the masses of
                the Dirac deltas.
            bin_widths: The internal bin widths that are intermediaries to get
                a binning.
            bin_heights:The internal bin heights that are intermediaries to get
                a binning.
            left: The position to which to do the handling.
        Returns:
            (boundary_positions, bin_widths, bin_heights): The boundary positions,
                bin widths, and bin heights that describe the output binning.
        """

        number_of_finite_dirac_deltas = (len(boundary_positions) - 1) // 2
        w0 = None
        det: float = np.nan
        if (
            number_of_finite_dirac_deltas
            >= PlotData.MIN_DIRAC_DELTAS_FOR_TAIL_DETECTION
        ):
            # Odd boundary positions and probabilities are the positions and masses
            # of the Dirac deltas, and even boundary positions the thresholds.
            if left:
                tail_shape = PlotData._tail_shape(
                    float(boundary_positions[1]),
                    float(boundary_probabilities[1]),
                    float(boundary_positions[3]),
                    float(boundary_probabilities[3]),
                    boundary_positions[2],
                    boundary_positions[4],
                    left,
                )
            else:
                tail_shape = PlotData._tail_shape(
                    float(boundary_positions[-2]),
                    float(boundary_probabilities[-2]),
                    float(boundary_positions[-4]),
                    float(boundary_probabilities[-4]),
                    boundary_positions[-3],
                    boundary_positions[-5],
                    left,
                )
            if tail_shape is not None:
                w0 = (
                    PlotData._outer_width_ratio(tail_shape)
                    * bin_widths[1 if left else -2]
                )
        # The condition reads the bin on the far side of the adjacent Dirac delta,
        # which the internal bins already hold from three Dirac deltas up. With two,
        # that bin is the other extremal bin, which is not computed yet.
        elif number_of_finite_dirac_deltas >= 3:
            p0 = boundary_probabilities[1 if left else -2]
            w1 = bin_widths[1 if left else -2]
            w2 = bin_widths[2 if left else -3]
            d2 = bin_heights[2 if left else -3]
            a = d2 * w1 - p0
            b = a * w1 - p0 * w2
            c = p0 * w1 * (w1 + w2)
            det = b * b - 4 * a * c

            if det >= 0:
                # There are real roots. Pick the smallest positive root if there is one.
                root1 = (-b + math.sqrt(det)) / (2 * a)
                root2 = (-b - math.sqrt(det)) / (2 * a)
                roots_positive = [root1 > 0, root2 > 0]

                if all(roots_positive):
                    w0 = min(root1, root2)
                elif any(roots_positive):
                    w0 = max(root1, root2)

            if math.isinf(det) or math.isnan(det):
                w0 = None

        if w0 is None:
            # The boundary condition d/dx = 0.
            boundary_positions[0 if left else -1] = boundary_positions[
                1 if left else -2
            ] + (-1 if left else 1) * (
                boundary_positions[2 if left else -2]
                - boundary_positions[1 if left else -3]
            )
        else:
            # The detected tail, or the boundary condition (d/dx)^2 = 0.
            boundary_positions[0 if left else -1] = (
                boundary_positions[1 if left else -2] + (-1 if left else 1) * w0
            )

        bin_widths[0 if left else -1] = (
            boundary_positions[1 if left else -1]
            - boundary_positions[0 if left else -2]
        )
        averageHeight = boundary_probabilities[1 if left else -2] / (
            bin_widths[0 if left else -1] + bin_widths[1 if left else -2]
        )
        bin_heights[0 if left else -1] = (
            averageHeight
            * bin_widths[1 if left else -2]
            / bin_widths[0 if left else -1]
        )
        bin_heights[1 if left else -2] = (
            averageHeight
            * bin_widths[0 if left else -1]
            / bin_widths[1 if left else -2]
        )

        return (boundary_positions, bin_widths, bin_heights)

    @staticmethod
    def _tail_shape(
        outer_position: float,
        outer_mass: float,
        next_position: float,
        next_mass: float,
        outer_threshold: float,
        next_threshold: float,
        left: bool,
    ) -> float | None:
        """
        Detects the shape of a tail from its two outermost Dirac deltas, as the shape
        xi of a generalized Pareto tail. The tail beyond a threshold has a mean
        excess, i.e., the mean distance of the tail from the threshold. The outer
        Dirac delta gives it at the outer threshold, and the mean of the two Dirac
        deltas gives it at the next threshold. For a generalized Pareto tail, the mean
        excess is linear in the threshold with slope xi / (1 - xi), which this inverts.
        A positive xi is a polynomial tail, zero an exponential tail, and a negative
        xi a bounded one. For a valid TTR the thresholds are the TTR's own split
        points, so the mean excesses are exact.

        Args:
            outer_position: The position of the extremal Dirac delta.
            outer_mass: Its mass.
            next_position: The position of its neighbour.
            next_mass: Its neighbour's mass.
            outer_threshold: The bin boundary between the two Dirac deltas.
            next_threshold: The bin boundary on the far side of the neighbour.
            left: Whether the tail is the left one.
        Returns:
            The shape xi, or `None` when the positions do not describe a tail.
        """

        mass = outer_mass + next_mass
        mean = (outer_mass * outer_position + next_mass * next_position) / mass
        if left:
            outer_excess = outer_threshold - outer_position
            next_excess = next_threshold - mean
            distance = next_threshold - outer_threshold
        else:
            outer_excess = outer_position - outer_threshold
            next_excess = mean - next_threshold
            distance = outer_threshold - next_threshold
        if not (outer_excess > 0 and next_excess > 0 and distance > 0):
            return None

        slope = (outer_excess - next_excess) / distance
        if slope <= -1:
            # A mean excess that shrinks this fast is a support bounded right past
            # the outer Dirac delta. That is the lowest shape the table holds.
            return PlotData.TAIL_SHAPE_TO_OUTER_WIDTH_RATIO[0][0]

        return slope / (1 + slope)

    @staticmethod
    def _outer_width_ratio(tail_shape: float) -> float:
        """
        Looks up the width of an extremal Dirac delta's outer bin, as a multiple of
        the width of its inner bin, for a tail of the given shape. Interpolates
        `TAIL_SHAPE_TO_OUTER_WIDTH_RATIO` linearly in the logarithm of the multiple.

        Args:
            tail_shape: The shape xi of the tail, from `_tail_shape`.
        Returns:
            The width of the outer bin over the width of the inner bin.
        """

        table = PlotData.TAIL_SHAPE_TO_OUTER_WIDTH_RATIO
        if tail_shape <= table[0][0]:
            return table[0][1]
        if tail_shape >= table[-1][0]:
            return table[-1][1]

        for (low_shape, low_ratio), (high_shape, high_ratio) in zip(
            table[:-1], table[1:]
        ):
            if tail_shape <= high_shape:
                fraction = (tail_shape - low_shape) / (high_shape - low_shape)
                return math.exp(
                    (1 - fraction) * math.log(low_ratio)
                    + fraction * math.log(high_ratio)
                )

        return table[-1][1]

    @staticmethod
    def _get_binning(
        boundary_positions: np.ndarray,
        boundary_probabilities: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Finds the binning for the calculated internal boundary positions and
        probabilities of finite and sorted Dirac deltas.

        Args:
            boundary_positions: The internal boundary positions that are
                intermediaries to get a binning, from `_determine_boundary_positions`.
                Their odd entries are the positions of the Dirac deltas.
            boundary_probabilities: The internal boundary probabilities that are
                intermediaries to get a binning. Their odd entries are the masses of
                the Dirac deltas.
        Returns:
            (boundary_positions, bin_widths, bin_heights): The boundary positions,
                bin widths, and bin heights that describe the output binning.
        """

        number_of_finite_dirac_deltas = (len(boundary_positions) - 1) // 2

        # Initialize the binning and populate it for the internal bins.
        numberOfBins = 2 * number_of_finite_dirac_deltas
        bin_widths = np.full(numberOfBins, np.nan)
        bin_widths[1:-1] = boundary_positions[2:-1] - boundary_positions[1:-2]
        bin_heights = np.full(numberOfBins, np.nan)

        # Vectorise internal bin height computation for Dirac deltas 1..N-2.
        # The bins left of Dirac deltas 1..N-2 are 2, 4, ..., 2N-4, and the bins
        # right of them are 3, 5, ..., 2N-3. Strided slices select them, because
        # the ulab NumPy of CircuitPython does not support integer-array indexing.
        if number_of_finite_dirac_deltas > 2:
            # The masses of Dirac deltas 1..N-2 are boundary probabilities 3, 5, ...
            masses = boundary_probabilities[
                3 : 2 * number_of_finite_dirac_deltas - 2 : 2
            ]
            left = slice(2, 2 * number_of_finite_dirac_deltas - 2, 2)
            right = slice(3, 2 * number_of_finite_dirac_deltas - 2, 2)
            w_left = bin_widths[left]
            w_right = bin_widths[right]
            avg_h = masses / (w_left + w_right)
            bin_heights[left] = avg_h * w_right / w_left
            bin_heights[right] = avg_h * w_left / w_right

        boundary_positions, bin_widths, bin_heights = PlotData._handle_extremal_bins(
            boundary_positions,
            boundary_probabilities,
            bin_widths,
            bin_heights,
            left=True,
        )
        boundary_positions, bin_widths, bin_heights = PlotData._handle_extremal_bins(
            boundary_positions,
            boundary_probabilities,
            bin_widths,
            bin_heights,
            left=False,
        )

        return (boundary_positions, bin_widths, bin_heights)

    @staticmethod
    def create_binning(
        finite_sorted_dirac_deltas: list[DiracDelta],
        exponent: int,
        use_ttr_binning: bool,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        - If `use_ttr_binning` is true:
            Creates a binning using the TTR binning method. The TTR binning method
            creates the unique binning (up to extremal bins determined by the imposed
            boundary conditions) such that the TTR of the binning exactly coincides
            with the input Dirac deltas. Requires the input Dirac deltas to form
            a valid TTR.
        - If `use_ttr_binning` is false:
            Creates a binning without requiring the valid TTR property, where the
            internal bin boundaries are determined only by adjacent Dirac deltas and
            the average of two bins surrounding a Dirac delta is the Dirac delta itself.

        Args:
            finite_sorted_dirac_deltas: The input Dirac deltas with finite
                and sorted positions. At least two are required.
            exponent: The TTR order, i.e., the base-2 logarithm of the number of
                Dirac deltas in the TTR. The number of bins in the output binning
                is twice the number of Dirac deltas in the TTR.
            use_ttr_binning: Flag specifying whether to use the TTR binning method.
        Returns:
            (boundary_positions, bin_widths, bin_heights): The boundary positions,
                bin widths, and bin heights that describe the output binning.
        Raises:
            ValueError: When given fewer than two Dirac deltas. Both bins that
                surround a lone Dirac delta would be bounded on the outside by a
                boundary placed relative to a Dirac delta that does not exist, so
                the binning has no widths. Callers plot a lone Dirac delta as a
                Dirac delta instead, which is what `_construct_plot_data` does.
        """

        positions = np.array(
            [dd.position for dd in finite_sorted_dirac_deltas], dtype=np.float64
        )
        masses = np.array(
            [dd.mass for dd in finite_sorted_dirac_deltas], dtype=np.float64
        )

        return PlotData._create_binning_from_arrays(
            positions, masses, exponent, use_ttr_binning
        )

    @staticmethod
    def _create_binning_from_arrays(
        positions: np.ndarray,
        masses: np.ndarray,
        exponent: int,
        use_ttr_binning: bool,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        `create_binning` for Dirac deltas given as arrays of their positions and
        masses, which `ttr_from_samples` uses to bin samples without building a
        Dirac delta per sample.

        Args:
            positions: The finite and sorted positions of the input Dirac deltas.
                At least two are required.
            masses: The masses of the input Dirac deltas.
            exponent: The TTR order, as for `create_binning`.
            use_ttr_binning: Flag specifying whether to use the TTR binning method.
        Returns:
            (boundary_positions, bin_widths, bin_heights): The boundary positions,
                bin widths, and bin heights that describe the output binning.
        Raises:
            ValueError: When given fewer than two Dirac deltas, as for
                `create_binning`.
        """

        if len(positions) < 2:
            raise ValueError(
                "plot_histogram_dirac_deltas: create_binning requires at least two "
                f"Dirac deltas, got {len(positions)}"
            )

        (
            boundary_positions,
            boundary_probabilities,
        ) = PlotData._determine_boundary_positions(
            positions, masses, exponent, use_ttr_binning
        )

        return PlotData._get_binning(boundary_positions, boundary_probabilities)

    @staticmethod
    def _cell_boundary_positions(
        positions: np.ndarray, interior_weight: float
    ) -> np.ndarray:
        """
        Places one bin boundary between each pair of adjacent Dirac deltas, at the
        fraction `interior_weight` of the way from the left to the right Dirac delta.
        The two extremal boundaries are mirror images of their inner neighbours
        about the extremal Dirac deltas, which is the (d/dx) = 0 boundary condition.

        Args:
            positions: Strictly increasing Dirac delta positions. At least two are
                required, since every boundary is placed relative to an adjacent
                Dirac delta.
            interior_weight: Position of an internal boundary within the interval
                between two adjacent Dirac deltas. `0.5` places it at the midpoint.
        Returns:
            boundary_positions: The (N + 1) boundary positions of the N cells.
        Raises:
            ValueError: When given fewer than two positions.
        """

        number_of_cells = len(positions)
        if number_of_cells < 2:
            raise ValueError(
                "plot_histogram_dirac_deltas: _cell_boundary_positions requires at "
                f"least two positions, got {number_of_cells}"
            )

        boundary_positions = np.array([np.nan] * (number_of_cells + 1))
        boundary_positions[1:-1] = (1 - interior_weight) * positions[
            :-1
        ] + interior_weight * positions[1:]
        boundary_positions[0] = 2 * positions[0] - boundary_positions[1]
        boundary_positions[-1] = 2 * positions[-1] - boundary_positions[-2]

        return boundary_positions

    @staticmethod
    def _first_collapsed_cell(
        boundary_positions: np.ndarray, positions: np.ndarray
    ) -> int | None:
        """
        Finds the first cell that does not hold its Dirac delta strictly inside it,
        i.e., whose bin is empty or inverted, or has a boundary on or past its Dirac
        delta. Boundaries that are valid in exact arithmetic can still collapse a
        cell once they are rounded.

        Args:
            boundary_positions: The (N + 1) boundary positions of the N cells.
            positions: The N Dirac delta positions, one per cell.
        Returns:
            The index of the first such cell, or `None` when every cell holds its
            Dirac delta.
        """

        for i in range(len(positions)):
            if not (boundary_positions[i] < positions[i] < boundary_positions[i + 1]):
                return i

        return None

    @staticmethod
    def _merge_into_nearest_neighbour(
        dirac_deltas: list[DiracDelta], index: int
    ) -> list[DiracDelta]:
        """
        Merges the Dirac delta at `index` with its nearest neighbour, the one across
        the smaller gap and the lower one on a tie. The merge keeps the mass and the
        mean of the two.

        Args:
            dirac_deltas: Dirac deltas with strictly increasing positions, at least
                two.
            index: The index of the Dirac delta to merge.
        Returns:
            The Dirac deltas with the two merged into one.
        """

        if index == 0:
            neighbour = 1
        elif index == len(dirac_deltas) - 1:
            neighbour = index - 1
        else:
            gap_below = dirac_deltas[index].position - dirac_deltas[index - 1].position
            gap_above = dirac_deltas[index + 1].position - dirac_deltas[index].position
            neighbour = index - 1 if gap_below <= gap_above else index + 1

        lower = min(index, neighbour)
        return (
            dirac_deltas[:lower]
            + [dirac_deltas[lower] + dirac_deltas[lower + 1]]
            + dirac_deltas[lower + 2 :]
        )

    @staticmethod
    def _local_mean_boundary_positions(positions: np.ndarray) -> np.ndarray | None:
        """
        Places the bin boundaries so that every bin is centred on the Dirac delta it
        holds, which preserves each Dirac delta's own mean and not just the mean of
        the whole distribution. This is the property the TTR binning method has, and
        the property `create_binning` gets from using two bins per Dirac delta.

        Centring bin `i` on Dirac delta `i` means (b[i] + b[i + 1]) / 2 == p[i], so
        the boundaries follow the recurrence b[i + 1] = 2 * p[i] - b[i]: the whole
        binning is determined by the first boundary alone. Writing e[i] for the
        distance p[i] - b[i], the width of bin `i` is 2 * e[i] and the recurrence is
        e[i + 1] = g[i] - e[i], for gaps g[i] = p[i + 1] - p[i]. Each e[i] is
        therefore affine in e[0] with an alternating sign, so requiring every
        boundary to stay strictly between the two Dirac deltas it separates,
        0 < e[i] < g[i], is a set of linear constraints on e[0].

        Those constraints have no solution for arbitrary Dirac delta spacing, since
        the recurrence is undamped and alternating gaps drive e[i] out of its
        interval. Where there is a solution, this takes the midpoint of it, which is
        the choice furthest from every constraint and so has the widest bins. The
        constraints are solved in exact arithmetic, so a feasible interval that is
        narrow enough can still yield boundaries that rounding pushes onto or past a
        Dirac delta. The constructed boundaries are therefore re-checked and also
        rejected in that case.

        Args:
            positions: Strictly increasing Dirac delta positions, at least two.
        Returns:
            boundary_positions: The (N + 1) boundary positions of the N Dirac deltas, or
                `None` when no first boundary exists that solve the constraints
                or when rounding pushes a bin boundary beyond the next Dirac delta.
        """

        gaps = positions[1:] - positions[:-1]

        # Intersect the constraints 0 < e[i] < g[i] to bound e[0].
        lowest, highest = 0.0, float(np.inf)
        constant, sign = 0.0, 1.0
        for gap in gaps:
            if sign > 0:
                lowest = max(lowest, -constant)
                highest = min(highest, float(gap) - constant)
            else:
                lowest = max(lowest, constant - float(gap))
                highest = min(highest, constant)
            constant, sign = float(gap) - constant, -sign

        if not lowest < highest:
            return None

        boundary_positions = np.array([np.nan] * (len(positions) + 1))
        distance_to_dirac_delta = (lowest + highest) / 2
        boundary_positions[0] = positions[0] - distance_to_dirac_delta
        for i, position in enumerate(positions):
            boundary_positions[i + 1] = position + distance_to_dirac_delta
            if i + 1 < len(positions):
                distance_to_dirac_delta = (
                    positions[i + 1] - position - distance_to_dirac_delta
                )

        # The constraints are solved in exact arithmetic, so re-check the result
        # rather than trust that rounding kept every bin valid and populated.
        if PlotData._first_collapsed_cell(boundary_positions, positions) is not None:
            return None

        return boundary_positions

    @staticmethod
    def create_non_uniform_binning(
        finite_sorted_dirac_deltas: list[DiracDelta],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Creates a binning with one bin per Dirac delta, where the bin holding a
        Dirac delta carries exactly that Dirac delta's mass and the internal bin
        boundaries lie between adjacent Dirac deltas. The bins therefore have
        non-uniform widths that follow the spacing of the Dirac deltas, and there
        are no empty bins between Dirac deltas, i.e., the binning reads the input
        as a continuous distribution with no gaps in its support.

        The total mass of the binning always equals that of the input. Where the
        Dirac delta spacing allows it, every bin is also centred on the Dirac delta
        it holds, so each Dirac delta's own mean is preserved and not merely the
        mean of the whole distribution, which is the property the TTR binning method
        has. That is not solvable for every spacing (see
        `_local_mean_boundary_positions`), and where it is not, the boundaries fall
        back to a single placement fraction shared by all of them, solved to match
        the mean of the whole distribution. That mean is then exact, unless matching
        it would collapse an extremal bin to zero width and the placement is
        therefore clamped to `MIN_INTERIOR_WEIGHT`. The mean is then approximate,
        but never further off than placing every boundary at a midpoint would leave
        it.

        Near a large magnitude, that placement can round a boundary onto or past a
        Dirac delta, which would leave a bin of zero width and infinite density. The
        Dirac delta of such a bin is then merged into its nearest neighbour, which
        keeps the mass and the mean, and the merged Dirac deltas are binned instead.
        The binning then has one bin fewer than there are input Dirac deltas.

        Args:
            finite_sorted_dirac_deltas: The input Dirac deltas with finite,
                strictly increasing positions. At least two are required.
        Returns:
            (boundary_positions, bin_widths, bin_heights): The boundary positions,
                bin widths, and bin heights that describe the output binning.
        Raises:
            ValueError: When given fewer than two Dirac deltas. A lone Dirac delta
                has no adjacent Dirac delta to place a bin boundary against, so it
                has no bin width. Callers plot it as a Dirac delta instead, which is
                what `_construct_plot_data` does. Also when merging leaves a lone
                Dirac delta, which takes two Dirac deltas only a few ulps apart.
        """

        if len(finite_sorted_dirac_deltas) < 2:
            raise ValueError(
                "plot_histogram_dirac_deltas: create_non_uniform_binning requires "
                f"at least two Dirac deltas, got {len(finite_sorted_dirac_deltas)}"
            )

        positions = np.array([dd.position for dd in finite_sorted_dirac_deltas])
        masses = np.array([dd.mass for dd in finite_sorted_dirac_deltas])

        # Preserving every Dirac delta's own mean also preserves the mean of the
        # whole distribution, so prefer it wherever the spacing admits it.
        boundary_positions = PlotData._local_mean_boundary_positions(positions)
        if boundary_positions is not None:
            bin_widths = boundary_positions[1:] - boundary_positions[:-1]
            return (boundary_positions, bin_widths, masses / bin_widths)

        target_mean = float(np.sum(positions * masses))

        def mean_of(interior_weight: float) -> float:
            boundaries = PlotData._cell_boundary_positions(positions, interior_weight)
            return float(np.sum(masses * (boundaries[:-1] + boundaries[1:]) / 2))

        # The mean of the binning is affine in `interior_weight`, so two
        # evaluations determine the value that reproduces the input mean. The mean
        # can also be independent of `interior_weight` (e.g., for two Dirac
        # deltas, where the mirrored boundaries centre both cells on their Dirac
        # delta for any placement), in which case any placement is mean-exact.
        mean_at_zero = mean_of(0.0)
        mean_at_one = mean_of(1.0)
        if abs(mean_at_one - mean_at_zero) < 1e-15:
            interior_weight = 0.5
        else:
            interior_weight = (target_mean - mean_at_zero) / (
                mean_at_one - mean_at_zero
            )

        # Keep the placement off the ends of [0, 1], where an extremal bin would
        # collapse to zero width: the width of an internal bin is a convex
        # combination of the two gaps adjacent to its Dirac delta, so it stays
        # positive throughout, but the width of an extremal bin is twice the weight
        # times the extremal gap. The bound is only there to keep the widths
        # positive, so it is as small as it can be: clamping is what makes the mean
        # of the binning approximate, and a placement near an end is the correct
        # answer for a value whose extremal Dirac deltas are much closer together
        # than the rest, where the density really is that much higher there.
        interior_weight = min(
            max(interior_weight, PlotData.MIN_INTERIOR_WEIGHT),
            1 - PlotData.MIN_INTERIOR_WEIGHT,
        )

        boundary_positions = PlotData._cell_boundary_positions(
            positions, interior_weight
        )

        # Near a large magnitude, the clamped placement can round a boundary onto or
        # past a Dirac delta, which would leave a bin of zero width and infinite
        # density. Merge that Dirac delta into its nearest neighbour, which keeps the
        # mass and the mean, and bin the merged Dirac deltas instead.
        collapsed_cell = PlotData._first_collapsed_cell(boundary_positions, positions)
        if collapsed_cell is not None:
            return PlotData.create_non_uniform_binning(
                PlotData._merge_into_nearest_neighbour(
                    finite_sorted_dirac_deltas, collapsed_cell
                )
            )

        bin_widths = boundary_positions[1:] - boundary_positions[:-1]
        bin_heights = masses / bin_widths

        return (boundary_positions, bin_widths, bin_heights)

    @staticmethod
    def _split_extremal_cells(
        finite_sorted_dirac_deltas: list[DiracDelta],
        boundary_positions: np.ndarray,
        bin_heights: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Splits the bin of each extremal Dirac delta of a non-uniform binning into two
        bins at the Dirac delta, whose outer bin follows the shape of the tail detected
        by `_tail_shape`. The inner bin keeps its boundary with the neighbouring bin.
        Both keep the Dirac delta's mass and its mean, the same way the two bins of a
        Dirac delta in `create_binning` do. A tail whose shape cannot be detected keeps
        its single bin, so the binning has up to two bins more than Dirac deltas.

        Args:
            finite_sorted_dirac_deltas: The Dirac deltas the binning was made from.
            boundary_positions: The (N + 1) boundary positions of the binning.
            bin_heights: The N bin heights of the binning.
        Returns:
            (boundary_positions, bin_heights): The boundary positions and bin heights
                of the binning with its extremal bins split. Unchanged when the
                binning does not have one bin per Dirac delta, which happens when
                `create_non_uniform_binning` merged Dirac deltas.
        """

        if len(bin_heights) < 3 or len(bin_heights) != len(finite_sorted_dirac_deltas):
            return (boundary_positions, bin_heights)

        positions = list(boundary_positions)
        heights = list(bin_heights)
        for left in (True, False):
            if left:
                dirac_delta = finite_sorted_dirac_deltas[0]
                tail_shape = PlotData._tail_shape(
                    dirac_delta.position,
                    dirac_delta.mass,
                    finite_sorted_dirac_deltas[1].position,
                    finite_sorted_dirac_deltas[1].mass,
                    positions[1],
                    positions[2],
                    left,
                )
                inner_width = positions[1] - dirac_delta.position
            else:
                dirac_delta = finite_sorted_dirac_deltas[-1]
                tail_shape = PlotData._tail_shape(
                    dirac_delta.position,
                    dirac_delta.mass,
                    finite_sorted_dirac_deltas[-2].position,
                    finite_sorted_dirac_deltas[-2].mass,
                    positions[-2],
                    positions[-3],
                    left,
                )
                inner_width = dirac_delta.position - positions[-2]
            if tail_shape is None or not inner_width > 0:
                continue

            outer_width = PlotData._outer_width_ratio(tail_shape) * inner_width
            average_height = dirac_delta.mass / (outer_width + inner_width)
            outer_height = average_height * inner_width / outer_width
            inner_height = average_height * outer_width / inner_width
            if left:
                positions = [
                    dirac_delta.position - outer_width,
                    dirac_delta.position,
                ] + positions[1:]
                heights = [outer_height, inner_height] + heights[1:]
            else:
                positions = positions[:-1] + [
                    dirac_delta.position,
                    dirac_delta.position + outer_width,
                ]
                heights = heights[:-1] + [inner_height, outer_height]

        return (
            np.array(positions, dtype=np.float64),
            np.array(heights, dtype=np.float64),
        )

    @staticmethod
    def _bin_pdf_expected_dirac_delta(
        boundary_positions: np.ndarray, bin_widths: np.ndarray, bin_heights: np.ndarray
    ) -> DiracDelta:
        """
        Computes the expected Dirac delta of an input bin PDF.

        Args:
            boundary_positions: Positions of bin boundaries of the input bin PDF.
            bin_widths: Widths of the bins of the input bin PDF.
            bin_heights: Heights of the bins of the input bin PDF.
        Returns:
            expected_dirac_delta: The expected Dirac delta in the format np.array([position, mass]).
        Raises:
            ValueError: If the bin PDF has no mass. A split of bins only a few
                ulps wide can leave one half empty.
        """

        probabilities = bin_widths * bin_heights
        probability_sum = float(np.sum(probabilities))
        if probability_sum == 0:
            raise ValueError(
                "The bin PDF has no mass, so it has no expected Dirac delta."
            )
        bin_centres = (boundary_positions[1:] + boundary_positions[:-1]) / 2
        moment_sum = float(np.sum(probabilities * bin_centres))

        expected_dirac_delta = DiracDelta(
            moment_sum / probability_sum, mass=probability_sum
        )

        return expected_dirac_delta

    @staticmethod
    def bin_pdf_to_ttr(
        boundary_positions: np.ndarray,
        bin_widths: np.ndarray,
        bin_heights: np.ndarray,
        order: int,
    ) -> list[DiracDelta]:
        """
        Computes TTR for an input bin PDF.

        For an input bin PDF with N bins, the shapes are:
            ``len(boundary_positions) == N + 1``
            ``len(bin_widths) == N``
            ``len(bin_heights) == N``

        Args:
            boundary_positions: Positions of the bin boundaries of the input bin PDF.
            bin_widths: Widths of the bins of the input bin PDF.
            bin_heights: Heights of the bins of the input bin PDF.
            order: TTR order.
        Returns:
            ttr: The TTR of the input bin PDF, a (2 ** `order`)-length array of Dirac deltas
                with each Dirac delta of the form np.array([position, mass]).
        Raises:
            ValueError: If a split leaves a half with no mass.
        """

        expected_dirac_delta = PlotData._bin_pdf_expected_dirac_delta(
            boundary_positions, bin_widths, bin_heights
        )
        ttr: list[DiracDelta] = []

        if order == 0:
            return [expected_dirac_delta]

        low_boundary_positions: np.ndarray = np.array([], dtype=np.float64)
        low_bin_widths: np.ndarray = np.array([], dtype=np.float64)
        low_bin_heights: np.ndarray = np.array([], dtype=np.float64)
        high_boundary_positions: np.ndarray = np.array([], dtype=np.float64)
        high_bin_widths: np.ndarray = np.array([], dtype=np.float64)
        high_bin_heights: np.ndarray = np.array([], dtype=np.float64)

        # Split at the first boundary at or past the mean. Comparing them all at once
        # is much faster than walking them, and finds the same boundary, NaN-valued
        # and unsorted boundaries included.
        at_or_past_mean = boundary_positions >= expected_dirac_delta.position
        i = int(np.argmax(at_or_past_mean))
        if at_or_past_mean[i]:
            boundary_position = boundary_positions[i]
            if boundary_position == expected_dirac_delta.position:
                low_boundary_positions = boundary_positions[: i + 1]
                low_bin_widths = bin_widths[:i]
                low_bin_heights = bin_heights[:i]
                high_boundary_positions = boundary_positions[i:]
                high_bin_widths = bin_widths[i:]
                high_bin_heights = bin_heights[i:]
            else:
                low_boundary_positions = np.append(
                    boundary_positions[:i], expected_dirac_delta.position
                )
                low_bin_widths = np.append(
                    bin_widths[: i - 1],
                    expected_dirac_delta.position - boundary_positions[i - 1],
                )
                low_bin_heights = bin_heights[:i]
                high_boundary_positions = np.insert(
                    boundary_positions[i:], 0, expected_dirac_delta.position
                )
                high_bin_widths = np.insert(
                    bin_widths[i:], 0, boundary_position - expected_dirac_delta.position
                )
                high_bin_heights = bin_heights[i - 1 :]

        ttr += PlotData.bin_pdf_to_ttr(
            low_boundary_positions, low_bin_widths, low_bin_heights, order - 1
        )
        ttr += PlotData.bin_pdf_to_ttr(
            high_boundary_positions, high_bin_widths, high_bin_heights, order - 1
        )

        return ttr

    @staticmethod
    def _ttr_of_binning(
        positions: np.ndarray, masses: np.ndarray, order: int
    ) -> list[DiracDelta]:
        """
        The TTR of order `order` of Dirac deltas with finite, sorted, distinct
        positions, taken of their binning by `create_binning` without the TTR
        binning method. This is how both `_construct_plot_data` and
        `ttr_from_samples` reduce Dirac deltas to a TTR.

        Args:
            positions: The positions of the Dirac deltas. At least two are required.
            masses: The masses of the Dirac deltas.
            order: The TTR order.
        Returns:
            The 2 ** `order` Dirac deltas of the TTR.
        Raises:
            ValueError: When given fewer than two Dirac deltas, or when a split of
                the binning leaves a half with no mass.
        """

        boundary_positions, bin_widths, bin_heights = (
            PlotData._create_binning_from_arrays(positions, masses, 0, False)
        )
        return PlotData.bin_pdf_to_ttr(
            boundary_positions, bin_widths, bin_heights, order
        )

    @staticmethod
    def ttr_from_samples(
        samples: np.ndarray | list[float], order: int
    ) -> DistributionalValue:
        """
        Reduces samples to a TTR of order `order`, i.e., to 2 ** `order` Dirac
        deltas, for example to export them as a Ux string.

        Gives the same Dirac deltas as reducing `DistributionalValue.from_samples`
        the way `_construct_plot_data` reduces more Dirac deltas than it can plot.
        Both combine coincident Dirac deltas with
        `DistributionalValue.combine_sorted_arrays`, and take the TTR of their
        binning with `_ttr_of_binning`. This works on arrays of the samples instead
        of a Dirac delta per sample, which for a million samples is over ten times
        faster. Where plotting falls back to merging neighbouring Dirac deltas when
        the binning cannot be split, this raises instead.

        Non-finite samples are not part of the TTR. They keep their share of the
        mass, as NaN, -Inf, and +Inf Dirac deltas.

        Args:
            samples: 1-D array of float samples (may contain NaN/Inf).
            order: The TTR order, at least 0.
        Returns:
            A `DistributionalValue` holding the 2 ** `order` Dirac deltas of the TTR,
                followed by any for non-finite samples. For an order of 0 and finite
                samples that are all equal, the TTR is the single Dirac delta at
                their position.
        Raises:
            ValueError: If `samples` is not 1-D, is empty, or `order` is negative.
                Also if no sample is finite, or if, for an order above 0, the
                finite samples are all equal, to within the threshold of
                `DistributionalValue.combine_dirac_deltas`, as there is then
                nothing to split. Also if the samples are too concentrated to split
                into a valid TTR of 2 ** `order` Dirac deltas in double precision,
                for example 999 samples at 5.0 and one at 5.000000000001 from order
                2. A lower order may then work.
        """

        samples = np.asarray(samples, dtype=np.float64)
        if np.ndim(samples) != 1:
            raise ValueError(
                f"samples must be a 1-D array, got {np.ndim(samples)} dimensions."
            )
        if len(samples) == 0:
            raise ValueError("samples array must not be empty.")
        if order < 0:
            raise ValueError(f"order must be at least 0, got {order}")

        # Each sample has the mass `DistributionalValue.from_samples` gives it.
        mass_per_sample = 1.0 / len(samples)
        positions = np.sort(samples[np.isfinite(samples)])
        masses = np.full(len(positions), mass_per_sample)
        if len(positions) >= 2:
            positions, masses, _ = DistributionalValue.combine_sorted_arrays(
                positions, masses
            )

        if len(positions) == 0:
            raise ValueError(
                "ttr_from_samples: no sample is finite, so there is no TTR"
            )

        dirac_deltas: list[DiracDelta]
        if len(positions) == 1:
            if order > 0:
                raise ValueError(
                    "ttr_from_samples: the finite samples are all equal, to within "
                    "the threshold of `DistributionalValue.combine_dirac_deltas`, so "
                    f"there is nothing to split into a TTR of order {order}. Order 0 "
                    "gives their single Dirac delta."
                )
            dirac_deltas = [DiracDelta(float(positions[0]), mass=float(masses[0]))]
        else:
            # The binning puts each boundary at the mass-weighted mean of the Dirac
            # deltas either side of it, so most of the mass can sit in bins only a
            # few doubles wide. Splitting those into many parts leaves some with no
            # mass, which `bin_pdf_to_ttr` rejects, or rounds the means of some onto
            # one position, which is no TTR.
            too_concentrated = (
                "ttr_from_samples: the samples are too concentrated for a TTR of "
                f"order {order} in double precision, as splitting them that finely "
                "leaves parts with no mass or with the same position. A lower order "
                "may work."
            )
            try:
                dirac_deltas = PlotData._ttr_of_binning(positions, masses, order)
            except ValueError as error:
                raise ValueError(too_concentrated) from error
            # Checking the TTR combines Dirac deltas at one position, which then
            # leaves fewer than the TTR has, even if the rest are a valid TTR.
            ttr = DistributionalValue(dirac_deltas=dirac_deltas)
            if not ttr.check_is_full_valid_TTR() or ttr.UR_order != 2**order:
                raise ValueError(too_concentrated)

        for position, is_position in (
            (np.nan, np.isnan(samples)),
            (-np.inf, samples == -np.inf),
            (np.inf, samples == np.inf),
        ):
            count = int(np.sum(is_position))
            if count > 0:
                dirac_deltas.append(DiracDelta(position, mass=count / len(samples)))

        dist = DistributionalValue(dirac_deltas=dirac_deltas)
        dist.sort()

        return dist

    def _construct_plot_data(self) -> None:
        """Constructs the `PlotData`, after parsing the given `DistributionalValue`.
        Generates the boundary positions and bin heights, ready for plotting.

        :raises ValueError: When the plotting_resolution is not a power of 2.
        """
        # Create the list of finite Dirac deltas.
        self.dist.drop_zero_mass_positions()

        self.dist.combine_dirac_deltas()

        # Create the list of finite sorted Dirac deltas.
        # Last three positions are for non-finite values
        finite_dirac_deltas: list[DiracDelta] = self.dist.finite_dirac_deltas

        # If no finite Dirac deltas found, then return.
        if len(finite_dirac_deltas) == 0:
            return

        if len(finite_dirac_deltas) == 1:
            self.positions = np.array(
                [finite_dirac_deltas[0].position], dtype=np.float64
            )
            self.masses = np.array([finite_dirac_deltas[0].mass], dtype=np.float64)
            return

        # Set plot resolution to (2 * N) where N is machine representation,
        # capped at MAX_BINS.
        machine_representation = 2 ** math.floor(math.log(self.dist.UR_order, 2))
        self.plotting_resolution = int(
            min(machine_representation * 2, self.MAX_BINS)
            if self.plotting_resolution is None
            else min(
                machine_representation * 2, self.plotting_resolution, self.MAX_BINS
            )
        )
        log2_of_plotting_resolution = self.plotting_resolution.bit_length() - 1
        self.plotting_ttr_order = log2_of_plotting_resolution - 1

        if self.plotting_resolution > 2 and self.plotting_resolution > 2 ** (
            self.plotting_ttr_order + 1
        ):
            raise ValueError(
                "plot_histogram_dirac_deltas: plotting_resolution must be a power of 2!"
            )

        # Use TTR binning method for Athens/Atlas as they use TTR split as compression method.
        # Not all Athens/Atlas outputs as valid TTRs as they could have ended with a libm operation
        # on just the positions and so does not trigger an additional split. Jupiter can produce
        # valid TTRs as well as so apply the check first before falling back to the non-uniform binning method.
        use_ttr_route = (
            self.dist.UR_type in TTR_NATIVE_UR_TYPES
            or self.dist.check_is_full_valid_TTR()
        )

        # The positions and masses of the finite Dirac deltas, which sorting puts
        # first.
        finite_positions = self.dist.positions[: len(finite_dirac_deltas)]
        finite_masses = self.dist.masses[: len(finite_dirac_deltas)]

        if use_ttr_route:
            try:
                # Find the TTR of the binning in which the average of the two bins
                # surrounding a Dirac delta is the Dirac delta itself. This is always
                # a valid TTR.
                ttr = PlotData._ttr_of_binning(
                    finite_positions, finite_masses, self.plotting_ttr_order
                )

                # Create the binning from the obtained (valid) TTR using the TTR binning method.
                boundary_positions, bin_widths, bin_heights = PlotData.create_binning(
                    ttr, self.plotting_ttr_order, True
                )

                self.positions = boundary_positions
                self.masses = bin_heights
                return
            # On CircuitPython, a bin narrower than single precision resolves has no
            # width, and dividing by it raises where NumPy gives an infinity.
            except (ValueError, TypeError, ZeroDivisionError):
                pass

        # For non-TTR input, use the non-uniform binning instead, which places one bin per Dirac delta with boundaries
        # following the spacing of the Dirac deltas, and two for each extremal Dirac
        # delta when there are enough to detect the tails. This reads the input as a
        # continuous distribution with no gaps in its support, whereas a
        # uniform-width histogram leaves empty bins wherever the Dirac deltas are
        # sparser than the bin width.
        # A binning needs two bins, so it needs two Dirac deltas. The order is only
        # 0 when the caller asked for a `plotting_resolution` of 2.
        reduction_order = max(self.plotting_ttr_order, 1)
        if len(finite_dirac_deltas) > 2**reduction_order:
            # More Dirac deltas than the plot can resolve. Reduce them to
            # (2 ** `reduction_order`) of them, which is the same number of Dirac
            # deltas the valid-TTR path represents the value with, via the same two
            # steps that path uses: bin the Dirac deltas without assuming a valid
            # TTR, then take the TTR of that binning. Both steps preserve the total
            # mass and the mean.
            try:
                finite_dirac_deltas = PlotData._ttr_of_binning(
                    finite_positions, finite_masses, reduction_order
                )
            # On CircuitPython, a support too narrow for single precision gives bins
            # with no width, and dividing by one raises ZeroDivisionError.
            except (ValueError, ZeroDivisionError):
                # A support only a few ulps wide cannot be split. Merge runs of
                # neighbouring Dirac deltas instead, into as many as the TTR
                # would give. Each merge keeps the mass and the mean. The runs
                # are those `np.array_split` gives, which ulab lacks: the first
                # `longer` of them hold one more Dirac delta.
                size, longer = divmod(len(finite_dirac_deltas), 2**reduction_order)
                bounds = [
                    run * size + min(run, longer)
                    for run in range(2**reduction_order + 1)
                ]
                finite_dirac_deltas = [
                    sum(
                        finite_dirac_deltas[start + 1 : end], finite_dirac_deltas[start]
                    )
                    for start, end in zip(bounds, bounds[1:])
                ]

        boundary_positions, bin_widths, bin_heights = (
            PlotData.create_non_uniform_binning(finite_dirac_deltas)
        )

        # A single bin centred on an extremal Dirac delta is a flat tail. From enough
        # Dirac deltas to detect the shape of the tail, follow it with two bins.
        if len(finite_dirac_deltas) >= PlotData.MIN_DIRAC_DELTAS_FOR_TAIL_DETECTION:
            boundary_positions, bin_heights = PlotData._split_extremal_cells(
                finite_dirac_deltas, boundary_positions, bin_heights
            )

        self.positions = boundary_positions
        self.masses = bin_heights
