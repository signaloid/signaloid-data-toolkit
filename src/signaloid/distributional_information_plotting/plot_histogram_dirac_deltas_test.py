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


import math
import os
import random
import unittest

import numpy as np
from signaloid.distributional.dirac_delta import DiracDelta
from signaloid.distributional.distributional import UR_TYPE_ATHENS, DistributionalValue
from signaloid.distributional_information_plotting.plot_histogram_dirac_deltas import (
    PlotData,
)
from signaloid.distributional_information_plotting.plot_wrapper import plot


class TestCreateBinning(unittest.TestCase):
    def test_create_binning_property_dirac_deltas_average_of_bins(self) -> None:
        """
        Tests that the binning created by the method `PlotData.create_binning()`
        satisfies for each input Dirac delta the property that the average of the two bins that
        surround the input Dirac delta is equal to the Dirac delta itself.
        """
        probability_threshold: float = 1e-12
        position_threshold: float = 1e-12
        number_of_testcases: int = 1000
        low_range: tuple[int, int] = (-100, 0)
        high_range: tuple[int, int] = (0, 100)
        input_dirac_delta_ensembles: list[list[DiracDelta]] = []
        number_of_dirac_deltas: list[int] = []

        for i in range(number_of_testcases):
            current_number_of_dirac_deltas = random.sample(range(2, 1_000 + 1), 1)[0]
            number_of_dirac_deltas.append(current_number_of_dirac_deltas)
            low_value = np.random.uniform(*low_range)
            high_value = np.random.uniform(*high_range)
            dirac_delta_positions = np.random.uniform(
                low_value, high_value, current_number_of_dirac_deltas
            )
            dirac_delta_masses = np.random.uniform(0, 1, current_number_of_dirac_deltas)
            dirac_delta_masses /= sum(dirac_delta_masses)

            dirac_deltas = [
                DiracDelta(position, mass=mass)
                for position, mass in zip(dirac_delta_positions, dirac_delta_masses)
            ]
            dirac_deltas.sort()

            input_dirac_delta_ensembles.append(dirac_deltas)

        for i, input_ensemble in enumerate(input_dirac_delta_ensembles):
            boundary_positions, bin_widths, bin_heights = PlotData.create_binning(
                input_ensemble, 0, False
            )

            for j in range(number_of_dirac_deltas[i]):
                probability_under_first_bin = bin_widths[2 * j] * bin_heights[2 * j]
                probability_under_second_bin = (
                    bin_widths[2 * j + 1] * bin_heights[2 * j + 1]
                )
                probability_under_bins = (
                    probability_under_first_bin + probability_under_second_bin
                )
                mean_of_first_bin = boundary_positions[2 * j] + bin_widths[2 * j] / 2
                mean_of_second_bin = (
                    boundary_positions[2 * j + 1] + bin_widths[2 * j + 1] / 2
                )
                mean_of_bins = (
                    mean_of_first_bin * probability_under_first_bin
                    + mean_of_second_bin * probability_under_second_bin
                ) / probability_under_bins
                self.assertLess(
                    abs(input_ensemble[j].mass - probability_under_bins),
                    probability_threshold,
                )
                self.assertLess(
                    abs(input_ensemble[j].position - mean_of_bins), position_threshold
                )

    def test_create_binning_property_preserve_ttr(self) -> None:
        """
        Tests that the binning created by the method `PlotData.create_binning()`
        preserves TTRs, that is, if the input Dirac deltas form a valid TTR, then the Dirac deltas
        of the TTR of the created binning exactly coincide with the input Dirac deltas.
        """
        probability_threshold: float = 1e-12
        position_threshold: float = 1e-12
        number_of_testcases: int = 1000
        number_of_samples: int = 10000
        gaussian_mean_range: tuple[int, int] = (-100, 100)
        gaussian_standard_deviation_range: tuple[int, int] = (1, 100)
        ttr_orders: list[int] = []
        input_ttrs: list[list[DiracDelta]] = []

        for i in range(number_of_testcases):
            gaussian_mean = np.random.uniform(*gaussian_mean_range)
            gaussian_standard_deviation = np.random.uniform(
                *gaussian_standard_deviation_range
            )
            dirac_delta_positions = np.random.normal(
                gaussian_mean, gaussian_standard_deviation, number_of_samples
            )
            dirac_delta_masses = [1 / number_of_samples] * number_of_samples
            ttr_order = random.sample(range(4, 11), 1)[0]

            input_dirac_deltas = [
                DiracDelta(position, mass=mass)
                for position, mass in zip(dirac_delta_positions, dirac_delta_masses)
            ]

            input_ttr_dirac_deltas = dirac_deltas_to_ttr(input_dirac_deltas, ttr_order)

            if len(input_ttr_dirac_deltas) != 2**ttr_order:
                continue
            ttr_orders.append(ttr_order)
            input_ttrs.append(input_ttr_dirac_deltas)

        for i, input_ttr in enumerate(input_ttrs):
            exponent = ttr_orders[i]

            boundary_positions, bin_widths, bin_heights = PlotData.create_binning(
                input_ttr, exponent, True
            )

            binning_dirac_deltas: list[DiracDelta] = []
            for position, width, height in zip(
                boundary_positions[:-1], bin_widths, bin_heights
            ):
                binning_dirac_deltas.append(
                    DiracDelta(position=position + width / 2, mass=width * height)
                )

            binning_ttr = dirac_deltas_to_ttr(binning_dirac_deltas, exponent)

            self.assertEqual(len(input_ttr), len(binning_ttr))

            for input_ttr_dd, binning_ttr_dd in zip(input_ttr, binning_ttr):
                self.assertLess(
                    abs(input_ttr_dd.mass - binning_ttr_dd.mass), probability_threshold
                )
                self.assertLess(
                    abs(input_ttr_dd.position - binning_ttr_dd.position),
                    position_threshold,
                )


class TestCreateBinningGuards(unittest.TestCase):
    """Covers rejection of input that `PlotData.create_binning()` cannot bin."""

    def test_create_binning_rejects_fewer_than_two_dirac_deltas(self) -> None:
        """A lone Dirac delta leaves both extremal boundaries to be placed relative
        to a Dirac delta that does not exist. That must fail loudly rather than
        return `NaN`-valued boundaries, widths, and heights."""
        for input_ensemble in ([], [DiracDelta(2.5, mass=1.0)]):
            for use_ttr_binning in (False, True):
                with self.subTest(
                    number_of_dirac_deltas=len(input_ensemble),
                    use_ttr_binning=use_ttr_binning,
                ):
                    with self.assertRaises(ValueError) as raised:
                        PlotData.create_binning(input_ensemble, 0, use_ttr_binning)
                    self.assertIn("at least two Dirac deltas", str(raised.exception))

        with self.assertRaises(ValueError):
            PlotData._determine_boundary_positions([DiracDelta(2.5, mass=1.0)], 0, True)

    def test_smallest_plotting_resolution_bins_a_valid_ttr(self) -> None:
        """A `plotting_resolution` of 2 puts `plotting_ttr_order` at 0, which would
        reduce a value to a single Dirac delta. The binning must still come out
        finite, with the requested number of bins and unit total mass."""
        positions = np.array([-1.5, -0.5, 0.5, 1.5])
        masses = np.full(4, 0.25)
        dirac_deltas = [
            DiracDelta(position, mass=mass) for position, mass in zip(positions, masses)
        ]

        for plotting_resolution in (2, 4, 8):
            with self.subTest(plotting_resolution=plotting_resolution):
                dist = DistributionalValue(
                    dirac_deltas=[
                        DiracDelta(dd.position, mass=dd.mass) for dd in dirac_deltas
                    ]
                )
                pd = PlotData(dist, plotting_resolution=plotting_resolution)

                self.assertEqual(len(pd.masses), plotting_resolution)
                self.assertFalse(bool(np.any(np.isnan(pd.positions))))
                self.assertFalse(bool(np.any(np.isnan(pd.masses))))

                widths = pd.positions[1:] - pd.positions[:-1]
                self.assertTrue(np.all(widths > 0))
                self.assertAlmostEqual(float(np.sum(widths * pd.masses)), 1.0, places=9)


class TestCreateNonUniformBinning(unittest.TestCase):
    """Covers `PlotData.create_non_uniform_binning()`, the binning used for input
    that does not form a full valid TTR."""

    @staticmethod
    def _random_ensembles(
        mass_distribution: str, number_of_testcases: int = 500, seed: int = 20260731
    ) -> list[list[DiracDelta]]:
        """Random ensembles of Dirac deltas with strictly increasing positions. The
        generator is seeded, so a failure here is reproducible rather than a flake."""
        rng = np.random.default_rng(seed)
        ensembles: list[list[DiracDelta]] = []

        for _ in range(number_of_testcases):
            number_of_dirac_deltas = int(rng.integers(2, 400 + 1))
            positions = np.unique(
                rng.uniform(
                    rng.uniform(-100, 0),
                    rng.uniform(0, 100),
                    number_of_dirac_deltas,
                )
            )
            if len(positions) < 2:
                continue

            masses: np.ndarray
            if mass_distribution == "equal":
                masses = np.full(len(positions), 1.0)
            elif mass_distribution == "uniform":
                masses = rng.uniform(0, 1, len(positions))
            else:
                # Heavily skewed masses, i.e., a few Dirac deltas hold almost all
                # of the probability mass.
                masses = rng.exponential(1.0, len(positions)) ** 3
            masses /= sum(masses)

            ensembles.append(
                [
                    DiracDelta(position, mass=mass)
                    for position, mass in zip(positions, masses)
                ]
            )

        return ensembles

    def test_create_non_uniform_binning_property_one_bin_per_dirac_delta(self) -> None:
        """
        Tests that the binning created by `PlotData.create_non_uniform_binning()`
        holds exactly one bin per input Dirac delta, that the bin boundaries are
        strictly increasing (so no bin is empty or inverted), that each Dirac delta
        lies strictly inside its own bin, and that each bin carries exactly the
        mass of the Dirac delta it holds.
        """
        probability_threshold: float = 1e-15

        for mass_distribution in ("equal", "uniform", "skewed"):
            for input_ensemble in self._random_ensembles(mass_distribution):
                positions = np.array([dd.position for dd in input_ensemble])
                masses = np.array([dd.mass for dd in input_ensemble])

                (
                    boundary_positions,
                    bin_widths,
                    bin_heights,
                ) = PlotData.create_non_uniform_binning(input_ensemble)

                self.assertEqual(len(bin_widths), len(input_ensemble))
                self.assertEqual(len(boundary_positions), len(input_ensemble) + 1)
                self.assertTrue(
                    np.all(boundary_positions[1:] > boundary_positions[:-1])
                )
                self.assertTrue(np.all(positions > boundary_positions[:-1]))
                self.assertTrue(np.all(positions < boundary_positions[1:]))
                self.assertTrue(
                    np.all(
                        np.abs(bin_widths * bin_heights - masses)
                        < probability_threshold
                    )
                )

    # The boundary placement clamp in `PlotData.create_non_uniform_binning()`.
    INTERIOR_WEIGHT_CLAMP: float = PlotData.MIN_INTERIOR_WEIGHT

    @staticmethod
    def _mean_of_binning(boundary_positions: np.ndarray, masses: np.ndarray) -> float:
        """Mean of the binning that holds `masses`, one mass per bin."""
        bin_centres = (boundary_positions[1:] + boundary_positions[:-1]) / 2
        return float(np.sum(masses * bin_centres))

    @classmethod
    def _solved_interior_weight(
        cls, positions: np.ndarray, masses: np.ndarray
    ) -> float:
        """The unclamped boundary placement that makes the mean of the binning equal
        the mean of the input, recomputed here so the test can tell whether
        `PlotData.create_non_uniform_binning()` had to clamp. Recovering it from the
        returned boundaries instead would compare a rounded quotient against the
        clamp bound, which is not reliable at the bound itself."""
        target_mean = float(np.sum(positions * masses))
        mean_at_zero = cls._mean_of_binning(
            PlotData._cell_boundary_positions(positions, 0.0), masses
        )
        mean_at_one = cls._mean_of_binning(
            PlotData._cell_boundary_positions(positions, 1.0), masses
        )
        if abs(mean_at_one - mean_at_zero) < 1e-15:
            return 0.5
        return (target_mean - mean_at_zero) / (mean_at_one - mean_at_zero)

    def test_create_non_uniform_binning_property_preserves_mean(self) -> None:
        """
        Tests that the binning created by `PlotData.create_non_uniform_binning()`
        has the same total mass and the same mean as the input Dirac deltas.

        Unevenly spaced Dirac deltas can need a boundary placement very close to one
        of the two Dirac deltas the boundary separates. For a 3-Dirac-delta input,
        for example, the mirrored extremal boundaries centre the outer two bins on
        their Dirac deltas, which forces the placement to (p1 - p0) / (p2 - p0)
        regardless of the masses. `MIN_INTERIOR_WEIGHT` therefore only has to keep
        an extremal bin from collapsing to zero width, so the mean should come out
        exact here. The property tested still allows for the clamp, requiring the
        mean to be no worse than a plain midpoint placement when it does bind.
        """
        probability_threshold: float = 1e-12
        position_threshold: float = 1e-9
        clamped_ensembles: int = 0
        total_ensembles: int = 0

        for mass_distribution in ("equal", "uniform", "skewed"):
            for input_ensemble in self._random_ensembles(mass_distribution):
                positions = np.array([dd.position for dd in input_ensemble])
                masses = np.array([dd.mass for dd in input_ensemble])
                input_mean = float(np.sum(positions * masses))
                total_ensembles += 1

                (
                    boundary_positions,
                    bin_widths,
                    bin_heights,
                ) = PlotData.create_non_uniform_binning(input_ensemble)

                probabilities = bin_widths * bin_heights
                binning_mean = self._mean_of_binning(boundary_positions, masses)
                self.assertLess(
                    abs(float(np.sum(probabilities)) - 1.0), probability_threshold
                )

                interior_weight = self._solved_interior_weight(positions, masses)
                if (
                    self.INTERIOR_WEIGHT_CLAMP
                    <= interior_weight
                    <= 1 - self.INTERIOR_WEIGHT_CLAMP
                ):
                    self.assertLess(
                        abs(binning_mean - input_mean),
                        position_threshold * max(abs(input_mean), 1.0),
                    )
                    continue

                clamped_ensembles += 1
                midpoint_mean = self._mean_of_binning(
                    PlotData._cell_boundary_positions(positions, 0.5), masses
                )
                self.assertLessEqual(
                    abs(binning_mean - input_mean),
                    abs(midpoint_mean - input_mean)
                    + position_threshold * max(abs(input_mean), 1.0),
                )

        # Clamping should stay rare. If it becomes common, the placement solve has
        # regressed rather than the inputs having become pathological.
        self.assertLess(clamped_ensembles, total_ensembles // 100 + 1)

    def test_create_non_uniform_binning_preserves_local_means(self) -> None:
        """Where the Dirac delta spacing admits it, every bin is centred on the
        Dirac delta it holds, so the binning preserves each Dirac delta's own mean
        and not just the mean of the whole distribution. This is what the TTR
        binning method preserves, and what a single shared boundary placement
        cannot: one scalar can only match one moment of the whole distribution."""
        rng = np.random.default_rng(3)
        smoothly_spaced_positions = {
            "uniform spacing": np.linspace(-5.0, 5.0, 64),
            "geometric spacing": np.exp(np.linspace(-2.0, 2.5, 32)),
            "quadratic spacing": np.linspace(0.0, 4.0, 24) ** 2,
            "two Dirac deltas": np.array([0.0, 1.0]),
        }

        for label, positions in smoothly_spaced_positions.items():
            with self.subTest(spacing=label):
                masses = rng.uniform(0.5, 1.5, len(positions))
                masses /= masses.sum()
                input_ensemble = [
                    DiracDelta(position, mass=mass)
                    for position, mass in zip(positions, masses)
                ]

                (
                    boundary_positions,
                    bin_widths,
                    bin_heights,
                ) = PlotData.create_non_uniform_binning(input_ensemble)

                bin_centres = (boundary_positions[1:] + boundary_positions[:-1]) / 2
                scale = float(positions[-1] - positions[0])
                self.assertLess(
                    float(np.max(np.abs(bin_centres - positions))), 1e-12 * scale
                )
                self.assertTrue(np.all(bin_widths > 0))
                self.assertTrue(
                    np.all(np.abs(bin_widths * bin_heights - masses) < 1e-15)
                )
                self.assertAlmostEqual(
                    float(np.sum(bin_widths * bin_heights)), 1.0, places=12
                )

    def test_local_mean_boundary_positions_reports_when_unsolvable(self) -> None:
        """The recurrence that centres every bin on its Dirac delta is undamped, so
        alternating gaps drive a boundary out from between the two Dirac deltas it
        separates. There is then no solution, which must be reported rather than
        returned as an invalid binning; the caller falls back to a single shared
        boundary placement, which still preserves mass and the mean."""
        positions = np.array([0.0, 10.0, 10.5, 20.5, 21.0, 31.0, 31.5])
        self.assertIsNone(PlotData._local_mean_boundary_positions(positions))

        masses = np.full(len(positions), 1.0 / len(positions))
        input_ensemble = [
            DiracDelta(position, mass=mass) for position, mass in zip(positions, masses)
        ]
        input_mean = float(np.sum(positions * masses))

        (
            boundary_positions,
            bin_widths,
            bin_heights,
        ) = PlotData.create_non_uniform_binning(input_ensemble)

        self.assertTrue(np.all(bin_widths > 0))
        self.assertAlmostEqual(float(np.sum(bin_widths * bin_heights)), 1.0, places=12)
        self.assertLess(
            abs(self._mean_of_binning(boundary_positions, masses) - input_mean),
            1e-9 * max(abs(input_mean), 1.0),
        )

    def test_local_mean_boundary_positions_centres_every_bin(self) -> None:
        """The solved boundaries must centre each bin on its Dirac delta and stay
        strictly between the Dirac deltas they separate."""
        positions = np.array([0.0, 1.0, 2.5, 4.0, 4.5])

        boundary_positions = PlotData._local_mean_boundary_positions(positions)

        self.assertIsNotNone(boundary_positions)
        assert boundary_positions is not None
        bin_centres = (boundary_positions[1:] + boundary_positions[:-1]) / 2
        self.assertTrue(np.allclose(bin_centres, positions, atol=1e-12))
        self.assertTrue(np.all(boundary_positions[1:] > boundary_positions[:-1]))
        self.assertTrue(np.all(positions > boundary_positions[:-1]))
        self.assertTrue(np.all(positions < boundary_positions[1:]))

    def test_create_non_uniform_binning_preserves_mean_when_dirac_deltas_crowd(
        self,
    ) -> None:
        """An interior Dirac delta very close to one neighbour needs boundaries very
        close to the interval ends. A single shared placement solved for the whole
        distribution's mean puts it at ~0.0034 here, which a large clamp would block
        and perturb the mean by percent. Centring every bin on its own Dirac delta
        is solvable for this spacing, so both the local means and the mean of the
        whole distribution must come out exact."""
        positions = np.array([-38.75449121, -38.44464168, 51.59049523])
        masses = np.full(3, 1.0 / 3.0)
        input_ensemble = [
            DiracDelta(position, mass=mass) for position, mass in zip(positions, masses)
        ]
        input_mean = float(np.sum(positions * masses))

        # A single shared placement would have to sit here, well inside the clamp
        # that used to block it.
        interior_weight = self._solved_interior_weight(positions, masses)
        self.assertLess(interior_weight, 0.01)
        self.assertGreater(interior_weight, PlotData.MIN_INTERIOR_WEIGHT)

        (
            boundary_positions,
            bin_widths,
            bin_heights,
        ) = PlotData.create_non_uniform_binning(input_ensemble)

        self.assertTrue(np.all(bin_widths > 0))
        self.assertAlmostEqual(float(np.sum(bin_widths * bin_heights)), 1.0, places=12)
        # The mean must be exact, not merely close: a clamp that blocked this
        # placement left it 1.2% out.
        self.assertLess(
            abs(self._mean_of_binning(boundary_positions, masses) - input_mean),
            1e-12 * abs(input_mean),
        )
        bin_centres = (boundary_positions[1:] + boundary_positions[:-1]) / 2
        self.assertTrue(np.allclose(bin_centres, positions, atol=1e-12))

    def test_create_non_uniform_binning_rejects_fewer_than_two_dirac_deltas(
        self,
    ) -> None:
        """A lone Dirac delta has no adjacent Dirac delta to place a boundary
        against, so it has no bin width. That must fail loudly rather than return
        `NaN`-valued boundaries, widths, and heights."""
        for input_ensemble in ([], [DiracDelta(2.5, mass=1.0)]):
            with self.subTest(number_of_dirac_deltas=len(input_ensemble)):
                with self.assertRaises(ValueError) as raised:
                    PlotData.create_non_uniform_binning(input_ensemble)
                self.assertIn("at least two Dirac deltas", str(raised.exception))

        for positions in (np.array([]), np.array([2.5])):
            with self.subTest(number_of_positions=len(positions)):
                with self.assertRaises(ValueError) as raised:
                    PlotData._cell_boundary_positions(positions, 0.5)
                self.assertIn("at least two positions", str(raised.exception))

    def test_single_dirac_delta_value_plots_without_binning(self) -> None:
        """`PlotData` must not route a single-Dirac-delta value into any binning:
        it plots the Dirac delta itself, with no `NaN` in the output."""
        dist = DistributionalValue(dirac_deltas=[DiracDelta(2.5, mass=1.0)])

        pd = PlotData(dist)

        self.assertEqual(len(pd.positions), 1)
        self.assertEqual(len(pd.masses), 1)
        self.assertFalse(bool(np.any(np.isnan(pd.positions))))
        self.assertFalse(bool(np.any(np.isnan(pd.masses))))
        self.assertAlmostEqual(float(pd.positions[0]), 2.5, places=12)

    def test_create_non_uniform_binning_two_dirac_deltas(self) -> None:
        """With two Dirac deltas the mirrored extremal boundaries centre each bin
        on its Dirac delta, so the mean is exact for any boundary placement."""
        input_ensemble = [DiracDelta(0.0, mass=0.4), DiracDelta(1.0, mass=0.6)]

        (
            boundary_positions,
            bin_widths,
            bin_heights,
        ) = PlotData.create_non_uniform_binning(input_ensemble)

        probabilities = bin_widths * bin_heights
        bin_centres = (boundary_positions[1:] + boundary_positions[:-1]) / 2

        self.assertEqual(len(bin_widths), 2)
        self.assertAlmostEqual(float(np.sum(probabilities)), 1.0, places=12)
        self.assertAlmostEqual(
            float(np.sum(probabilities * bin_centres)), 0.6, places=12
        )

    # Near 1e6, no binning centres every bin on its Dirac delta, and the shared
    # placement is clamped to `MIN_INTERIOR_WEIGHT`. That puts the first internal
    # boundary 2e-14 above the first Dirac delta, which rounds onto it.
    COLLAPSING_POSITIONS: list[float] = [1e6, 1e6 + 2e-8, 1e6 + 1, 1e6 + 1 + 2e-8]
    COLLAPSING_MASSES: list[float] = [0.01, 0.98, 1e-10, 0.01]

    def test_create_non_uniform_binning_merges_a_cell_that_rounding_collapses(
        self,
    ) -> None:
        """A bin that rounding collapses to zero width would have an infinite
        density. Its Dirac delta is merged into its nearest neighbour instead, so
        every bin is finite, and the mass and the mean are kept."""
        positions = np.array(self.COLLAPSING_POSITIONS)
        masses = np.array(self.COLLAPSING_MASSES)
        input_ensemble = [
            DiracDelta(position, mass=mass) for position, mass in zip(positions, masses)
        ]
        self.assertIsNone(PlotData._local_mean_boundary_positions(positions))

        (
            boundary_positions,
            bin_widths,
            bin_heights,
        ) = PlotData.create_non_uniform_binning(input_ensemble)

        probabilities = bin_widths * bin_heights
        total_mass = float(np.sum(masses))
        input_mean = float(np.sum(positions * masses)) / total_mass
        binning_mean = self._mean_of_binning(boundary_positions, probabilities)

        self.assertEqual(len(bin_widths), len(input_ensemble) - 1)
        self.assertTrue(np.all(boundary_positions[1:] > boundary_positions[:-1]))
        self.assertTrue(np.all(np.isfinite(bin_heights)))
        self.assertAlmostEqual(float(np.sum(probabilities)), total_mass, places=12)
        self.assertLess(
            abs(binning_mean / total_mass - input_mean), 1e-12 * abs(input_mean)
        )

    def test_value_with_a_collapsing_cell_plots_finite_densities(self) -> None:
        """The same Dirac deltas, with a NaN and an unspecified representation
        type, reach the non-uniform binning through `PlotData`."""
        dist = DistributionalValue(
            dirac_deltas=[
                DiracDelta(position, mass=mass)
                for position, mass in zip(
                    self.COLLAPSING_POSITIONS, self.COLLAPSING_MASSES
                )
            ]
            + [DiracDelta(float("nan"), mass=0.01)]
        )

        pd = PlotData(dist)

        self.assertTrue(np.all(np.isfinite(pd.masses)))
        self.assertTrue(np.all(np.diff(pd.positions) > 0))
        self.assertTrue(math.isfinite(pd.max_value))


class TestPlotDataFromSamples(unittest.TestCase):
    """Tests for PlotData.from_samples() (delegates to DistributionalValue)."""

    def test_basic_finite_samples(self) -> None:
        """from_samples should produce valid PlotData from finite floats."""
        np.random.seed(42)
        samples = np.random.normal(0, 1, 1000)
        pd = PlotData.from_samples(samples)

        self.assertGreater(len(pd.positions), 0)
        self.assertGreater(len(pd.masses), 0)
        self.assertFalse(pd.dist.has_special_values)
        self.assertAlmostEqual(pd.dist.nan_dirac_delta.mass, 0.0)
        self.assertAlmostEqual(pd.dist.neg_inf_dirac_delta.mass, 0.0)
        self.assertAlmostEqual(pd.dist.pos_inf_dirac_delta.mass, 0.0)
        self.assertIsNotNone(pd.dist.mean)

    def test_density_integrates_to_approximately_one(self) -> None:
        """Histogram density should integrate to ~1 when all samples are finite."""
        np.random.seed(42)
        samples = np.random.normal(5, 2, 10_000)
        pd = PlotData.from_samples(samples)

        bin_widths = pd.positions[1:] - pd.positions[:-1]
        total_area = float(np.sum(bin_widths * pd.masses))
        self.assertAlmostEqual(total_area, 1.0, places=1)

    def test_special_value_masses(self) -> None:
        """NaN, -Inf, +Inf masses should match their proportions."""
        samples = np.array(
            [1.0, 2.0, 3.0, np.nan, np.nan, -np.inf, np.inf, np.inf, np.inf, 4.0]
        )
        pd = PlotData.from_samples(samples)

        self.assertTrue(pd.dist.has_special_values)
        self.assertAlmostEqual(pd.dist.nan_dirac_delta.mass, 2 / 10)
        self.assertAlmostEqual(pd.dist.neg_inf_dirac_delta.mass, 1 / 10)
        self.assertAlmostEqual(pd.dist.pos_inf_dirac_delta.mass, 3 / 10)

    def test_all_identical_samples(self) -> None:
        """All-identical finite samples should produce a single Dirac delta."""
        samples = np.full(100, 3.14)
        pd = PlotData.from_samples(samples)

        self.assertEqual(len(pd.positions), 1)
        self.assertAlmostEqual(pd.positions[0], 3.14)
        self.assertAlmostEqual(pd.masses[0], 1.0)

    def test_empty_samples_raises(self) -> None:
        """An empty array should raise ValueError."""
        with self.assertRaises(ValueError):
            PlotData.from_samples(np.array([]))

    def test_plotting_resolution_is_power_of_two(self) -> None:
        """The plotting_resolution should be a power of 2."""
        np.random.seed(42)
        samples = np.random.normal(0, 1, 1000)
        pd = PlotData.from_samples(samples)

        self.assertIsNotNone(pd.plotting_resolution)
        assert pd.plotting_resolution is not None
        self.assertEqual(pd.plotting_resolution & (pd.plotting_resolution - 1), 0)

    def test_custom_plotting_resolution(self) -> None:
        """plotting_resolution parameter should control the number of bins."""
        np.random.seed(42)
        samples = np.random.normal(0, 1, 1000)
        pd = PlotData.from_samples(samples, plotting_resolution=32)

        self.assertIsNotNone(pd.plotting_resolution)
        assert pd.plotting_resolution is not None
        self.assertLessEqual(pd.plotting_resolution, 32)

    def test_mean_value_close_to_sample_mean(self) -> None:
        """mean_value should be close to the mean of the finite samples."""
        np.random.seed(42)
        samples = np.random.normal(5, 1, 500)
        pd = PlotData.from_samples(samples)

        finite = samples[np.isfinite(samples)]
        self.assertIsNotNone(pd.dist.mean)
        assert pd.dist.mean is not None
        self.assertAlmostEqual(pd.dist.mean, float(np.mean(finite)), places=5)

    def test_plot_from_samples_succeeds(self) -> None:
        """plot() should succeed when given PlotData built from samples."""
        import matplotlib

        matplotlib.use("Agg")

        np.random.seed(42)
        samples = np.random.normal(0, 1, 500)
        pd = PlotData.from_samples(samples)

        result = plot(pd, path="/dev/null", save=True)
        self.assertTrue(result)

    def test_plot_from_samples_with_special_values_succeeds(self) -> None:
        """plot() should succeed for samples containing NaN/Inf."""
        import matplotlib

        matplotlib.use("Agg")

        np.random.seed(42)
        samples = np.concatenate(
            [
                np.random.normal(0, 1, 400),
                np.full(50, np.nan),
                np.full(25, np.inf),
                np.full(25, -np.inf),
            ]
        )
        pd = PlotData.from_samples(samples)

        result = plot(pd, path="/dev/null", save=True)
        self.assertTrue(result)


class TestNonTTRBinningFallback(unittest.TestCase):
    """Covers the non-uniform binning fallback used when `is_full_valid_TTR`
    is False or the TTR pipeline raises."""

    @staticmethod
    def _fixture_path(name: str) -> str:
        here = os.path.realpath(os.path.dirname(__file__))
        return os.path.join(here, name)

    def _assert_gap_free_density(self, pd: PlotData, dist: DistributionalValue) -> None:
        """The fallback binning has one bin per Dirac delta, holds no empty bins,
        and preserves the total mass and the mean of the input exactly. Call this
        after constructing the `PlotData`, which cures the `DistributionalValue`."""
        self.assertIsNotNone(pd.plotting_resolution)
        assert pd.plotting_resolution is not None
        assert pd.plotting_ttr_order is not None
        expected_number_of_bins = min(
            len(dist.finite_dirac_deltas), 2**pd.plotting_ttr_order
        )
        self.assertEqual(len(pd.positions), expected_number_of_bins + 1)
        self.assertEqual(len(pd.masses), expected_number_of_bins)

        widths = pd.positions[1:] - pd.positions[:-1]
        self.assertTrue(np.all(widths > 0))
        self.assertTrue(np.all(pd.masses > 0))

        total_area = float(np.sum(widths * pd.masses))
        self.assertAlmostEqual(total_area, 1.0, places=9)

        bin_centres = (pd.positions[1:] + pd.positions[:-1]) / 2
        binning_mean = float(np.sum(widths * pd.masses * bin_centres))
        assert dist.mean is not None
        self.assertAlmostEqual(binning_mean, dist.mean, places=9)

    def test_invalid_ttr_ux_string_falls_back_to_non_uniform_binning(self) -> None:
        """The fixture is a Ux string that is not a valid TTR. The fallback
        should produce a gap-free, non-uniform-width binning."""
        with open(self._fixture_path("invalid_ttr_ux_string.dat")) as f:
            ux_data = f.read().strip()
        dist = DistributionalValue.parse(ux_data)
        self.assertIsNotNone(dist)
        assert dist is not None
        self.assertFalse(dist.is_full_valid_TTR)

        pd = PlotData(dist)

        self._assert_gap_free_density(pd, dist)

        # The widths must genuinely vary. A uniform-width histogram is what this
        # binning replaces.
        widths = pd.positions[1:] - pd.positions[:-1]
        self.assertFalse(np.allclose(widths, widths[0]))

    def test_clustered_repeated_positions_fallback_succeeds(self) -> None:
        """Heavy duplication / tight clusters must not crash the fallback.
        Shape and density invariants still hold."""
        samples = np.concatenate(
            [
                np.full(200, 1.0),
                np.full(200, 2.0),
                np.full(200, 3.0),
                np.full(50, 2.0 + 1e-12),
                np.full(50, 2.0 - 1e-12),
            ]
        )
        pd = PlotData.from_samples(samples)

        self._assert_gap_free_density(pd, pd.dist)

    def test_raw_samples_are_reduced_to_plotting_ttr_order(self) -> None:
        """With more Dirac deltas than the plot can resolve, the fallback reduces
        them to the same number of Dirac deltas the valid-TTR path uses, and
        still holds the invariants."""
        rng = np.random.default_rng(0)
        pd = PlotData.from_samples(rng.normal(3.0, 1.5, 5000))

        assert pd.plotting_ttr_order is not None
        self.assertEqual(len(pd.masses), 2**pd.plotting_ttr_order)
        self._assert_gap_free_density(pd, pd.dist)

    def test_non_uniform_binning_leaves_no_gaps_where_uniform_would(self) -> None:
        """exp() applied to the positions of a valid normal TTR is not a valid
        TTR. Its Dirac deltas span a wide range, so a uniform-width histogram
        would leave most bins empty. The non-uniform binning leaves none."""
        positions = np.exp(np.linspace(-2.0, 2.5, 16))
        masses = np.full(16, 1.0 / 16)
        dist = DistributionalValue.from_weighted_samples(positions, masses)
        self.assertFalse(dist.is_full_valid_TTR)

        pd = PlotData(dist)

        assert pd.plotting_resolution is not None
        uniform_edges = np.linspace(
            float(positions[0]), float(positions[-1]), pd.plotting_resolution + 1
        )
        uniform_counts, _ = np.histogram(positions, bins=uniform_edges)
        self.assertGreater(int(np.sum(uniform_counts == 0)), 0)

        self._assert_gap_free_density(pd, dist)


class TestNearCoincidentDiracDeltas(unittest.TestCase):
    """Athens outputs whose Dirac deltas sit one or two ulps apart, from #51."""

    UX_STRINGS = (
        "Ux040000000000000001C063FDA1CAC0830100000002C063FDA1CAC0830126666666"
        "66666600C063FDA1CAC083005999999999999C00",
        "Ux040000000000000001C063FDA1CAC0830000000002C063FDA1CAC083007A5E353F"
        "7CED9400C063FDA1CAC082FC05A1CAC083126E80",
    )

    def test_near_coincident_dirac_deltas_plot_as_one(self) -> None:
        """They combine into one Dirac delta, and plotting does not raise."""
        for ux_string in self.UX_STRINGS:
            with self.subTest(ux_string=ux_string):
                dist = DistributionalValue.parse(ux_string)
                assert dist is not None

                pd = PlotData(dist)

                self.assertEqual(len(pd.dist.finite_dirac_deltas), 1)

    def test_a_split_with_an_empty_half_raises_value_error(self) -> None:
        """
        Splitting bins a few ulps wide can leave one half with no mass. That
        raises ValueError, which the TTR route catches to fall back.
        """
        dist = DistributionalValue.parse(self.UX_STRINGS[0])
        assert dist is not None
        boundary_positions, bin_widths, bin_heights = PlotData.create_binning(
            dist.finite_dirac_deltas, 0, False
        )

        with self.assertRaises(ValueError):
            PlotData.bin_pdf_to_ttr(boundary_positions, bin_widths, bin_heights, 1)

    def test_a_support_a_few_hundred_ulps_wide_plots(self) -> None:
        """
        The Ux string printed for c = (a + b) / (a - b) in #51 combines into
        these three Dirac deltas, a few hundred ulps apart around 1. Reducing
        them for the plot cannot split them, so neighbours are merged instead,
        down to the number of bins the plot asks for, keeping mass and mean.
        """
        dist = DistributionalValue(
            UR_type=UR_TYPE_ATHENS,
            dirac_deltas=[
                DiracDelta(float.fromhex("0x1.ffffffffffbe5p-1"), mass=0.0048875),
                DiracDelta(float.fromhex("0x1.0000000000001p+0"), mass=0.9904447),
                DiracDelta(float.fromhex("0x1.0000000000034p+0"), mass=0.0046678),
            ],
        )

        pd = PlotData(dist)

        assert pd.plotting_ttr_order is not None
        self.assertEqual(len(pd.masses), 2 ** max(pd.plotting_ttr_order, 1))
        self.assertTrue(np.all(np.isfinite(pd.masses)))
        widths = pd.positions[1:] - pd.positions[:-1]
        self.assertAlmostEqual(float(np.sum(widths * pd.masses)), 1.0, places=9)
        bin_centres = (pd.positions[1:] + pd.positions[:-1]) / 2
        binning_mean = float(np.sum(widths * pd.masses * bin_centres))
        assert dist.mean is not None
        self.assertAlmostEqual(binning_mean, dist.mean, places=9)


def dirac_deltas_to_ttr(
    dirac_deltas: list[DiracDelta], order: int, count: int = 0
) -> list[DiracDelta]:
    """
    Computes the TTR for an input ensemble of Dirac deltas.

    Args:
        dirac_deltas: Input ensemble of n Dirac deltas specified as [position, probability mass].
            Numpy array with shape (n, 2).
        order: TTR order.
        count: Counts recursion level. Always use 0.
    Returns:
        ttr: The TTR of the input bin PDF, a (2 ** `order`)-length array of Dirac deltas
            with each Dirac delta of the form np.array([position, mass]).
    """
    if count == 0:
        # Normalize mass
        normalizer_total_mass: float = 0
        for dd in dirac_deltas:
            normalizer_total_mass += dd.mass

        for dd in dirac_deltas:
            dd.mass /= normalizer_total_mass

    count += 1

    current_dirac_delta: list[DiracDelta] = []
    low_dirac_deltas: list[DiracDelta] = []
    high_dirac_deltas: list[DiracDelta] = []

    if len(dirac_deltas) > 0:
        total_mass: float = 0
        average_position: float = 0
        for dd in dirac_deltas:
            total_mass += dd.mass
            average_position += dd.position * dd.mass
        average_position /= total_mass

        current_dirac_delta = [DiracDelta(average_position, mass=total_mass)]
        low_dirac_deltas = [dd for dd in dirac_deltas if dd.position < average_position]
        high_dirac_deltas = [
            dd for dd in dirac_deltas if dd.position >= average_position
        ]

    ttr: list[DiracDelta] = []
    if order > 0:
        order -= 1

        ttr.extend(dirac_deltas_to_ttr(low_dirac_deltas, order, count))
        ttr.extend(dirac_deltas_to_ttr(high_dirac_deltas, order, count))

        return ttr

    ttr.extend(current_dirac_delta)

    return ttr


if __name__ == "__main__":
    unittest.main()
