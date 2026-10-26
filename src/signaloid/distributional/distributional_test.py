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


import csv
import os
import unittest

import numpy as np
from signaloid.distributional.dirac_delta import DiracDelta
from signaloid.distributional.distributional import (
    DistributionalValue,
)


def read_string_bytes_pairs_from_csv(
    csv_filename: str,
) -> list[tuple[str, bytes]] | None:
    """
    Reads pairs of Ux Strings and Ux Binary Data (in hex format) from a csv file

    Args:
        csv_filename: The input csv file path
    Returns:
        pairs: list of tuples of (string_value, bytearray_value)
    """
    __location__ = os.path.realpath(
        os.path.join(os.getcwd(), os.path.dirname(__file__))
    )
    csv_filepath = os.path.join(__location__, csv_filename)

    pairs: list[tuple[str, bytes]] = []
    with open(csv_filepath, "r") as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if len(row) == 2:
                string_value = row[0]
                bytearray_value = bytes.fromhex(row[1])
                pairs.append((string_value, bytearray_value))
    return pairs


def to_padded_ux_binary(ux_binary: bytes) -> bytes:
    """Convert legacy format to Ux Binary Data.

    `DistributionalValue.export` always emits the correct Ux Binary Data format
    with the 3-byte marker. A legacy byte array is converted by inserting the
    3-byte marker (the 0xF0 start byte plus two 0x00 padding bytes) after the
    8-byte particle value. A byte array already in the Ux Binary Data format
    (byte 8 == 0xF0) is returned unchanged.

    Args:
        ux_binary: A stored array in either the Ux Binary Data or the legacy
        format.

    Returns:
        The equivalent Ux Binary Data array.
    """
    if (
        len(ux_binary) >= 12
        and ux_binary[8:11] == b"\xf0\x00\x00"
        and ux_binary[11] != 0x00
    ):
        return ux_binary

    if ux_binary[8] == 0x00:
        return ux_binary[:8] + b"\xf0\x00\x00\x04" + ux_binary[9:]

    if ux_binary[8:12] == b"\xf0\x00\x00\x00":
        return ux_binary[:8] + b"\xf0\x00\x00\x04" + ux_binary[12:]

    return ux_binary[:8] + b"\xf0\x00\x00" + ux_binary[8:]


class TestUxParsing(unittest.TestCase):
    def test_parse_ux_strings_values(
        self,
        input_filename: str = "./test_ux_value_pairs.csv",
    ) -> None:
        """
        Test parsing Ux String values and converting them to Ux Binary Data.

        The encoder always emits the correct Ux Binary Data format, so each
        result is compared against the Ux Binary normalization of the stored
        byte array. This works whether the stored array is in the legacy or the
        Ux Binary layout (the corpus contains both).
        """

        ux_pairs = read_string_bytes_pairs_from_csv(input_filename)
        self.assertIsNotNone(ux_pairs)

        if ux_pairs is None:
            return

        for pair in ux_pairs:
            distValueFromUxString = DistributionalValue.parse(pair[0])
            self.assertIsNotNone(distValueFromUxString)
            if distValueFromUxString is not None:
                self.assertEqual(
                    bytes(distValueFromUxString),
                    to_padded_ux_binary(
                        pair[1]
                    ),  # Convert legacy to Ux Binary Data for comparison
                )

    def test_parse_ux_binary_values(
        self,
        input_filename: str = "./test_ux_value_pairs.csv",
    ) -> None:
        """
        Test parsing Ux Binary Data and converting it to Ux Strings.

        The corpus is a mix of legacy (unpadded) and Ux Binary (padded)
        layout byte arrays, so this exercises parsing of both layouts.
        """

        ux_pairs = read_string_bytes_pairs_from_csv(input_filename)
        self.assertIsNotNone(ux_pairs)

        if ux_pairs is None:
            return

        for pair in ux_pairs:
            distValueFromBytes = DistributionalValue.parse(pair[1])
            self.assertIsNotNone(distValueFromBytes)
            if distValueFromBytes is not None:
                self.assertEqual(str(distValueFromBytes), pair[0])

    def test_check_is_full_valid_TTR(self):
        """
        Test checking DistributionalValues for Full & Valid TTR
        """
        probabilitiesSlope = [
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.8,
            0.7,
            0.6,
            0.5,
            0.4,
            0.3,
            0.2,
            0.1,
        ]
        dist = DistributionalValue(
            dirac_deltas=[
                DiracDelta(position=position, mass=mass)
                for position, mass in enumerate(probabilitiesSlope)
            ]
        )
        self.assertFalse(dist.check_is_full_valid_TTR())

        dist = DistributionalValue(
            dirac_deltas=[DiracDelta(position=i, mass=0.1 * (i + 1)) for i in range(16)]
        )
        self.assertFalse(dist.check_is_full_valid_TTR())

        dist = DistributionalValue(
            dirac_deltas=[DiracDelta(position=i, mass=0.1) for i in range(16)]
        )
        self.assertTrue(dist.check_is_full_valid_TTR())

        gaussianMotherTTR16Positions = [
            -2.2194097942437231,
            -1.5678879053053274,
            -1.1997100902860450,
            -0.9205473016275229,
            -0.6859608829556935,
            -0.4772650338604341,
            -0.2817093825097764,
            -0.0931705533484249,
            0.0931705533484249,
            0.2817093825097764,
            0.4772650338604341,
            0.6859608829556935,
            0.9205473016275229,
            1.1997100902860450,
            1.5678879053053274,
            2.2194097942437231,
        ]
        gaussianMotherTTR16Probabilities = [
            0.0339789420851602,
            0.0520280112620429,
            0.0601091352703015,
            0.0663526532241763,
            0.0686569819948156,
            0.0714941307381180,
            0.0732557829230397,
            0.0741243625023456,
            0.0741243625023456,
            0.0732557829230397,
            0.0714941307381180,
            0.0686569819948156,
            0.0663526532241763,
            0.0601091352703015,
            0.0520280112620429,
            0.0339789420851602,
        ]
        dist = DistributionalValue(
            dirac_deltas=[
                DiracDelta(position=position, mass=mass)
                for position, mass in zip(
                    gaussianMotherTTR16Positions, gaussianMotherTTR16Probabilities
                )
            ]
        )
        self.assertTrue(dist.check_is_full_valid_TTR())

        exponentialMotherTTR16Positions = [
            0.0463101200943282,
            0.1434539976121732,
            0.2473298485428934,
            0.3589406008993048,
            0.4795601941139194,
            0.6107690295987392,
            0.7545735507649249,
            0.9136511262522184,
            1.0940956889209921,
            1.3020972163662070,
            1.5437301020205228,
            1.8320038271142292,
            2.1944919248629803,
            2.6809449590891815,
            3.4180232931306736,
            5.0000000000000000,
        ]
        exponentialMotherTTR16Probabilities = [
            0.0898043375672398,
            0.0869428363237056,
            0.0839866334859530,
            0.0809192993370894,
            0.0777683023008333,
            0.0744401048312553,
            0.0709620923605228,
            0.0672969526219585,
            0.0650216515596327,
            0.0606655024127962,
            0.0559943437573481,
            0.0508626602050525,
            0.0462377199658062,
            0.0393104949029426,
            0.0314714294791298,
            0.0183156388887342,
        ]
        dist = DistributionalValue(
            dirac_deltas=[
                DiracDelta(position=position, mass=mass)
                for position, mass in zip(
                    exponentialMotherTTR16Positions, exponentialMotherTTR16Probabilities
                )
            ]
        )
        self.assertTrue(dist.check_is_full_valid_TTR())

        laplaceMotherTTR16Positions = [
            -4.0000000000000000,
            -2.4180232931306736,
            -1.6809449590891815,
            -1.1944919248629803,
            -0.8320038271142292,
            -0.5437301020205228,
            -0.3020972163662070,
            -0.0940956889209921,
            0.0940956889209921,
            0.3020972163662070,
            0.5437301020205228,
            0.8320038271142292,
            1.1944919248629803,
            1.6809449590891815,
            2.4180232931306736,
            4.0000000000000000,
        ]
        laplaceMotherTTR16Probabilities = [
            0.0248935341839320,
            0.0427741074343744,
            0.0534285019812003,
            0.0628435769862145,
            0.0691295224912407,
            0.0761042035660443,
            0.0824529664115212,
            0.0883735869454727,
            0.0883735869454727,
            0.0824529664115212,
            0.0761042035660443,
            0.0691295224912407,
            0.0628435769862145,
            0.0534285019812003,
            0.0427741074343744,
            0.0248935341839320,
        ]
        dist = DistributionalValue(
            dirac_deltas=[
                DiracDelta(position=position, mass=mass)
                for position, mass in zip(
                    laplaceMotherTTR16Positions, laplaceMotherTTR16Probabilities
                )
            ]
        )
        self.assertTrue(dist.check_is_full_valid_TTR())

        logisticMotherTTR16Positions = [
            -4.5562387049543025,
            -2.9418421436602556,
            -2.1587856510880987,
            -1.6144447065491812,
            -1.1828706566397532,
            -0.8138506279227880,
            -0.4770709791304498,
            -0.1572607957522052,
            0.1572607957522052,
            0.4770709791304498,
            0.8138506279227880,
            1.1828706566397532,
            1.6144447065491812,
            2.1587856510880987,
            2.9418421436602556,
            4.5562387049543025,
        ]
        logisticMotherTTR16Probabilities = [
            0.0281433474818693,
            0.0475738959372374,
            0.0580005680032459,
            0.0662821885776473,
            0.0702935070589577,
            0.0743920703246041,
            0.0770073268795160,
            0.0783070957369223,
            0.0783070957369223,
            0.0770073268795160,
            0.0743920703246041,
            0.0702935070589577,
            0.0662821885776473,
            0.0580005680032459,
            0.0475738959372374,
            0.0281433474818693,
        ]
        dist = DistributionalValue(
            dirac_deltas=[
                DiracDelta(position=position, mass=mass)
                for position, mass in zip(
                    logisticMotherTTR16Positions, logisticMotherTTR16Probabilities
                )
            ]
        )
        self.assertTrue(dist.check_is_full_valid_TTR())

        gumbel1MotherTTR16Positions = [
            -1.3722373849766128,
            -0.9285915299813774,
            -0.6394795296057504,
            -0.3977265029274728,
            -0.1773165596233781,
            0.0336306411453416,
            0.2453949715783402,
            0.4637578398478052,
            0.6941280516995109,
            0.9432140395136739,
            1.2205265612985903,
            1.5396243403711020,
            1.9288304844468449,
            2.4378800713555984,
            3.1930831163570588,
            4.7879260278453658,
        ]
        gumbel1MotherTTR16Probabilities = [
            0.0476295697963972,
            0.0667881624671499,
            0.0730630564420902,
            0.0772153676471943,
            0.0768815694664325,
            0.0772987007741402,
            0.0765686001679291,
            0.0749309749136895,
            0.0721865495667472,
            0.0689189559863393,
            0.0648623956919026,
            0.0599040616210838,
            0.0552629494040690,
            0.0475407218497456,
            0.0384337406899792,
            0.0225146235151103,
        ]
        dist = DistributionalValue(
            dirac_deltas=[
                DiracDelta(position=position, mass=mass)
                for position, mass in zip(
                    gumbel1MotherTTR16Positions, gumbel1MotherTTR16Probabilities
                )
            ]
        )
        self.assertTrue(dist.check_is_full_valid_TTR())


class TestDistributionalValueFromSamples(unittest.TestCase):
    """Tests for DistributionalValue.from_samples()."""

    def test_basic_finite_samples(self) -> None:
        """from_samples should produce a valid DistributionalValue."""
        np.random.seed(42)
        samples = np.random.normal(0, 1, 100)
        dist = DistributionalValue.from_samples(samples)

        self.assertEqual(dist.UR_order, 100)
        self.assertIsNotNone(dist.mean)
        self.assertFalse(dist.has_special_values)

    def test_special_values_separated(self) -> None:
        """NaN, -Inf, +Inf should be separated after sort()."""
        samples = np.array(
            [1.0, 2.0, 3.0, np.nan, np.nan, -np.inf, np.inf, np.inf, np.inf, 4.0]
        )
        dist = DistributionalValue.from_samples(samples)
        dist.sort()

        self.assertTrue(dist.has_special_values)
        self.assertAlmostEqual(dist.nan_dirac_delta.mass, 2 / 10)
        self.assertAlmostEqual(dist.neg_inf_dirac_delta.mass, 1 / 10)
        self.assertAlmostEqual(dist.pos_inf_dirac_delta.mass, 3 / 10)

    def test_equal_mass_dirac_deltas(self) -> None:
        """Each sample should become a Dirac delta with mass 1/n."""
        samples = [1.0, 2.0, 3.0, 4.0]
        dist = DistributionalValue.from_samples(samples)

        for dd in dist.dirac_deltas:
            self.assertAlmostEqual(dd.mass, 0.25)

    def test_empty_samples_raises(self) -> None:
        """An empty array should raise ValueError."""
        with self.assertRaises(ValueError):
            DistributionalValue.from_samples(np.array([]))

    def test_all_nan_samples(self) -> None:
        """All-NaN samples should have nan_mass == 1 after sort."""
        dist = DistributionalValue.from_samples(np.full(50, np.nan))
        dist.sort()

        self.assertTrue(dist.has_special_values)
        self.assertAlmostEqual(dist.nan_dirac_delta.mass, 1.0)
        self.assertEqual(len(dist.finite_dirac_deltas), 0)

    def test_all_identical_samples(self) -> None:
        """All-identical samples should combine to one Dirac delta."""
        dist = DistributionalValue.from_samples(np.full(100, 3.14))
        dist.combine_dirac_deltas()

        finite = dist.finite_dirac_deltas
        self.assertEqual(len(finite), 1)
        self.assertAlmostEqual(finite[0].position, 3.14)
        self.assertAlmostEqual(finite[0].mass, 1.0)


class TestPositiveMassSupport(unittest.TestCase):
    """`DistributionalValue._positive_mass_support` keeps only the
    strictly-positive-mass Dirac deltas."""

    def test_drops_zero_mass_keeps_positive(self) -> None:
        """Zero-mass deltas are dropped. Positive-mass ones are kept."""
        dist = DistributionalValue(
            dirac_deltas=[
                DiracDelta(position=1.0, mass=0.4),
                DiracDelta(position=2.0, mass=0.0),
                DiracDelta(position=3.0, mass=0.6),
            ]
        )
        positions, masses = dist._positive_mass_support()
        np.testing.assert_allclose(positions, [1.0, 3.0])
        np.testing.assert_allclose(masses, [0.4, 0.6])

    def test_preserves_order_and_pairs_positions_with_masses(self) -> None:
        """Surviving positions and masses stay aligned, in original order."""
        dist = DistributionalValue(
            dirac_deltas=[
                DiracDelta(position=5.0, mass=0.5),
                DiracDelta(position=-2.0, mass=0.0),
                DiracDelta(position=1.0, mass=0.5),
            ]
        )
        positions, masses = dist._positive_mass_support()
        np.testing.assert_allclose(positions, [5.0, 1.0])
        np.testing.assert_allclose(masses, [0.5, 0.5])

    def test_all_zero_mass_returns_empty(self) -> None:
        """All-zero-mass distribution yields empty arrays."""
        dist = DistributionalValue(
            dirac_deltas=[
                DiracDelta(position=1.0, mass=0.0),
                DiracDelta(position=2.0, mass=0.0),
            ]
        )
        positions, masses = dist._positive_mass_support()
        self.assertEqual(positions.size, 0)
        self.assertEqual(masses.size, 0)

    def test_empty_distribution_returns_empty(self) -> None:
        """Empty distribution yields empty arrays."""
        dist = DistributionalValue(dirac_deltas=[])
        positions, masses = dist._positive_mass_support()
        self.assertEqual(positions.size, 0)
        self.assertEqual(masses.size, 0)

    def test_mass_is_the_only_filter(self) -> None:
        """Mass is the sole filter: a positive-mass non-finite position is
        retained (special-value handling is the caller's concern)."""
        dist = DistributionalValue(
            dirac_deltas=[
                DiracDelta(position=1.0, mass=0.5),
                DiracDelta(position=float("inf"), mass=0.5),
                DiracDelta(position=2.0, mass=0.0),
            ]
        )
        positions, masses = dist._positive_mass_support()
        self.assertEqual(positions.size, 2)
        self.assertTrue(np.isinf(positions).any())
        np.testing.assert_allclose(masses, [0.5, 0.5])


class TestCombineDiracDeltas(unittest.TestCase):
    """Combining Dirac deltas that sit closer than the merge threshold."""

    def test_coincident_dirac_deltas_combine_whatever_the_sign(self) -> None:
        """
        Two Dirac deltas one ulp apart combine into one, for a negative mean as
        for a positive one. The position is from a Ux string in #51.
        """
        for position in (-159.9259999999995, 159.9259999999995):
            with self.subTest(position=position):
                dist = DistributionalValue(
                    dirac_deltas=[
                        DiracDelta(position, mass=0.3),
                        DiracDelta(float(np.nextafter(position, np.inf)), mass=0.7),
                    ]
                )

                dist.combine_dirac_deltas()

                self.assertEqual(len(dist.finite_dirac_deltas), 1)
                self.assertAlmostEqual(dist.finite_dirac_deltas[0].mass, 1.0)


class TestDistributionalValueRange(unittest.TestCase):
    """`DistributionalValue.range` — positions.max() - positions.min()."""

    def test_single_dirac_has_zero_range(self) -> None:
        """A degenerate single-Dirac distribution has range 0.0."""
        dist = DistributionalValue(dirac_deltas=[DiracDelta(position=3.14, mass=1.0)])
        self.assertAlmostEqual(dist.range, 0.0, places=12)

    def test_multi_dirac_range_equals_max_minus_min(self) -> None:
        """Range = positions.max() - positions.min()."""
        dist = DistributionalValue(
            dirac_deltas=[
                DiracDelta(position=1.0, mass=0.3),
                DiracDelta(position=3.0, mass=0.5),
                DiracDelta(position=5.0, mass=0.2),
            ]
        )
        self.assertAlmostEqual(dist.range, 4.0, places=12)

    def test_range_independent_of_position_order(self) -> None:
        """Unsorted positions yield the same range as sorted."""
        positions = [5.0, 1.0, 3.0]
        masses = [0.2, 0.5, 0.3]
        dist = DistributionalValue(
            dirac_deltas=[
                DiracDelta(position=p, mass=m) for p, m in zip(positions, masses)
            ]
        )
        self.assertAlmostEqual(dist.range, 4.0, places=12)

    def test_range_returns_nan_when_nan_position_has_positive_mass(self) -> None:
        """NaN-position mass renders the support undefined."""
        dist = DistributionalValue(
            dirac_deltas=[
                DiracDelta(position=1.0, mass=0.3),
                DiracDelta(position=float("nan"), mass=0.4),
                DiracDelta(position=3.0, mass=0.3),
            ]
        )
        self.assertTrue(np.isnan(dist.range))

    def test_range_returns_inf_for_any_infinite_position_mass(self) -> None:
        """Any ±Inf-position mass → range is +Inf."""
        for label, special_position in (
            ("+inf", float("inf")),
            ("-inf", float("-inf")),
        ):
            with self.subTest(case=label):
                dist = DistributionalValue(
                    dirac_deltas=[
                        DiracDelta(position=1.0, mass=0.5),
                        DiracDelta(position=special_position, mass=0.5),
                    ]
                )
                self.assertEqual(dist.range, float("inf"))

        dist = DistributionalValue(
            dirac_deltas=[
                DiracDelta(position=float("-inf"), mass=0.3),
                DiracDelta(position=1.0, mass=0.4),
                DiracDelta(position=float("inf"), mass=0.3),
            ]
        )
        self.assertEqual(dist.range, float("inf"))

    def test_range_ignores_zero_mass_nan_placeholder(self) -> None:
        """Zero-mass NaN position is treated as a placeholder."""
        dist = DistributionalValue(
            dirac_deltas=[
                DiracDelta(position=1.0, mass=0.5),
                DiracDelta(position=float("nan"), mass=0.0),
                DiracDelta(position=3.0, mass=0.5),
            ]
        )
        self.assertAlmostEqual(dist.range, 2.0, places=12)

    def test_range_returns_nan_for_empty_distribution(self) -> None:
        """Empty distribution: range is NaN."""
        dist = DistributionalValue(dirac_deltas=[])
        self.assertTrue(np.isnan(dist.range))

    def test_range_ignores_zero_mass_finite_outlier(self) -> None:
        """A zero-mass finite Dirac must not widen the support."""
        dist = DistributionalValue(
            dirac_deltas=[
                DiracDelta(position=1.0, mass=0.5),
                DiracDelta(position=3.0, mass=0.5),
                DiracDelta(position=100.0, mass=0.0),
            ]
        )
        self.assertAlmostEqual(dist.range, 2.0, places=12)

    def test_range_returns_nan_for_all_zero_mass_finite(self) -> None:
        """All-zero-mass finite distribution has undefined range."""
        dist = DistributionalValue(
            dirac_deltas=[
                DiracDelta(position=1.0, mass=0.0),
                DiracDelta(position=3.0, mass=0.0),
            ]
        )
        self.assertTrue(np.isnan(dist.range))


class TestDistributionalValueQuantile(unittest.TestCase):
    """`DistributionalValue.quantile(t)` — empirical inverse CDF."""

    def test_single_dirac_returns_position_for_any_t(self) -> None:
        """Single-Dirac: every quantile equals the position."""
        dist = DistributionalValue(dirac_deltas=[DiracDelta(position=3.14, mass=1.0)])
        for t in (0.0, 0.25, 0.5, 0.75, 1.0):
            with self.subTest(t=t):
                self.assertAlmostEqual(dist.quantile(t), 3.14, places=12)

    def test_quantile_at_one_returns_max(self) -> None:
        """quantile(1.0) returns the maximum position."""
        dist = DistributionalValue(
            dirac_deltas=[
                DiracDelta(position=1.0, mass=0.5),
                DiracDelta(position=3.0, mass=0.5),
            ]
        )
        self.assertAlmostEqual(dist.quantile(1.0), 3.0, places=12)

    def test_quantile_at_zero_returns_min(self) -> None:
        """quantile(0.0) returns the minimum position."""
        dist = DistributionalValue(
            dirac_deltas=[
                DiracDelta(position=1.0, mass=0.5),
                DiracDelta(position=3.0, mass=0.5),
            ]
        )
        self.assertAlmostEqual(dist.quantile(0.0), 1.0, places=12)

    def test_quantile_known_two_point(self) -> None:
        """Hand-computed quantiles for positions [1, 3] with masses [0.3, 0.7]."""
        dist = DistributionalValue(
            dirac_deltas=[
                DiracDelta(position=1.0, mass=0.3),
                DiracDelta(position=3.0, mass=0.7),
            ]
        )
        self.assertAlmostEqual(dist.quantile(0.2), 1.0, places=12)
        self.assertAlmostEqual(dist.quantile(0.5), 3.0, places=12)

    def test_quantile_rejects_out_of_range(self) -> None:
        """t must lie in [0, 1]. Values outside raise ValueError."""
        dist = DistributionalValue(dirac_deltas=[DiracDelta(position=1.0, mass=1.0)])
        with self.assertRaises(ValueError):
            dist.quantile(-0.1)
        with self.assertRaises(ValueError):
            dist.quantile(1.1)

    def test_quantile_returns_nan_for_nan_input(self) -> None:
        """NaN input returns NaN."""
        dist = DistributionalValue(
            dirac_deltas=[
                DiracDelta(position=1.0, mass=0.5),
                DiracDelta(position=3.0, mass=0.5),
            ]
        )
        self.assertTrue(np.isnan(dist.quantile(float("nan"))))

    def test_quantile_returns_nan_for_zero_total_mass(self) -> None:
        """Zero total mass returns NaN."""
        for label, dist in (
            ("empty", DistributionalValue(dirac_deltas=[])),
            (
                "all_zero_mass",
                DistributionalValue(dirac_deltas=[DiracDelta(position=1.0, mass=0.0)]),
            ),
        ):
            with self.subTest(case=label):
                self.assertTrue(np.isnan(dist.quantile(0.5)))

    def test_quantile_ignores_zero_mass_outside_support(self) -> None:
        """Zero-mass Diracs below/above the support must not shift quantiles."""
        dist = DistributionalValue(
            dirac_deltas=[
                DiracDelta(position=-100.0, mass=0.0),
                DiracDelta(position=1.0, mass=0.5),
                DiracDelta(position=3.0, mass=0.5),
                DiracDelta(position=100.0, mass=0.0),
            ]
        )
        self.assertAlmostEqual(dist.quantile(0.0), 1.0, places=12)
        self.assertAlmostEqual(dist.quantile(1.0), 3.0, places=12)


class TestDistributionalValueMedian(unittest.TestCase):
    """`DistributionalValue.median` — quantile(0.5)."""

    def test_median_equals_quantile_half(self) -> None:
        """median is the 0.5 quantile by definition."""
        dist = DistributionalValue(
            dirac_deltas=[
                DiracDelta(position=1.0, mass=0.3),
                DiracDelta(position=3.0, mass=0.7),
            ]
        )
        self.assertAlmostEqual(dist.median, dist.quantile(0.5), places=12)

    def test_median_of_symmetric_three_point(self) -> None:
        """Symmetric {-1, 0, 1} with equal masses: median = 0."""
        dist = DistributionalValue(
            dirac_deltas=[
                DiracDelta(position=-1.0, mass=1 / 3),
                DiracDelta(position=0.0, mass=1 / 3),
                DiracDelta(position=1.0, mass=1 / 3),
            ]
        )
        self.assertAlmostEqual(dist.median, 0.0, places=12)


class TestDistributionalValueCdf(unittest.TestCase):
    """`DistributionalValue.cdf(x)`: vectorised empirical CDF."""

    def test_cdf_below_support_is_zero(self) -> None:
        """cdf(x) = 0 when x < min(positions)."""
        dist = DistributionalValue(
            dirac_deltas=[
                DiracDelta(position=1.0, mass=0.5),
                DiracDelta(position=3.0, mass=0.5),
            ]
        )
        self.assertAlmostEqual(dist.cdf(0.0), 0.0, places=12)

    def test_cdf_at_and_above_support_is_one(self) -> None:
        """cdf(x) = 1 when x ≥ max(positions) (right-continuous)."""
        dist = DistributionalValue(
            dirac_deltas=[
                DiracDelta(position=1.0, mass=0.5),
                DiracDelta(position=3.0, mass=0.5),
            ]
        )
        self.assertAlmostEqual(dist.cdf(3.0), 1.0, places=12)
        self.assertAlmostEqual(dist.cdf(4.0), 1.0, places=12)

    def test_cdf_known_three_point(self) -> None:
        """Hand-computed CDF for equal-mass positions [1, 2, 3]."""
        dist = DistributionalValue(
            dirac_deltas=[
                DiracDelta(position=1.0, mass=1 / 3),
                DiracDelta(position=2.0, mass=1 / 3),
                DiracDelta(position=3.0, mass=1 / 3),
            ]
        )
        self.assertAlmostEqual(dist.cdf(0.0), 0.0, places=12)
        self.assertAlmostEqual(dist.cdf(1.0), 1 / 3, places=12)
        self.assertAlmostEqual(dist.cdf(1.5), 1 / 3, places=12)
        self.assertAlmostEqual(dist.cdf(2.0), 2 / 3, places=12)
        self.assertAlmostEqual(dist.cdf(3.0), 1.0, places=12)

    def test_cdf_accepts_array_input(self) -> None:
        """cdf is vectorised: passing an array returns an array."""
        dist = DistributionalValue(
            dirac_deltas=[
                DiracDelta(position=1.0, mass=0.5),
                DiracDelta(position=3.0, mass=0.5),
            ]
        )
        xs = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        expected = np.array([0.0, 0.5, 0.5, 1.0, 1.0])
        np.testing.assert_allclose(dist.cdf(xs), expected, atol=1e-12)

    def test_cdf_is_monotonic_non_decreasing(self) -> None:
        """CDF is non-decreasing on a dense grid."""
        dist = DistributionalValue(
            dirac_deltas=[
                DiracDelta(position=1.0, mass=0.3),
                DiracDelta(position=2.0, mass=0.4),
                DiracDelta(position=3.0, mass=0.3),
            ]
        )
        xs = np.linspace(-1.0, 5.0, 100)
        cdfs = dist.cdf(xs)
        self.assertTrue(np.all(np.diff(cdfs) >= -1e-12))

    def test_cdf_array_matches_scalar_elementwise(self) -> None:
        """Vectorised call agrees with repeated scalar calls."""
        dist = DistributionalValue(
            dirac_deltas=[
                DiracDelta(position=1.0, mass=0.3),
                DiracDelta(position=2.0, mass=0.4),
                DiracDelta(position=3.0, mass=0.3),
            ]
        )
        xs = np.linspace(-1.0, 5.0, 100)
        array_result = dist.cdf(xs)
        for i, x in enumerate(xs):
            self.assertAlmostEqual(array_result[i], dist.cdf(float(x)), places=12)

    def test_cdf_returns_nan_for_nan_input(self) -> None:
        """NaN input returns NaN. Array NaNs propagate per slot."""
        dist = DistributionalValue(
            dirac_deltas=[
                DiracDelta(position=1.0, mass=0.5),
                DiracDelta(position=3.0, mass=0.5),
            ]
        )
        self.assertTrue(np.isnan(dist.cdf(float("nan"))))
        result = dist.cdf(np.array([1.0, float("nan"), 3.0]))
        self.assertAlmostEqual(result[0], 0.5, places=12)
        self.assertTrue(np.isnan(result[1]))
        self.assertAlmostEqual(result[2], 1.0, places=12)

    def test_cdf_returns_nan_for_zero_total_mass(self) -> None:
        """Zero total mass: scalar → NaN, array → all-NaN."""
        for label, dist in (
            ("empty", DistributionalValue(dirac_deltas=[])),
            (
                "all_zero_mass",
                DistributionalValue(dirac_deltas=[DiracDelta(position=1.0, mass=0.0)]),
            ),
        ):
            with self.subTest(case=label):
                self.assertTrue(np.isnan(dist.cdf(0.5)))
                array_result = dist.cdf(np.array([0.0, 1.0, 2.0]))
                self.assertEqual(array_result.shape, (3,))
                self.assertTrue(np.all(np.isnan(array_result)))

    def test_cdf_ignores_zero_mass_diracs(self) -> None:
        """Zero-mass Diracs — including non-finite placeholders — leave the
        CDF unchanged. Quantile/cdf do not call sort() so such placeholders
        would otherwise reach searchsorted."""
        dist = DistributionalValue(
            dirac_deltas=[
                DiracDelta(position=1.0, mass=0.5),
                DiracDelta(position=3.0, mass=0.5),
                DiracDelta(position=-100.0, mass=0.0),
                DiracDelta(position=float("nan"), mass=0.0),
                DiracDelta(position=float("inf"), mass=0.0),
            ]
        )
        self.assertAlmostEqual(dist.cdf(0.0), 0.0, places=12)
        self.assertAlmostEqual(dist.cdf(1.0), 0.5, places=12)
        self.assertAlmostEqual(dist.cdf(3.0), 1.0, places=12)
        self.assertAlmostEqual(dist.cdf(50.0), 1.0, places=12)


class TestDistributionalValueCdfQuantileInverse(unittest.TestCase):
    """Quasi-inverse contract between `cdf` and `quantile`."""

    def _two_point(self) -> DistributionalValue:
        return DistributionalValue(
            dirac_deltas=[
                DiracDelta(position=1.0, mass=0.3),
                DiracDelta(position=3.0, mass=0.7),
            ]
        )

    def test_round_trip_exact_at_support_points(self) -> None:
        """Q(F(s)) == s for every support point s."""
        dist = self._two_point()
        for s in (1.0, 3.0):
            with self.subTest(s=s):
                self.assertAlmostEqual(dist.quantile(dist.cdf(s)), s, places=12)

    def test_quantile_of_cdf_floors_to_largest_support_below(self) -> None:
        """For x between support points, Q(F(x)) floors onto the support."""
        dist = self._two_point()
        self.assertAlmostEqual(dist.quantile(dist.cdf(2.0)), 1.0, places=12)

    def test_cdf_of_quantile_exact_at_cumulative_mass_values(self) -> None:
        """F(Q(t)) == t for cumulative-mass values t."""
        dist = self._two_point()
        for t in (0.3, 1.0):
            with self.subTest(t=t):
                self.assertAlmostEqual(dist.cdf(dist.quantile(t)), t, places=12)


class TestDistributionalValueRawMoments(unittest.TestCase):
    """`DistributionalValue.raw_moments(order)` — Σ p_i · x_i^order."""

    def test_zeroth_raw_moment_is_total_mass(self) -> None:
        """raw_moments(0) = Σ p_i = 1 for a normalised distribution."""
        dist = DistributionalValue(
            dirac_deltas=[
                DiracDelta(position=1.0, mass=0.3),
                DiracDelta(position=2.0, mass=0.7),
            ]
        )
        self.assertAlmostEqual(dist.raw_moments(0), 1.0, places=12)

    def test_first_raw_moment_equals_mean(self) -> None:
        """raw_moments(1) = mean — internal consistency."""
        dist = DistributionalValue(
            dirac_deltas=[
                DiracDelta(position=1.0, mass=0.3),
                DiracDelta(position=2.0, mass=0.7),
            ]
        )
        mean = dist.mean
        assert mean is not None
        self.assertAlmostEqual(dist.raw_moments(1), mean, places=12)

    def test_raw_moments_of_dirac_match_v_pow_k(self) -> None:
        """For a Dirac at v with mass 1: raw_moments(k) = v^k."""
        v = 3.0
        dist = DistributionalValue(dirac_deltas=[DiracDelta(position=v, mass=1.0)])
        for k in (0, 1, 2, 3, 4):
            with self.subTest(k=k):
                self.assertAlmostEqual(dist.raw_moments(k), v**k, places=12)

    def test_raw_moments_with_nan_position_mass_returns_nan_for_k_ge_1(
        self,
    ) -> None:
        """NaN-position mass → NaN for k≥1, 1.0 for k=0."""
        dist = DistributionalValue(
            dirac_deltas=[
                DiracDelta(position=1.0, mass=0.3),
                DiracDelta(position=float("nan"), mass=0.4),
                DiracDelta(position=3.0, mass=0.3),
            ]
        )
        self.assertAlmostEqual(dist.raw_moments(0), 1.0, places=12)
        for k in (1, 2, 3, 4):
            with self.subTest(k=k):
                self.assertTrue(np.isnan(dist.raw_moments(k)))

    def test_raw_moments_ignores_zero_mass_nan_placeholder(self) -> None:
        """Zero-mass NaN placeholder doesn't affect the moment."""
        dist = DistributionalValue(
            dirac_deltas=[
                DiracDelta(position=1.0, mass=0.5),
                DiracDelta(position=float("nan"), mass=0.0),
                DiracDelta(position=3.0, mass=0.5),
            ]
        )
        self.assertAlmostEqual(dist.raw_moments(1), 2.0, places=12)
        self.assertAlmostEqual(dist.raw_moments(2), 5.0, places=12)

    def test_raw_moments_with_only_pos_inf_mass(self) -> None:
        """Only +Inf in support → +Inf for k≥1, 1.0 for k=0."""
        dist = DistributionalValue(
            dirac_deltas=[
                DiracDelta(position=1.0, mass=0.5),
                DiracDelta(position=float("inf"), mass=0.5),
            ]
        )
        self.assertAlmostEqual(dist.raw_moments(0), 1.0, places=12)
        for k in (1, 2, 3, 4):
            with self.subTest(k=k):
                self.assertEqual(dist.raw_moments(k), float("inf"))

    def test_raw_moments_with_only_neg_inf_alternates_sign_by_parity(
        self,
    ) -> None:
        """Only -Inf in support: +Inf for even k, -Inf for odd k."""
        dist = DistributionalValue(
            dirac_deltas=[
                DiracDelta(position=1.0, mass=0.5),
                DiracDelta(position=float("-inf"), mass=0.5),
            ]
        )
        self.assertAlmostEqual(dist.raw_moments(0), 1.0, places=12)
        self.assertEqual(dist.raw_moments(1), float("-inf"))
        self.assertEqual(dist.raw_moments(2), float("inf"))
        self.assertEqual(dist.raw_moments(3), float("-inf"))
        self.assertEqual(dist.raw_moments(4), float("inf"))

    def test_raw_moments_handles_unnormalised_masses(self) -> None:
        """Unnormalised masses [3, 7] match normalised [0.3, 0.7]."""
        unnormalised = DistributionalValue(
            dirac_deltas=[
                DiracDelta(position=1.0, mass=3.0),
                DiracDelta(position=3.0, mass=7.0),
            ]
        )
        normalised = DistributionalValue(
            dirac_deltas=[
                DiracDelta(position=1.0, mass=0.3),
                DiracDelta(position=3.0, mass=0.7),
            ]
        )
        for k in (0, 1, 2, 3, 4):
            with self.subTest(k=k):
                self.assertAlmostEqual(
                    unnormalised.raw_moments(k),
                    normalised.raw_moments(k),
                    places=12,
                )

    def test_raw_moments_with_both_pm_inf_masses(self) -> None:
        """Both ±Inf: +Inf for even k, NaN for odd k."""
        dist = DistributionalValue(
            dirac_deltas=[
                DiracDelta(position=float("-inf"), mass=0.3),
                DiracDelta(position=1.0, mass=0.4),
                DiracDelta(position=float("inf"), mass=0.3),
            ]
        )
        self.assertAlmostEqual(dist.raw_moments(0), 1.0, places=12)
        self.assertTrue(np.isnan(dist.raw_moments(1)))
        self.assertEqual(dist.raw_moments(2), float("inf"))
        self.assertTrue(np.isnan(dist.raw_moments(3)))
        self.assertEqual(dist.raw_moments(4), float("inf"))


class TestDistributionalValueCentralMoments(unittest.TestCase):
    """`DistributionalValue.central_moments(order)`: Σ p_i · (x_i − mean)^order."""

    def test_zeroth_central_moment_is_total_mass(self) -> None:
        """central_moments(0) = Σ p_i = 1."""
        dist = DistributionalValue(
            dirac_deltas=[
                DiracDelta(position=1.0, mass=0.3),
                DiracDelta(position=2.0, mass=0.7),
            ]
        )
        self.assertAlmostEqual(dist.central_moments(0), 1.0, places=12)

    def test_first_central_moment_is_zero(self) -> None:
        """central_moments(1) = 0 by the definition of mean."""
        dist = DistributionalValue(
            dirac_deltas=[
                DiracDelta(position=1.0, mass=0.3),
                DiracDelta(position=2.0, mass=0.7),
            ]
        )
        self.assertAlmostEqual(dist.central_moments(1), 0.0, places=12)

    def test_second_central_moment_equals_variance(self) -> None:
        """central_moments(2) == variance."""
        dist = DistributionalValue(
            dirac_deltas=[
                DiracDelta(position=1.0, mass=0.3),
                DiracDelta(position=2.0, mass=0.4),
                DiracDelta(position=3.0, mass=0.3),
            ]
        )
        variance = dist.variance
        assert variance is not None
        self.assertAlmostEqual(dist.central_moments(2), variance, places=12)

    def test_central_moments_of_dirac_are_zero_for_order_ge_1(self) -> None:
        """All central moments of a single Dirac are 0 for k≥1."""
        dist = DistributionalValue(dirac_deltas=[DiracDelta(position=3.0, mass=1.0)])
        for k in (1, 2, 3, 4):
            with self.subTest(k=k):
                self.assertAlmostEqual(dist.central_moments(k), 0.0, places=12)

    def test_central_moments_zeroth_is_one_even_for_non_finite_mean(self) -> None:
        """central_moments(0) = 1.0 even when the mean isn't finite."""
        for label, pos, mass in (
            ("only NaN", float("nan"), 1.0),
            ("only +Inf", float("inf"), 1.0),
            ("only -Inf", float("-inf"), 1.0),
        ):
            with self.subTest(case=label):
                dist = DistributionalValue(
                    dirac_deltas=[DiracDelta(position=pos, mass=mass)]
                )
                self.assertAlmostEqual(dist.central_moments(0), 1.0, places=12)

    def test_central_moments_ignores_zero_mass_nan_placeholder(self) -> None:
        """Zero-mass NaN placeholder doesn't affect the moment."""
        dist = DistributionalValue(
            dirac_deltas=[
                DiracDelta(position=1.0, mass=0.5),
                DiracDelta(position=float("nan"), mass=0.0),
                DiracDelta(position=3.0, mass=0.5),
            ]
        )
        self.assertAlmostEqual(dist.central_moments(1), 0.0, places=12)
        self.assertAlmostEqual(dist.central_moments(2), 1.0, places=12)

    def test_central_moments_handles_unnormalised_masses(self) -> None:
        """Unnormalised masses [3, 7] match normalised [0.3, 0.7]."""
        unnormalised = DistributionalValue(
            dirac_deltas=[
                DiracDelta(position=1.0, mass=3.0),
                DiracDelta(position=3.0, mass=7.0),
            ]
        )
        normalised = DistributionalValue(
            dirac_deltas=[
                DiracDelta(position=1.0, mass=0.3),
                DiracDelta(position=3.0, mass=0.7),
            ]
        )
        for k in (0, 1, 2, 3, 4):
            with self.subTest(k=k):
                self.assertAlmostEqual(
                    unnormalised.central_moments(k),
                    normalised.central_moments(k),
                    places=12,
                )

    def test_central_moments_and_variance_both_refuse_non_finite_mean(
        self,
    ) -> None:
        """Non-finite mean: central_moments(2) → NaN, variance → None."""
        dist = DistributionalValue(
            dirac_deltas=[
                DiracDelta(position=1.0, mass=0.5),
                DiracDelta(position=float("inf"), mass=0.5),
            ]
        )
        self.assertTrue(np.isnan(dist.central_moments(2)))
        self.assertIsNone(dist.variance)

    def test_central_moments_returns_nan_for_any_non_finite_mean(self) -> None:
        """k≥1 returns NaN for any non-finite-mean distribution."""
        cases = (
            ("NaN mass>0", [1.0, float("nan"), 3.0], [0.3, 0.4, 0.3]),
            ("only +Inf", [1.0, float("inf")], [0.5, 0.5]),
            ("only -Inf", [float("-inf"), 1.0], [0.5, 0.5]),
            ("both ±Inf", [float("-inf"), 1.0, float("inf")], [0.3, 0.4, 0.3]),
        )
        for label, positions, masses in cases:
            dist = DistributionalValue(
                dirac_deltas=[
                    DiracDelta(position=p, mass=m) for p, m in zip(positions, masses)
                ]
            )
            for k in (1, 2, 3, 4):
                with self.subTest(case=label, k=k):
                    self.assertTrue(np.isnan(dist.central_moments(k)))


class TestFromWeightedSamples(unittest.TestCase):
    def test_normalises_masses_and_round_trips_positions(self) -> None:
        """Masses are normalised to sum to 1. Positions round-trip in order."""
        dist = DistributionalValue.from_weighted_samples(
            [0.0, 1.0, 2.0], [1.0, 2.0, 1.0]
        )
        np.testing.assert_array_equal(dist.positions, np.array([0.0, 1.0, 2.0]))
        np.testing.assert_allclose(dist.masses, np.array([0.25, 0.5, 0.25]))
        self.assertAlmostEqual(float(np.sum(dist.masses)), 1.0)

    def test_already_normalised_masses_unchanged(self) -> None:
        """Inputs already summing to 1 are preserved verbatim after normalising."""
        dist = DistributionalValue.from_weighted_samples([10.0, 20.0], [0.3, 0.7])
        np.testing.assert_allclose(dist.masses, np.array([0.3, 0.7]))

    def test_empty_positions_raises(self) -> None:
        with self.assertRaises(ValueError):
            DistributionalValue.from_weighted_samples([], [1.0])

    def test_empty_masses_raises(self) -> None:
        with self.assertRaises(ValueError):
            DistributionalValue.from_weighted_samples([1.0], [])

    def test_length_mismatch_raises(self) -> None:
        with self.assertRaises(ValueError):
            DistributionalValue.from_weighted_samples([0.0, 1.0], [1.0])

    def test_non_finite_total_mass_raises(self) -> None:
        with self.assertRaises(ValueError):
            DistributionalValue.from_weighted_samples([0.0, 1.0], [float("inf"), 1.0])

    def test_zero_total_mass_raises(self) -> None:
        with self.assertRaises(ValueError):
            DistributionalValue.from_weighted_samples([0.0, 1.0], [0.0, 0.0])

    def test_negative_mass_raises(self) -> None:
        # A negative mass would normalise to a negative "probability". Reject it
        # at construction rather than build an invalid distribution.
        with self.assertRaises(ValueError):
            DistributionalValue.from_weighted_samples([1.0, 2.0], [1.0, -0.5])


class TestUxBinaryFormatDetection(unittest.TestCase):
    """Ux Binary Data format and legacy format layout detection and generation.

    The Ux Binary Data format inserts a 3-byte marker (a 0xF0 start byte
    plus two 0x00 padding bytes) between the particle value and the
    representation type. `parse` accepts both layouts. `export`/`bytes`
    always emit the correct Ux Binary Data format.
    """

    # A legacy format: particle value, then the
    # representation type byte 0x00 at offset 8 (!= 0xF0).
    LEGACY_FORMAT_HEX = (
        "09168733bf9ad93f000100000000000000c7c72324c19ad93f"
        "01000000c7c72324c19ad93f0000000000000080"
    )

    def _padded_format_hex(self) -> str:
        """The LEGACY_FORMAT_HEX value rewritten in the Ux Binary layout."""
        return self.LEGACY_FORMAT_HEX[:16] + "f0000004" + self.LEGACY_FORMAT_HEX[18:]

    def test_export_writes_format_marker(self) -> None:
        """`bytes(dist)` places 0xF0 0x00 0x00 right after the particle."""
        dist = DistributionalValue(
            particle_value=1.5,
            UR_type=4,
            dirac_deltas=[DiracDelta(position=1.5, mass=1.0)],
        )
        encoded = bytes(dist)

        # 8-byte particle value, then the 3-byte marker.
        self.assertEqual(encoded[8:11], b"\xf0\x00\x00")
        # Representation type now lives at offset 11.
        self.assertEqual(encoded[11], 4)

    def test_export_round_trips_through_parse(self) -> None:
        """A value survives export -> parse unchanged (Ux Binary layout)."""
        dist = DistributionalValue(
            particle_value=-2.25,
            UR_type=0,
            dirac_deltas=[
                DiracDelta(position=-2.25, mass=0.5),
                DiracDelta(position=3.5, mass=0.5),
            ],
        )
        parsed = DistributionalValue.parse(bytes(dist))

        self.assertIsNotNone(parsed)
        assert parsed is not None
        self.assertEqual(parsed.particle_value, dist.particle_value)
        self.assertEqual(parsed.UR_type, dist.UR_type if dist.UR_type != 0x00 else 0x04)
        np.testing.assert_array_equal(parsed.positions, dist.positions)
        np.testing.assert_array_equal(parsed.raw_masses, dist.raw_masses)

    def test_legacy_and_padded_layouts_parse_equal(self) -> None:
        """The same value in either layout parses to an equal result."""
        legacy = DistributionalValue.parse(bytes.fromhex(self.LEGACY_FORMAT_HEX))
        padded = DistributionalValue.parse(bytes.fromhex(self._padded_format_hex()))

        self.assertIsNotNone(legacy)
        self.assertIsNotNone(padded)
        assert legacy is not None and padded is not None
        self.assertEqual(legacy.particle_value, padded.particle_value)
        self.assertEqual(
            legacy.UR_type if legacy.UR_type != 0x00 else 0x04, padded.UR_type
        )
        self.assertEqual(legacy.UR_order, padded.UR_order)
        np.testing.assert_array_equal(legacy.positions, padded.positions)
        np.testing.assert_array_equal(legacy.raw_masses, padded.raw_masses)

    def test_export_of_legacy_input_matches_padded_layout(self) -> None:
        """Re-encoding a legacy input yields the Ux Binary layout bytes."""
        parsed = DistributionalValue.parse(bytes.fromhex(self.LEGACY_FORMAT_HEX))
        self.assertIsNotNone(parsed)
        assert parsed is not None
        self.assertEqual(bytes(parsed), bytes.fromhex(self._padded_format_hex()))

    def test_too_short_buffer_returns_none(self) -> None:
        """Buffers too short to detect the layout return None."""
        self.assertIsNone(DistributionalValue.parse(b""))
        self.assertIsNone(DistributionalValue.parse(bytes(8)))


class TestDistributionalValueInverseCdf(unittest.TestCase):
    """Tests for the interpolating, array-capable ``inverse_cdf`` (folded in
    from the former ``distribution_functions`` module)."""

    def _equal_mass(self) -> DistributionalValue:
        """Four equally-weighted points at 0, 1, 2, 3."""
        return DistributionalValue.from_weighted_samples(
            [0.0, 1.0, 2.0, 3.0], [1.0, 1.0, 1.0, 1.0]
        )

    def test_weighted_branch_interpolates(self) -> None:
        """Weighted branch: interp(0.5, [.25,.5,.75,1], [0,1,2,3]) = 1.0."""
        self.assertAlmostEqual(self._equal_mass().inverse_cdf(0.5), 1.0)

    def test_samples_branch_uses_np_quantile(self) -> None:
        """Samples branch: np.quantile([0,1,2,3], 0.5) = 1.5."""
        self.assertAlmostEqual(
            self._equal_mass().inverse_cdf(0.5, treat_as_samples=True), 1.5
        )

    def test_branches_differ(self) -> None:
        """Weighted and samples branches are distinct (load-bearing for the
        reported quantiles downstream)."""
        dist = self._equal_mass()
        self.assertNotAlmostEqual(
            dist.inverse_cdf(0.5), dist.inverse_cdf(0.5, treat_as_samples=True)
        )

    def test_array_input_weighted_branch(self) -> None:
        """The weighted branch accepts an array and returns an array."""
        out = self._equal_mass().inverse_cdf(np.array([0.25, 0.5, 0.75, 1.0]))
        self.assertIsInstance(out, np.ndarray)
        np.testing.assert_allclose(out, np.array([0.0, 1.0, 2.0, 3.0]))

    def test_scalar_input_returns_float(self) -> None:
        """Scalar ``p`` returns a plain float."""
        self.assertIsInstance(self._equal_mass().inverse_cdf(0.5), float)

    def test_default_is_weighted_branch(self) -> None:
        """Omitting ``treat_as_samples`` selects the weighted branch."""
        dist = self._equal_mass()
        self.assertAlmostEqual(
            dist.inverse_cdf(0.5), dist.inverse_cdf(0.5, treat_as_samples=False)
        )


class TestDistributionalValueCdfTreatAsSamples(unittest.TestCase):
    """The ``treat_as_samples`` branch of ``cdf``: unweighted empirical CDF."""

    def test_weighted_and_samples_cdf_agree_on_equal_mass(self) -> None:
        """For the equal-mass distribution both branches give cdf(1.0)=0.5:
        the weighted step lookup and ``(count <= 1) / 4``."""
        dist = DistributionalValue.from_weighted_samples(
            [0.0, 1.0, 2.0, 3.0], [1.0, 1.0, 1.0, 1.0]
        )
        self.assertAlmostEqual(float(dist.cdf(1.0)), 0.5)
        self.assertAlmostEqual(float(dist.cdf(1.0, treat_as_samples=True)), 0.5)

    def test_samples_cdf_ignores_masses(self) -> None:
        """The samples branch is ``(count of positions <= x) / N``, ignoring
        masses: for [0,1,2,3] cdf(2.0)=3/4 regardless of the weights."""
        dist = DistributionalValue.from_weighted_samples(
            [0.0, 1.0, 2.0, 3.0], [0.7, 0.1, 0.1, 0.1]
        )
        self.assertAlmostEqual(float(dist.cdf(2.0, treat_as_samples=True)), 0.75)


if __name__ == "__main__":
    unittest.main()
