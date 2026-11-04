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

import math
import unittest
from typing import Any

import numpy as np

from signaloid.distributional.dirac_delta import DiracDelta


class TestDiracDelta(unittest.TestCase):
    def test_dirac_delta_comparisons(self) -> None:
        """
        Test comparing Dirac Deltas
        """

        d1: DiracDelta = DiracDelta(position=1.2, mass=2.3)
        d2: DiracDelta = DiracDelta(position=4.5, mass=5.6)

        self.assertTrue(d1 < d2)
        self.assertTrue(d1 <= d2)
        self.assertFalse(d1 == d2)
        self.assertTrue(d1 != d2)
        self.assertFalse(d1 >= d2)
        self.assertFalse(d1 > d2)

    def test_dirac_delta_sorting(self) -> None:
        """
        Test sorting Dirac Deltas
        """

        d1 = DiracDelta(position=1.0, mass=0.1)
        d2 = DiracDelta(position=2.0, mass=0.2)
        d3 = DiracDelta(position=3.0, mass=0.3)
        d4 = DiracDelta(position=4.0, mass=0.4)

        a: list[DiracDelta] = [d2, d1, d4, d3]
        a.sort()

        b: list[DiracDelta] = [d1, d2, d3, d4]

        for dd_a, dd_b in zip(a, b):
            self.assertTrue(dd_a == dd_b)

    def test_dirac_delta_operations(self) -> None:
        """
        Test adding Dirac Deltas
        """

        d1: DiracDelta = DiracDelta(position=1.2, mass=2.3)
        d2: DiracDelta = DiracDelta(position=4.5, mass=5.6)
        d3 = d1 + d2

        combined_mass: float = 2.3 + 5.6
        combined_position: float = (1.2 * 2.3 + 4.5 * 5.6) / combined_mass
        self.assertAlmostEqual(d3.position, combined_position)
        self.assertTrue(d3.mass == combined_mass)

    def test_adding_dirac_deltas_at_one_position_stays_there(self) -> None:
        """
        However many Dirac deltas at one position are added, the sum is exactly at
        that position. The weighted mean of the positions drifts away from it.
        """
        for position in (5.0, -3.7, 0.1, 1e10):
            with self.subTest(position=position):
                total = DiracDelta(position, mass=0.001)
                for _ in range(999):
                    total += DiracDelta(position, mass=0.001)

                self.assertEqual(total.position, position)
                self.assertAlmostEqual(total.mass, 1.0)

    def test_the_sum_lies_between_the_positions(self) -> None:
        """The position of a sum lies between the positions added, also when they
        are only a few ulps apart."""
        for low, high in ((1.2, 4.5), (5.0, math.nextafter(5.0, 6.0)), (-7.0, 1e-9)):
            for low_mass, high_mass in ((0.999, 0.001), (0.5, 0.5), (1e-9, 3.0)):
                with self.subTest(low=low, high=high, low_mass=low_mass):
                    for total in (
                        DiracDelta(low, mass=low_mass)
                        + DiracDelta(high, mass=high_mass),
                        DiracDelta(high, mass=high_mass)
                        + DiracDelta(low, mass=low_mass),
                    ):
                        self.assertLessEqual(low, total.position)
                        self.assertLessEqual(total.position, high)

    def test_adding_infinite_positions(self) -> None:
        """Infinite positions add as in the weighted mean of the positions."""
        self.assertEqual(
            (DiracDelta(math.inf, mass=0.5) + DiracDelta(math.inf, mass=0.5)).position,
            math.inf,
        )
        self.assertEqual(
            (DiracDelta(-math.inf, mass=0.5) + DiracDelta(5.0, mass=0.5)).position,
            -math.inf,
        )

    def test_addition_does_not_depend_on_the_order(self) -> None:
        """`a + b` and `b + a` are the same Dirac delta, to the last bit."""
        rng = np.random.default_rng(162)
        for _ in range(1000):
            a = DiracDelta(float(rng.normal(0, 10)), mass=float(rng.uniform(0, 1)))
            b = DiracDelta(float(rng.normal(0, 10)), mass=float(rng.uniform(0, 1)))
            self.assertEqual(
                ((a + b).position, (a + b).mass), ((b + a).position, (b + a).mass)
            )

    def test_integer_positions_add_and_compare_as_floats(self) -> None:
        """NumPy integer positions, which would overflow or wrap around, are taken
        as floats."""
        for dtype, low, high in (
            (np.uint64, 3, 5),
            (np.uint8, 10, 200),
            # Their difference of 200 overflows an int8.
            (np.int8, -100, 100),
            (int, 3, 5),
        ):
            with self.subTest(dtype=dtype):
                # Deliberately not floats.
                low_position: Any = dtype(low)
                high_position: Any = dtype(high)
                lower = DiracDelta(low_position, mass=0.5)
                higher = DiracDelta(high_position, mass=0.5)

                self.assertEqual((higher + lower).position, (low + high) / 2)
                self.assertTrue(lower.similar(higher, high - low))

    def test_dirac_deltas_without_mass_add_at_their_midpoint(self) -> None:
        """Two Dirac deltas with no mass have no weighted mean, and add at the
        midpoint of their positions rather than dividing by zero."""
        total = DiracDelta(1.0, mass=0.0) + DiracDelta(3.0, mass=0.0)

        self.assertEqual((total.position, total.mass), (2.0, 0.0))

    def test_many_builds_what_building_each_would(self) -> None:
        """`DiracDelta.many` gives the same Dirac deltas as building each, and
        rejects a negative mass in the same way."""
        positions = [0.5, -2.0, 3.25]
        for mass in (0.25, 1e-6, 0.0):
            with self.subTest(mass=mass):
                self.assertEqual(
                    [
                        (dd.position, dd.mass, dd.raw_mass)
                        for dd in DiracDelta.many(positions, mass)
                    ],
                    [
                        (dd.position, dd.mass, dd.raw_mass)
                        for dd in (DiracDelta(p, mass=mass) for p in positions)
                    ],
                )
        with self.assertRaises(ValueError):
            DiracDelta.many(positions, -0.5)

    def test_changes_are_counted(self) -> None:
        """Changing a position or a mass counts a change, which values that keep
        arrays of their Dirac deltas check. Building a Dirac delta does not."""
        dirac_delta = DiracDelta(1.0, mass=0.5)
        DiracDelta(2.0, raw_mass=1)
        for change in (
            lambda: setattr(dirac_delta, "position", 2.0),
            lambda: setattr(dirac_delta, "mass", 0.25),
            lambda: setattr(dirac_delta, "raw_mass", 1),
        ):
            count = DiracDelta.change_count
            change()
            self.assertEqual(DiracDelta.change_count, count + 1)

        count = DiracDelta.change_count
        DiracDelta(3.0, mass=0.5)
        self.assertEqual(DiracDelta.change_count, count)


class TestDiracDeltaMassValidation(unittest.TestCase):
    """`DiracDelta` rejects negative masses, both at construction and via
    the `mass` / `raw_mass` setters, for the floating-point `mass` and
    fixed-point `raw_mass` inputs alike."""

    def test_rejects_negative_mass(self) -> None:
        with self.assertRaises(ValueError):
            DiracDelta(position=1.0, mass=-0.5)

    def test_rejects_negative_raw_mass(self) -> None:
        with self.assertRaises(ValueError):
            DiracDelta(position=1.0, raw_mass=-1)

    def test_accepts_zero_and_positive_mass(self) -> None:
        self.assertEqual(DiracDelta(position=1.0, mass=0.0).mass, 0.0)
        self.assertEqual(DiracDelta(position=1.0, mass=0.5).mass, 0.5)

    def test_accepts_zero_and_positive_raw_mass(self) -> None:
        self.assertEqual(DiracDelta(position=1.0, raw_mass=0).raw_mass, 0)
        self.assertEqual(DiracDelta(position=1.0, raw_mass=10).raw_mass, 10)

    def test_mass_setter_rejects_negative(self) -> None:
        dd = DiracDelta(position=1.0, mass=0.5)
        with self.assertRaises(ValueError):
            dd.mass = -0.5
        self.assertEqual(dd.mass, 0.5)

    def test_raw_mass_setter_rejects_negative(self) -> None:
        dd = DiracDelta(position=1.0, raw_mass=10)
        with self.assertRaises(ValueError):
            dd.raw_mass = -10
        self.assertEqual(dd.raw_mass, 10)

    def test_mass_setter_allows_nan(self) -> None:
        dd = DiracDelta(position=float("nan"), mass=0.0)
        dd.mass = float("nan")
        self.assertTrue(math.isnan(dd.mass))


if __name__ == "__main__":
    unittest.main()
