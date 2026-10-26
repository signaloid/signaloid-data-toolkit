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
import shutil
import tempfile
import unittest

from signaloid.distributional.dirac_delta import DiracDelta
from signaloid.distributional.distributional import DistributionalValue
from signaloid.distributional_information_plotting.plot_histogram_dirac_deltas import (
    PlotData,
)

from .plot_wrapper import plot


def read_ux_strings_from_csv(csv_filename: str) -> list[str] | None:
    """
    Reads Ux strings from a csv file

    Args:
        csv_filename: The input csv file path
    Returns:
        ux_strings: list of Ux strings
    """
    __location__ = os.path.realpath(
        os.path.join(os.getcwd(), os.path.dirname(__file__))
    )
    csv_filepath = os.path.join(__location__, csv_filename)

    ux_strings: list[str] = []
    with open(csv_filepath, "r") as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            ux_strings.append(row[0])

    return ux_strings


class TestPlotting(unittest.TestCase):
    def setUp(self) -> None:
        # Create a temporary directory for testing
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self) -> None:
        # Remove the temporary directory even if the test fails
        shutil.rmtree(self.test_dir)

    def test_plot_wrapper(
        self,
        input_filename: str = "../distributional/test_ux_value_pairs.csv",
    ) -> None:
        """
        Test parsing and plotting Ux Strings
        """
        # Read Ux strings from test data csv
        ux_strings = read_ux_strings_from_csv(input_filename)
        self.assertIsNotNone(ux_strings)

        if ux_strings is None:
            return

        # Run the save function for each filepath
        for i, ux_string in enumerate(ux_strings):
            distValue = DistributionalValue.parse(ux_string)
            self.assertIsNotNone(distValue)
            if distValue is None:
                return

            plot_data = PlotData(distValue)
            self.assertIsNotNone(plot_data, "Failed to parse data")

            filepath = os.path.join(self.test_dir, f"file{i}.png")
            plot(plot_data, path=filepath, save=True)
            self.assertTrue(
                os.path.exists(filepath), f"File was not created: {filepath}"
            )

    def test_plot_wrapper_only_special_values(self) -> None:
        """
        Test plotting Ux Strings whose whole mass sits on NaN, leaving no
        finite Dirac deltas for the main axis.
        """
        ux_strings = [
            # A single Dirac delta at NaN.
            "Ux0400000000000000017FF8000000000000000000017FF8000000000000"
            "8000000000000000",
            # 32 zero-mass finite Dirac deltas plus a NaN carrying the mass.
            "Ux0400000000000000017FF800000000000000000021C06549521C2ED7E4"
            "0000000000000000C052EC095CFD2B8C0000000000000000C0476FEFD1405E73"
            "0000000000000000C040B852AD17B1CE0000000000000000C03CD616F8ECBE72"
            "0000000000000000C034FAD4ADEC7E8B0000000000000000C02E3C55BA2FFE4E"
            "0000000000000000C02533B0C61E71E40000000000000000C01F0B1B4A60417A"
            "0000000000000000C0177BF48F5AA0370000000000000000C012278F34BA5A4F"
            "0000000000000000C00B494DF79C8EC80000000000000000C00378C8F7D45F6A"
            "0000000000000000BFF9AC2EA807783E0000000000000000BFEC9A891E8FB6AE"
            "0000000000000000BFD27AF326A7122A00000000000000003FD61B9AED96BA85"
            "00000000000000003FEFF075F92AD91F00000000000000003FFA6103C2AB7006"
            "0000000000000000400273B75BC90788"
            "0000000000000000400864B61919B4C70000000000000000400FB04B62F6F0DF"
            "000000000000000040144B54DDC4F8650000000000000000401A05E758A5C890"
            "000000000000000040205080F7A4197A0000000000000000402486C51BA801B6"
            "0000000000000000402A60914BD6BE2C000000000000000040322C3C946DC7FD"
            "0000000000000000403C5327FEBCD6FA000000000000000040484D6F78FECB8B"
            "000000000000000040533C2D4970446C0000000000000000406670243FE9FA5B"
            "00000000000000007FF80000000000008000000000000000",
        ]

        for i, ux_string in enumerate(ux_strings):
            dist_value = DistributionalValue.parse(ux_string)
            self.assertIsNotNone(dist_value)
            if dist_value is None:
                return

            plot_data = PlotData(dist_value)
            self.assertEqual(len(plot_data.positions), 0)

            filepath = os.path.join(self.test_dir, f"special{i}.png")
            plot(plot_data, path=filepath, save=True)
            self.assertTrue(
                os.path.exists(filepath), f"File was not created: {filepath}"
            )

    def test_plot_wrapper_cell_collapsed_by_rounding(self) -> None:
        """
        Test plotting a value whose non-uniform binning would, before merging,
        have a zero-width bin with an infinite density and so an infinite y limit.
        """
        positions = [1e6, 1e6 + 2e-8, 1e6 + 1, 1e6 + 1 + 2e-8]
        masses = [0.01, 0.98, 1e-10, 0.01]
        dist_value = DistributionalValue(
            dirac_deltas=[
                DiracDelta(position, mass=mass)
                for position, mass in zip(positions, masses)
            ]
        )

        filepath = os.path.join(self.test_dir, "collapsed.png")
        plot(PlotData(dist_value), path=filepath, save=True)
        self.assertTrue(os.path.exists(filepath), f"File was not created: {filepath}")


if __name__ == "__main__":
    unittest.main()
