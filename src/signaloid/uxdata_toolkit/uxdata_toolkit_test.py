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


"""Tests for signaloid-uxdata-toolkit CLI."""

import argparse
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import signaloid.uxdata_toolkit as toolkit
from signaloid.distributional.distributional import DistributionalValue

# ============================================================================
# Constants
# ============================================================================

SAMPLE_UX_STRING = "-0.000000Ux040000000000000001BCB03C52D58D3CE400000020C0048D2279B8AFF701B1807E6239F600BFFFF1F7A03E82B602A7EB881EDAA6C0BFFAFF0EC92B7D6E0321FCB58BB7F440BFF763B747227C880386DDE1E09EAEC0BFF47A086FFA91B003BA2C11EF08E580BFF1FDCC06ED8E9703F77BE72797F440BFEF88C5501503BA0429DAF9A7420E80BFEB75E0A582D1610454636C25A09D40BFE7B77D572D585D0456D709533085C0BFE43A6FD58615450472E978D476CD40BFE0E4BE4A8177310489FC4176BD8E80BFDB5AB694DC2F6D049CBBFB39689F80BFD51A3EFE52F0A004AAD48A9FB27AC0BFCDF7B07C22983104B59D8153294540BFC1E9102D4ACC1304BCB0EDEE5C1900BFA7D59C0E0AD59404C0374A760BE0C03FA7D59C0E0AD5A204C0374A760BE0C03FC1E9102D4ACC0504BCB0EDEE5C1C803FCDF7B07C22983104B59D81532945403FD51A3EFE52F0A904AAD48A9FB277003FDB5AB694DC2F6D049CBBFB39689F803FE0E4BE4A8177310489FC4176BD8E803FE43A6FD586153C0472E978D476D0C03FE7B77D572D58680456D709533082403FEB75E0A582D1560454636C25A0A1003FEF88C5501503D10429DAF9A7420AC03FF1FDCC06ED8EA203F77BE72797F7E03FF47A086FFA91AE03BA2C11EF08DE603FF763B747227C810386DDE1E09EAB203FFAFF0EC92B7D710321FCB58BB7F4403FFFF1F7A03E82AE02A7EB881EDAA32040048D2279B8AFD801B1807E6239FD40"
SAMPLE_UX_BINARY = "09168733bf9ad93f000100000000000000c7c72324c19ad93f01000000c7c72324c19ad93f0000000000000080"
SAMPLE_UX_STRING_WITH_SPECIAL_VALUES = "nanUx0400000000000000017FF8000000000000000000063FF0000000000000155555555555550040080000000000001555555555555500000000000000000015555555555555007FF80000000000001555555555555500FFF000000000000015555555555555007FF00000000000001555555555555500"


# ============================================================================
# Unit Tests: Argument Parsing
# ============================================================================
class TestArgumentParsing(unittest.TestCase):
    """Tests for CLI argument parsing."""

    def test_plot_command_basic(self) -> None:
        args = toolkit.parse_arguments(["plot", f"--ux-data={SAMPLE_UX_STRING}"])
        self.assertEqual(args.command, "plot")
        self.assertEqual(args.ux_data, SAMPLE_UX_STRING)
        self.assertIsNone(args.output)

    def test_plot_command_custom_output(self) -> None:
        args = toolkit.parse_arguments(
            ["plot", "-o", "custom.png", f"--ux-data={SAMPLE_UX_STRING}"]
        )
        self.assertEqual(args.output, "custom.png")

    def test_sample_command_basic(self) -> None:
        num_samples = 50
        args = toolkit.parse_arguments(
            [
                "sample",
                "--num-samples",
                str(num_samples),
                f"--ux-data={SAMPLE_UX_STRING}",
            ]
        )
        self.assertEqual(args.command, "sample")
        self.assertEqual(args.ux_data, SAMPLE_UX_STRING)
        self.assertIsNone(args.output)
        self.assertEqual(args.num_samples, num_samples)

    def test_sample_command_with_num_samples(self) -> None:
        args = toolkit.parse_arguments(
            ["sample", "--num-samples", "500", f"--ux-data={SAMPLE_UX_STRING}"]
        )
        self.assertEqual(args.num_samples, 500)

    def test_sample_command_custom_output(self) -> None:
        args = toolkit.parse_arguments(
            [
                "sample",
                "-o",
                "out.txt",
                "--num-samples",
                "10",
                f"--ux-data={SAMPLE_UX_STRING}",
            ]
        )
        self.assertEqual(args.output, "out.txt")

    def test_no_command_provided(self) -> None:
        args = toolkit.parse_arguments([])
        self.assertIsNone(args.command)

    def test_plot_command_data_out(self) -> None:
        args = toolkit.parse_arguments(["plot", "--data-out", "data.out"])
        self.assertEqual(args.data_out, "data.out")
        self.assertIsNone(args.ux_data)

    def test_plot_missing_ux_data(self) -> None:
        with self.assertRaises(SystemExit):
            toolkit.parse_arguments(["plot"])

    def test_plot_ux_data_and_data_out_rejected(self) -> None:
        with self.assertRaises(SystemExit):
            toolkit.parse_arguments(
                [
                    "plot",
                    f"--ux-data={SAMPLE_UX_STRING}",
                    "--data-out",
                    "data.out",
                ]
            )

    def test_sample_missing_ux_data(self) -> None:
        with self.assertRaises(SystemExit):
            toolkit.parse_arguments(["sample", "--num-samples", "10"])

    def test_sample_missing_num_samples(self) -> None:
        with self.assertRaises(SystemExit):
            toolkit.parse_arguments(["sample", "--ux-data", SAMPLE_UX_STRING])

    def test_sample_zero_num_samples_rejected(self) -> None:
        with self.assertRaises(SystemExit):
            toolkit.parse_arguments(
                ["sample", "--num-samples", "0", "--ux-data", SAMPLE_UX_BINARY]
            )

    def test_sample_negative_num_samples_rejected(self) -> None:
        with self.assertRaises(SystemExit):
            toolkit.parse_arguments(
                ["sample", "--num-samples", "-5", "--ux-data", SAMPLE_UX_BINARY]
            )


# ============================================================================
# Unit Tests: command_plot
# ============================================================================
class TestCommandPlot(unittest.TestCase):
    """Tests for the plot command handler."""

    def _make_plot_args(self, tmp_dir: str, **overrides: object) -> argparse.Namespace:
        defaults: dict[str, object] = {
            "command": "plot",
            "ux_data": SAMPLE_UX_STRING,
            # No "data_out": command_plot must still accept a Namespace built
            # without it, as callers did before --data-out existed.
            "output": str(Path(tmp_dir) / "test_plot.png"),
        }
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    def test_plot_saves_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            args = self._make_plot_args(tmp_dir)
            toolkit.command_plot(args)
            self.assertTrue(Path(args.output).exists())

    def test_plot_with_ux_binary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            args = self._make_plot_args(tmp_dir, ux_data=SAMPLE_UX_BINARY)
            toolkit.command_plot(args)
            self.assertTrue(Path(args.output).exists())

    def test_plot_output_is_nonzero_size(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            args = self._make_plot_args(tmp_dir)
            toolkit.command_plot(args)
            self.assertGreater(Path(args.output).stat().st_size, 0)

    def test_plot_invalid_ux_data(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            args = self._make_plot_args(tmp_dir, ux_data="invalid_data")
            with self.assertRaises(SystemExit):
                toolkit.command_plot(args)

    def test_plot_empty_ux_data(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            args = self._make_plot_args(tmp_dir, ux_data="   ")
            with self.assertRaises(SystemExit):
                toolkit.command_plot(args)

    def test_plot_strips_whitespace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            args = self._make_plot_args(tmp_dir, ux_data=f"  {SAMPLE_UX_STRING}  ")
            toolkit.command_plot(args)
            self.assertTrue(Path(args.output).exists())

    def test_plot_with_special_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            args = self._make_plot_args(
                tmp_dir, ux_data=SAMPLE_UX_STRING_WITH_SPECIAL_VALUES
            )
            toolkit.command_plot(args)
            self.assertTrue(Path(args.output).exists())
            self.assertGreater(Path(args.output).stat().st_size, 0)

    def test_plot_with_data_out(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            np.random.seed(42)
            samples = np.random.normal(0, 1, 1000)
            data_out = Path(tmp_dir) / "data.out"
            data_out.write_text(
                "1234\n" + "".join(f"{s!r}\n" for s in samples.tolist()) + "inf\n"
            )
            args = self._make_plot_args(tmp_dir, ux_data=None, data_out=str(data_out))
            with redirect_stdout(io.StringIO()) as stdout:
                toolkit.command_plot(args)
            self.assertIn("Number of samples: 1001 (1 non-finite)", stdout.getvalue())
            self.assertTrue(Path(args.output).exists())
            self.assertGreater(Path(args.output).stat().st_size, 0)

    def test_plot_data_out_with_invalid_sample(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            data_out = Path(tmp_dir) / "data.out"
            data_out.write_text("1234\n1.0\nnot_a_sample\n")
            args = self._make_plot_args(tmp_dir, ux_data=None, data_out=str(data_out))
            with self.assertRaises(SystemExit):
                toolkit.command_plot(args)
            self.assertFalse(Path(args.output).exists())

    def test_plot_data_out_missing_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            args = self._make_plot_args(
                tmp_dir,
                ux_data=None,
                data_out=str(Path(tmp_dir) / "missing.out"),
            )
            with redirect_stdout(io.StringIO()) as stdout:
                with self.assertRaises(SystemExit):
                    toolkit.command_plot(args)
            self.assertIn("No such file or directory", stdout.getvalue())
            self.assertIn("Ensure the file holds", stdout.getvalue())

    def test_plot_output_failure_does_not_blame_data_out(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            data_out = Path(tmp_dir) / "data.out"
            data_out.write_text("1234\n1.0\n2.0\n3.0\n")
            args = self._make_plot_args(
                tmp_dir,
                ux_data=None,
                data_out=str(data_out),
                output=str(Path(tmp_dir) / "missing_dir" / "plot.png"),
            )
            with redirect_stdout(io.StringIO()) as stdout:
                with self.assertRaises(SystemExit):
                    toolkit.command_plot(args)
            self.assertNotIn("Ensure the file holds", stdout.getvalue())


# ============================================================================
# Unit Tests: command_sample
# ============================================================================
class TestCommandSample(unittest.TestCase):
    """Tests for the sample command handler."""

    def _make_sample_args(
        self, tmp_dir: str, **overrides: object
    ) -> argparse.Namespace:
        defaults = {
            "command": "sample",
            "ux_data": SAMPLE_UX_STRING,
            "output": str(Path(tmp_dir) / "test_samples.txt"),
            "num_samples": 100,
        }
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    def test_sample_saves_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            args = self._make_sample_args(tmp_dir)
            toolkit.command_sample(args)
            self.assertTrue(Path(args.output).exists())

    def test_sample_correct_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            args = self._make_sample_args(tmp_dir)
            toolkit.command_sample(args)
            data = np.loadtxt(args.output, delimiter=",")
            self.assertEqual(data.size, args.num_samples)

    def test_sample_with_ux_binary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            args = self._make_sample_args(
                tmp_dir, ux_data=SAMPLE_UX_BINARY, num_samples=50
            )
            toolkit.command_sample(args)
            data = np.loadtxt(args.output, delimiter=",")
            self.assertEqual(data.size, 50)

    def test_sample_values_are_finite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            args = self._make_sample_args(tmp_dir)
            toolkit.command_sample(args)
            data = np.loadtxt(args.output, delimiter=",")
            self.assertTrue(np.all(np.isfinite(data)))

    def test_sample_invalid_ux_data(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            args = self._make_sample_args(
                tmp_dir, ux_data="invalid_data", num_samples=10
            )
            with self.assertRaises(SystemExit):
                toolkit.command_sample(args)

    def test_sample_empty_ux_data(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            args = self._make_sample_args(tmp_dir, ux_data="   ", num_samples=10)
            with self.assertRaises(SystemExit):
                toolkit.command_sample(args)

    def test_sample_strips_whitespace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            args = self._make_sample_args(
                tmp_dir, ux_data=f"  {SAMPLE_UX_STRING}  ", num_samples=10
            )
            toolkit.command_sample(args)
            self.assertTrue(Path(args.output).exists())

    def test_sample_single_sample(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            args = self._make_sample_args(tmp_dir, num_samples=1)
            toolkit.command_sample(args)
            data = np.loadtxt(args.output, delimiter=",")
            self.assertEqual(data.size, 1)

    def test_sample_with_special_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            args = self._make_sample_args(
                tmp_dir,
                ux_data=SAMPLE_UX_STRING_WITH_SPECIAL_VALUES,
                num_samples=100,
            )
            toolkit.command_sample(args)
            self.assertTrue(Path(args.output).exists())
            data = np.loadtxt(args.output, delimiter=",")
            self.assertEqual(data.size, 100)


# ============================================================================
# Unit Tests: main()
# ============================================================================
class TestMain(unittest.TestCase):
    """Tests for the main dispatch function."""

    def test_main_no_command_exits(self) -> None:
        with self.assertRaises(SystemExit):
            toolkit.main(argv=[])


# ============================================================================
# Integration Tests
# ============================================================================
class TestIntegrationPlot(unittest.TestCase):
    """Integration tests for the full plot pipeline."""

    def test_plot_end_to_end_ux_string(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output = str(Path(tmp_dir) / "e2e_plot.png")
            args = toolkit.parse_arguments(
                ["plot", "-o", output, f"--ux-data={SAMPLE_UX_STRING}"]
            )
            toolkit.command_plot(args)

            output_path = Path(output)
            self.assertTrue(output_path.exists())
            self.assertGreater(output_path.stat().st_size, 0)

    def test_plot_end_to_end_ux_binary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output = str(Path(tmp_dir) / "e2e_bytes_plot.png")
            args = toolkit.parse_arguments(
                ["plot", "-o", output, "--ux-data", SAMPLE_UX_BINARY]
            )
            toolkit.command_plot(args)

            output_path = Path(output)
            self.assertTrue(output_path.exists())
            self.assertGreater(output_path.stat().st_size, 0)


class TestIntegrationSample(unittest.TestCase):
    """Integration tests for the full sample pipeline."""

    def test_sample_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output = str(Path(tmp_dir) / "e2e_samples.txt")
            n_samples = 50
            args = toolkit.parse_arguments(
                [
                    "sample",
                    "-o",
                    output,
                    "--num-samples",
                    str(n_samples),
                    f"--ux-data={SAMPLE_UX_STRING}",
                ]
            )
            toolkit.command_sample(args)

            data = np.loadtxt(output, delimiter=",")
            self.assertEqual(data.size, n_samples)
            self.assertTrue(np.all(np.isfinite(data)))

    def test_sample_default_output_filename(self) -> None:
        args = toolkit.parse_arguments(
            ["sample", "--num-samples", "1", f"--ux-data={SAMPLE_UX_STRING}"]
        )
        self.assertIsNone(args.output)

    def test_sample_statistical_sanity(self) -> None:
        """Verify that samples have the correct statistics."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            output = str(Path(tmp_dir) / "stats_samples.txt")
            n_samples = 100_000
            np.random.seed(42)
            args = toolkit.parse_arguments(
                [
                    "sample",
                    "-o",
                    output,
                    "--num-samples",
                    str(n_samples),
                    f"--ux-data={SAMPLE_UX_STRING}",
                ]
            )
            toolkit.command_sample(args)

            data = np.loadtxt(output, delimiter=",")
            self.assertEqual(data.size, n_samples)

            dist_value = DistributionalValue.parse(SAMPLE_UX_STRING)
            if dist_value is None:
                raise ValueError("Failed to parse Ux-data into DistributionalValue.")

            if dist_value.mean is not None and dist_value.variance is not None:
                threshold_mean = 5 * np.sqrt(dist_value.variance) / np.sqrt(n_samples)
                threshold_std = (
                    5 * np.sqrt(dist_value.variance) / np.sqrt(2 * n_samples)
                )

                self.assertLess(np.abs(np.mean(data) - dist_value.mean), threshold_mean)
                self.assertLess(
                    np.abs(np.std(data) - np.sqrt(dist_value.variance)),
                    threshold_std,
                )


if __name__ == "__main__":
    unittest.main()
