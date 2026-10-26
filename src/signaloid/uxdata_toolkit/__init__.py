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

"""
Signaloid Ux Data Toolkit

Toolkit for working with Signaloid Ux Data.

Ux String format specification:
    - Particle value (double in string format)
    - "Ux"                                              (   2 chars)
    - Representation type (uint8_t)                     (   2 chars)
    - Number of samples (uint64_t)                      (  16 chars) (unused)
    - Mean value of distribution (double)               (  16 chars)
    - Number of non-zero mass Dirac deltas (uint32_t)   (   8 chars)
    - Pairs of:
        - Support position (float/double)               (8/16 chars)
        - Probability mass (uint64_t)                   (  16 chars)

    Ux Binary specification:
        For more information see https://docs.signaloid.io/docs/uxhw-api/ux-data-format/

        Ux Binary Data Format:
            - Particle value (double)                           (  8 bytes)
            - Representation type (uint32_t)                    (  4 bytes)
            - Number of samples (uint64_t)                      (  8 bytes) (unused)
            - Mean value of distribution (double)               (  8 bytes)
            - Number of non-zero mass Dirac deltas (uint32_t)   (  4 bytes)
            - Pairs of:
                - Support position (float/double)               (4/8 bytes)
                - Probability mass (uint64_t)                   (  8 bytes)

        Legacy format:
            - Particle value (double)                           (  8 bytes)
            - Representation type (uint8_t)                     (  1 byte )
            - Number of samples (uint64_t)                      (  8 bytes) (unused)
            - Mean value of distribution (double)               (  8 bytes)
            - Number of non-zero mass Dirac deltas (uint32_t)   (  4 bytes)
            - Pairs of:
                - Support position (float/double)               (4/8 bytes)
                - Probability mass (uint64_t)                   (  8 bytes)

Requirements:
- pip install git+https://github.com/signaloid/signaloid-data-toolkit
"""

import argparse
import sys
import traceback
from typing import Any
from signaloid.distributional.distributional import DistributionalValue
from signaloid.distributional_information_plotting.plot_histogram_dirac_deltas import (
    PlotData,
)
from signaloid.distributional_information_plotting.plot_wrapper import plot
from signaloid.distributional_information_plotting.sample_generator import (
    sample_from_distributional_value,
)
import numpy as np


def _parse_ux_data(ux_data_raw: str) -> DistributionalValue:
    """Validate, parse, and print info for a Ux Data string.

    Args:
        ux_data_raw: Raw Ux String or Ux Binary from CLI input.

    Returns:
        The parsed DistributionalValue.

    Raises:
        SystemExit: If the input is empty.
        ValueError: If the input cannot be parsed into a DistributionalValue.
    """
    ux_data = ux_data_raw.strip()

    if not ux_data:
        print("Error: --ux-data cannot be an empty string.")
        sys.exit(1)

    dist_value = DistributionalValue.parse(ux_data)
    if dist_value is None:
        raise ValueError("Failed to parse Ux Data into DistributionalValue.")

    print("Parsed Ux Data:")
    print(f"Particle value: {dist_value.particle_value}")
    print(f"Mean: {dist_value.mean}")
    print(f"Variance: {dist_value.variance}")
    print(f"Number of Dirac deltas: {dist_value.UR_order}")
    print(f"Precision: {'64-bit' if dist_value.double_precision else '32-bit'}")

    return dist_value


def _positive_int(value: str) -> int:
    """Argparse type that enforces a positive integer (>= 1)."""
    try:
        ivalue = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid int value: '{value}'")
    if ivalue < 1:
        raise argparse.ArgumentTypeError(
            f"--num-samples must be a positive integer (>= 1), got {ivalue}"
        )
    return ivalue


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Toolkit for working with Signaloid Ux Data."
    )

    command_subparsers = parser.add_subparsers(
        dest="command",
        title="Command",
    )

    command_plot_parser(command_subparsers)
    command_sample_parser(command_subparsers)

    args = parser.parse_args(argv)
    return args


def command_plot_parser(command_subparsers: Any) -> None:
    plot_parser = command_subparsers.add_parser(
        "plot",
        help="Plot Signaloid Ux Data distributional data",
        description="Plot Signaloid Ux Data distributional data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  %(prog)s --ux-data=0.40007Ux0000000000000000013FD99AC12423C7C7000000013FD99AC12423C7C78000000000000000
  %(prog)s --ux-data=09168733bf9ad93f000100000000000000c7c72324c19ad93f01000000c7c72324c19ad93f0000000000000080
  %(prog)s -o output.png --ux-data=0.40007Ux0000000000000000013FD99AC12423C7C7000000013FD99AC12423C7C78000000000000000
        """,
    )

    plot_parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Output PNG filename (if not specified, displays plot interactively)",
    )

    plot_parser.add_argument(
        "--ux-data",
        type=str,
        required=True,
        help='Ux String or Ux Binary to plot (e.g., "90.6Ux04000000...")',
    )

    plot_parser.set_defaults(func=command_plot)


def command_plot(args: argparse.Namespace) -> None:
    """Plot a Signaloid Ux Data distributional value.

    Example usage:
        signaloid-uxdata-toolkit plot --ux-data=0.40007Ux000...
        signaloid-uxdata-toolkit plot -o output.png --ux-data=0.40007Ux000...
    """
    print("Signaloid Ux Data Plotter")
    print("=" * 50)

    try:
        dist_value = _parse_ux_data(args.ux_data)

        # Create PlotData object from the distributional value
        plot_data = PlotData(dist_value)

        # Plot the distribution using the plot wrapper function
        print("Generating plot...")

        if args.output is None:
            # Show the plot interactively
            plot(plot_data, save=False)
        else:
            plot(plot_data, path=args.output, save=True)

    except Exception as e:
        print(f"\nError parsing or plotting Ux Data: {e}")
        traceback.print_exc()
        print("\nEnsure the input has a valid Ux Data format.")
        sys.exit(1)


def command_sample_parser(command_subparsers: Any) -> None:
    sample_parser = command_subparsers.add_parser(
        "sample",
        help="Sample from Signaloid Ux Data distributional data",
        description="Sample from Signaloid Ux Data distributional data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
%(prog)s --ux-data=0.40007Ux0000000000000000013FD99AC12423C7C7000000013FD99AC12423C7C78000000000000000 --num-samples 10
%(prog)s --ux-data=09168733bf9ad93f000100000000000000c7c72324c19ad93f01000000c7c72324c19ad93f0000000000000080 --num-samples 100
%(prog)s -o samples.txt --ux-data=0.40007Ux0000000000000000013FD99AC12423C7C7000000013FD99AC12423C7C78000000000000000 --num-samples 10
        """,
    )

    sample_parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Output filename (example: samples.txt)",
    )

    sample_parser.add_argument(
        "--num-samples",
        type=_positive_int,
        required=True,
        help="Number of samples to generate (must be >= 1)",
    )

    sample_parser.add_argument(
        "--ux-data",
        type=str,
        required=True,
        help='Ux String or Ux Binary to sample from (e.g., "90.6Ux04000000...")',
    )

    sample_parser.set_defaults(func=command_sample)


def command_sample(args: argparse.Namespace) -> None:
    """Sample from a Signaloid Ux Data distributional value.

    Example usage:
        signaloid-uxdata-toolkit sample --ux-data=0.40007Ux000... --num-samples 10
        signaloid-uxdata-toolkit sample -o samples.txt --ux-data=0.40007Ux000... --num-samples 100
    """
    print("Signaloid Ux Data Sampler")
    print("=" * 50)

    try:
        dist_value = _parse_ux_data(args.ux_data)

        samples = sample_from_distributional_value(dist_value, args.num_samples)

        # Print or save samples
        if args.output:
            np.savetxt(args.output, samples, delimiter=",")
            print(f"Saved samples to: {args.output}")
        else:
            for sample in samples:
                print(sample)
    except Exception as e:
        print(f"\nError parsing or sampling Ux Data: {e}")
        traceback.print_exc()
        print("\nEnsure the input has a valid Ux Data format.")
        sys.exit(1)


def main(argv: list[str] | None = None) -> None:
    """
    Main function to parse and run the selected command.
    """
    # Parse arguments
    args: argparse.Namespace = parse_arguments(argv)

    if args.command == "plot":
        command_plot(args)
    elif args.command == "sample":
        command_sample(args)
    else:
        print("Unknown command. Use -h for help.")
        sys.exit(1)
