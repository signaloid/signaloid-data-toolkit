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
Runs the plotting path of the Cardputer demo in the CircuitPython emulator that
`setup-emulator.sh` builds: Ux bytes in single precision, `parse`, `str`,
`PlotData` and `signaloid.circuitpython.plot_wrapper.plot`. The display
libraries are stubbed out, so only the numeric code runs for real.

Usage: micropython check_plotting.py <the src directory of this repository>
Exits with status 1 if any distribution fails to plot.
"""

import math
import random
import sys

import ulab

# Upstream ulab registers its NumPy as "ulab_dot_numpy", CircuitPython's as
# "ulab.numpy".
sys.modules["ulab.numpy"] = ulab.numpy
sys.path.insert(0, sys.argv[1])


class Stub:
    """Accepts any call, attribute, item or arithmetic, standing in for the
    display libraries."""

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return Stub()

    def __getattr__(self, name):
        return Stub()

    def __getitem__(self, key):
        return Stub()

    def __setitem__(self, key, value):
        pass

    def __iter__(self):
        return iter(())

    def __len__(self):
        return 0

    def append(self, item):
        pass

    def __lt__(self, other):
        return False

    __gt__ = __le__ = __ge__ = __lt__

    def __neg__(self):
        return Stub()

    def __add__(self, other):
        return Stub()

    __radd__ = __sub__ = __rsub__ = __mul__ = __rmul__ = __add__
    __truediv__ = __rtruediv__ = __floordiv__ = __rfloordiv__ = __add__


class Display:
    width = 240
    height = 135
    root_group = None


class Cartesian(Stub):
    """Stands in for uplot's Cartesian, with the pixel points that the plot
    wrapper reads back."""

    points = [(40, 120), (200, 20), (60, 100), (80, 90), (100, 80)]


board = Stub()
board.DISPLAY = Display()
cartesian = Stub()
cartesian.Cartesian = Cartesian
for name in (
    "displayio",
    "terminalio",
    "adafruit_display_text",
    "circuitpython_uplot",
    "circuitpython_uplot.plot",
    "circuitpython_uplot.scatter",
):
    sys.modules[name] = Stub()
sys.modules["board"] = board
sys.modules["circuitpython_uplot.cartesian"] = cartesian

from signaloid.circuitpython import plot_wrapper  # noqa: E402
from signaloid.distributional.dirac_delta import DiracDelta  # noqa: E402
from signaloid.distributional.distributional import (  # noqa: E402
    DistributionalValue,
)
from signaloid.distributional_information_plotting.plot_histogram_dirac_deltas import (  # noqa: E402, E501
    PlotData,
)

# The plot wrapper opens the logo of the Cardputer demo, which is not here.
plot_wrapper.open = lambda path, mode: None

# The resolution the Cardputer demo plots at.
PLOTTING_RESOLUTION = 32

# The result of `add 2.0(5) 5.0(3)` on a C0-microSD+, as the Cardputer printed it.
SAMPLE_UX_STRING = (
    "7.0Ux040000000000000010401C0000000000010000001040190000000000000"
    "1C71C71C71C71C04019666666666667038E38E38E38E3804019CCCCCCCCCCCD0"
    "555555555555540401A333333333333071C71C71C71C700401A99999999999A0"
    "8E38E38E38E3900401B0000000000000AAAAAAAAAAAAA80401B6666666666670"
    "C71C71C71C71C80401BCCCCCCCCCCCD0E38E38E38E38E00401C3333333333340"
    "E38E38E38E38E00401C99999999999A0C71C71C71C71C80401D0000000000000"
    "AAAAAAAAAAAAA80401D66666666666708E38E38E38E3900401DCCCCCCCCCCCD0"
    "71C71C71C71C700401E3333333333340555555555555540401E99999999999A0"
    "38E38E38E38E380401F00000000000001C71C71C71C71C0"
)

UR_TYPES = (0x00, 0x04, 0x06, 0x07)


def random_distributions(count):
    """Distributions of 2 to 128 Dirac deltas of every UR type, from wide to
    a few hundred single-precision ulps wide, a quarter of them with repeated
    positions."""
    random.seed(1)
    for index in range(count):
        size = random.choice((2, 3, 4, 7, 8, 16, 31, 32, 64, 128))
        centre = random.uniform(-100, 100)
        spread = 10 ** random.uniform(-3, 2)
        positions = [centre + random.uniform(-spread, spread) for _ in range(size)]
        if index % 4 == 0:
            positions[size // 2 :] = positions[: size - size // 2]
        weights = [random.uniform(0.1, 1.0) for _ in range(size)]
        total = sum(weights)
        yield DistributionalValue(
            particle_value=sum(p * w for p, w in zip(positions, weights)) / total,
            dirac_deltas=[
                DiracDelta(position=p, mass=w / total)
                for p, w in zip(positions, weights)
            ],
            UR_type=random.choice(UR_TYPES),
        )


def plot(value):
    """Plots a value as the Cardputer demo plots a result."""
    single_precision = DistributionalValue(
        particle_value=value.particle_value,
        dirac_deltas=value.dirac_deltas,
        UR_type=value.UR_type,
        double_precision=False,
    )
    # CircuitPython's bytes() calls __bytes__, MicroPython's does not.
    distribution = DistributionalValue.parse(
        single_precision.__bytes__(), double_precision=False
    )
    str(distribution)
    plot_data = PlotData(distribution, plotting_resolution=PLOTTING_RESOLUTION)
    for mass in plot_data.masses:
        if not (math.isfinite(mass) and mass >= 0):
            raise ValueError("bin height %r" % mass)
    plot_wrapper.plot(plot_data=plot_data, plot_name="check")


values = [DistributionalValue.parse(SAMPLE_UX_STRING)]
values += list(random_distributions(200))
failures = 0
for index, value in enumerate(values):
    try:
        plot(value)
    except Exception as error:
        failures += 1
        print(
            "FAIL distribution %d: %d Dirac deltas, UR type 0x%02x"
            % (index, len(value.dirac_deltas), value.UR_type)
        )
        sys.print_exception(error)

print("%d of %d distributions plotted" % (len(values) - failures, len(values)))
sys.exit(1 if failures else 0)
