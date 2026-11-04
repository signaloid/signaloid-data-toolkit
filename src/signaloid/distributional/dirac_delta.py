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

# Use this multiplier to convert to/from floating point to fixed point masses
FIXED_POINT_ONE = 0x8000000000000000


def combined_position_and_mass(
    position: float, mass: float, other_position: float, other_mass: float
) -> tuple[float, float]:
    """
    The position and mass of the Dirac delta that combines two, as `DiracDelta`
    addition gives it. The mass is the sum of the two masses, and the position the
    mean of the two positions weighted by them, or their midpoint when neither has
    any mass.

    The position is found by moving from the lower position towards the higher by
    the higher one's share of the mass. Unlike the weighted mean of the positions,
    which rounds to an ulp or so off even when they are equal, that stays between
    the two positions, and is exact when they are equal. Combining many Dirac
    deltas at one position therefore stays at it, rather than drifting away a
    little with each one. Starting from the lower position makes the result the
    same whichever Dirac delta comes first. Positions too far apart for their
    difference to be finite, or not finite themselves, take the weighted mean,
    which handles infinities. The positions are taken as floats, so that integer
    positions, e.g., NumPy unsigned integers, cannot overflow or wrap around.

    Args:
        position: The position of one Dirac delta.
        mass: Its mass.
        other_position: The position of the other Dirac delta.
        other_mass: Its mass.
    Returns:
        (position, mass): The position and mass of the combined Dirac delta.
    """

    low, low_mass = float(position), mass
    high, high_mass = float(other_position), other_mass
    if high < low:
        low, low_mass, high, high_mass = high, high_mass, low, low_mass

    combined_mass = low_mass + high_mass
    # Two Dirac deltas with no mass have no weighted mean, and combine at the
    # midpoint of their positions.
    high_share = high_mass / combined_mass if combined_mass != 0 else 0.5
    difference = high - low
    if math.isfinite(difference):
        if high_share <= 0.5:
            return low + difference * high_share, combined_mass
        low_share = low_mass / combined_mass
        return high - difference * low_share, combined_mass

    if combined_mass == 0:
        return (low + high) / 2, combined_mass

    return (low * low_mass + high * high_mass) / combined_mass, combined_mass


class DiracDelta:
    # Counts the changes made to the positions and masses of Dirac deltas after
    # they were built. A `DistributionalValue` keeps arrays of the positions and
    # masses of its Dirac deltas, and compares this count to tell whether they may
    # have changed since.
    change_count: int = 0

    def __init__(
        self, position: float, raw_mass: int | None = None, mass: float | None = None
    ) -> None:
        """Initializes the Dirac Delta, given a position, and a mass in either
        a fixed-point or a floating point representation. If both representations
        are given, the fixed-point one will be used, and the floating-point one
        will be ignored.

        :param position: The Dirac Delta position.
        :type position: float
        :param raw_mass: The Dirac Delta mass in 64bit fixed-point representation,
        defaults to None.
        :type raw_mass: int | None
        :param mass: The Dirac Delta mass in floating-point representation,
        defaults to None.
        :type mass: float | None
        """
        self._position: float = position
        self._raw_mass: int = 0
        self._mass: float = 0

        if raw_mass is not None:
            self._set_raw_mass(raw_mass)
        elif mass is not None:
            self._set_mass(mass)

    @classmethod
    def many(cls, positions: list[float], mass: float) -> "list[DiracDelta]":
        """Builds Dirac deltas at the given positions, all with the given mass, as
        `DiracDelta(position, mass=mass)` builds each. The mass is checked and
        converted to fixed point once, rather than for each, which for many Dirac
        deltas is about twice as fast.

        :param positions: The positions of the Dirac deltas.
        :type positions: list[float]
        :param mass: The floating-point mass of each.
        :type mass: float
        :return: The Dirac deltas, in the order of their positions.
        :rtype: list[DiracDelta]
        :raises ValueError: If the given mass is negative.
        """
        mass_and_raw_mass = cls(0.0, mass=mass)
        float_mass = mass_and_raw_mass._mass
        raw_mass = mass_and_raw_mass._raw_mass
        new = object.__new__
        dirac_deltas: list[DiracDelta] = []
        for position in positions:
            dirac_delta = new(cls)
            dirac_delta._position = position
            dirac_delta._mass = float_mass
            dirac_delta._raw_mass = raw_mass
            dirac_deltas.append(dirac_delta)

        return dirac_deltas

    @property
    def position(self) -> float:
        """The Dirac Delta position.

        :return: The position
        :rtype: float
        """
        return self._position

    @position.setter
    def position(self, value: float) -> None:
        """Sets the Dirac Delta position.

        :param value: The position to use.
        :type value: float
        """
        self._position = value
        DiracDelta.change_count += 1

    @property
    def raw_mass(self) -> int:
        """The Dirac Delta mass in 64bit fixed-point representation.

        :return: The mass
        :rtype: int
        """
        return self._raw_mass

    @raw_mass.setter
    def raw_mass(self, value: int) -> None:
        """Sets the Dirac Delta mass given a 64bit fixed-point mass.

        :param value: The 64bit fixed-point mass to use.
        :type value: int
        :raises ValueError: If the given mass is negative.
        """
        self._set_raw_mass(value)
        DiracDelta.change_count += 1

    def _set_raw_mass(self, value: int) -> None:
        """Sets the mass given a 64bit fixed-point mass, as `raw_mass` does, without
        counting a change, for building a Dirac delta."""
        if value < 0:
            raise ValueError(f"raw_mass must be non-negative; got {value}.")

        self._raw_mass = value

        # The probability mass is a fixed-point format with FIXED_POINT_ONE
        # representing 1.0. Dividing by FIXED_POINT_ONE gets the float it
        # represents.
        self._mass = value / FIXED_POINT_ONE

    @property
    def mass(self) -> float:
        """The Dirac Delta mass in floating-point representation.

        :return: The mass
        :rtype: float
        """
        return self._mass

    @mass.setter
    def mass(self, value: float) -> None:
        """Sets the Dirac Delta mass given a floating-point mass.

        :param value: The floating-point mass to use.
        :type value: float
        :raises ValueError: If the given mass is negative.
        """
        self._set_mass(value)
        DiracDelta.change_count += 1

    def _set_mass(self, value: float) -> None:
        """Sets the mass given a floating-point mass, as `mass` does, without
        counting a change, for building a Dirac delta."""
        if value < 0:
            raise ValueError(f"mass must be non-negative; got {value}.")

        self._mass = value

        # `math.isnan`, as `np.isnan` is slow on a single float and this runs for
        # every Dirac delta built.
        if math.isnan(self._mass):
            self._raw_mass = 0
        else:
            # The probability mass is a fixed-point format with FIXED_POINT_ONE
            # representing 1.0. Multiplying by FIXED_POINT_ONE gets the fixed point
            # it represents.
            self._raw_mass = int(value * FIXED_POINT_ONE)

    def __add__(self, other: "DiracDelta") -> "DiracDelta":
        """Adds two Dirac Delta generating a new one, combining the two positions,
        and masses.

        :param other: The second Dirac Delta to use for adding.
        :type other: DiracDelta
        :return: A new Dirac Delta, with combined position, and mass.
        :rtype: DiracDelta
        """
        combined_position, combined_mass = combined_position_and_mass(
            self.position, self.mass, other.position, other.mass
        )

        return DiracDelta(position=combined_position, mass=combined_mass)

    def __lt__(self, other: "DiracDelta") -> bool:
        """Checks if this Dirac Delta is less than the given Dirac Delta.

        :param other: The second Dirac Delta to use for the comparison.
        :type other: DiracDelta
        :return: `True` if this Dirac Delta is less than the given Dirac Delta,
        `False` otherwise.
        :rtype: bool
        """
        return self.position < other.position

    def __le__(self, other: "DiracDelta") -> bool:
        """Checks if this Dirac Delta is less than or equal to the given Dirac Delta.

        :param other: The second Dirac Delta to use for the comparison.
        :type other: DiracDelta
        :return: `True` if this Dirac Delta is less than or equal to the given
        Dirac Delta, `False` otherwise.
        :rtype: bool
        """
        return self.position <= other.position

    def __eq__(self, other: object) -> bool:
        """Checks if this Dirac Delta is equal to the given Dirac Delta.

        :param other: The second Dirac Delta to use for the comparison.
        :type other: DiracDelta
        :return: `True` if this Dirac Delta is equal to the given Dirac Delta,
        `False` otherwise.
        :rtype: bool
        """
        if not isinstance(other, DiracDelta):
            return NotImplemented
        return self.position == other.position

    def __ne__(self, other: object) -> bool:
        """Checks if this Dirac Delta is not equal to the given Dirac Delta.

        :param other: The second Dirac Delta to use for the comparison.
        :type other: DiracDelta
        :return: `True` if this Dirac Delta is not equal to the given Dirac Delta,
        `False` otherwise.
        :rtype: bool
        """
        if not isinstance(other, DiracDelta):
            return NotImplemented
        return self.position != other.position

    def __ge__(self, other: "DiracDelta") -> bool:
        """Checks if this Dirac Delta is greater than or equal to the given Dirac Delta.

        :param other: The second Dirac Delta to use for the comparison.
        :type other: DiracDelta
        :return: `True` if this Dirac Delta is greater than the given Dirac Delta,
        `False` otherwise.
        :rtype: bool
        """
        return self.position >= other.position

    def __gt__(self, other: "DiracDelta") -> bool:
        """Checks if this Dirac Delta is greater than or equal to the given Dirac Delta.

        :param other: The second Dirac Delta to use for the comparison.
        :type other: DiracDelta
        :return: `True` if this Dirac Delta is greater than or equal to the given
        Dirac Delta, `False` otherwise.
        :rtype: bool
        """
        return self.position > other.position

    def __str__(self) -> str:
        """Creates a string representation of the Dirac Delta.

        :return: The string representation.
        :rtype: str
        """
        return f"[{self.position}, {self.mass}]"

    def similar(
        self: "DiracDelta",
        other: "DiracDelta",
        threshold: float,
    ) -> bool:
        """Checks if this Dirac Delta and the given one are similar to each other,
        based on the given threshold.

        :param other: The second Dirac Delta to use for the comparison.
        :type other: DiracDelta
        :param threshold: The threshold to use for the comparison
        :type threshold: float
        :return: `True` if similar, `False` otherwise.
        :rtype: bool
        """
        # As floats, so that integer positions cannot overflow or wrap around.
        return abs(float(self.position) - float(other.position)) <= threshold

    @property
    def isFinite(self) -> bool:
        """Checks if the Dirac Delta is has a finite position.

        :return: `True` is the Dirac Delta position is finite, `False` otherwise.
        :rtype: bool
        """
        return math.isfinite(self.position)
