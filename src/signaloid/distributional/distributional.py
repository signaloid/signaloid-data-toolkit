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
import os
import sys
import re
import struct

if sys.implementation.name != "circuitpython":
    # Use numpy for accelerated computing
    import numpy as np
else:
    # Use the extended version of ulab's numpy when running on CircuitPython
    from signaloid.circuitpython.extended_ulab_numpy import np  # type: ignore[no-redef]

from signaloid.distributional.dirac_delta import DiracDelta, combined_position_and_mass

# CircuitPython has no typing module, so it gets a no-op @overload instead.
# mypy treats TYPE_CHECKING as always true, so it only sees typing.overload.
TYPE_CHECKING = False
if not TYPE_CHECKING and sys.implementation.name == "circuitpython":

    def overload(function):
        return function

else:
    from typing import Any, Iterable, overload

# The format strings used with struct.pack & struct.unpack for parsing and
# dumping `DistributionalValue` data.
STRUCT_FORMATS: dict[str, dict[str, str]] = {
    "str": {
        "particle": "",
        "UR_type": ">B",
        "sample_count": ">Q",
        "mean": ">d",
        "UR_order": ">I",
        "position_single": ">f",
        "position_double": ">d",
        "mass": ">Q",
    },
    "bytes": {
        "particle": "<d",
        "UR_type": "<B",
        "sample_count": "<Q",
        "mean": "<d",
        "UR_order": "<I",
        "position_single": "<f",
        "position_double": "<d",
        "mass": "<Q",
    },
}

# Representation type byte, as carried at offset 11 of the Ux Binary layout.
# 0x00 is not a core: it is the "not stated" value, for a value assembled in
# Python rather than read off a core (raw samples, a collapse result). Keeping
# it as the constructor default matters because `TTR_NATIVE_UR_TYPES` below
# makes the byte decide how a value is binned, and defaulting to a real core
# would silently claim a provenance the value does not have.
UR_TYPE_UNSPECIFIED = 0x00
UR_TYPE_ATHENS = 0x04
UR_TYPE_JUPITER = 0x06
UR_TYPE_ATLAS = 0x07

# Cores whose Dirac deltas are a TTR by construction. Their values should be
# binned by the TTR route even when `check_is_full_valid_TTR` says otherwise:
# curing and zero-mass dropping routinely leave such a value with a
# non-power-of-2 Dirac count, which fails the check without making the value
# any less TTR-shaped.
#
# Every other type is left to `check_is_full_valid_TTR` to route: the TTR
# binning when the value does happen to be a valid TTR, the non-uniform
# binning when it does not. That covers `UR_TYPE_UNSPECIFIED` and Jupiter,
# whose Dirac deltas are particles rather than a TTR.
TTR_NATIVE_UR_TYPES: frozenset[int] = frozenset({UR_TYPE_ATHENS, UR_TYPE_ATLAS})

# First byte of the Ux Binary Data format, used to distinguish it from the legacy format
UX_BINARY_FORMAT_MARKER = 0xF0

# By default, `DistributionalValue.combine_dirac_deltas` combines Dirac deltas closer
# together than the larger of these multiples of the size of the mean and of the
# range.
COMBINE_RELATIVE_MEAN_THRESHOLD = 1e-14
COMBINE_RELATIVE_RANGE_THRESHOLD = 1e-12


def _sum_in_order(values: np.ndarray) -> float:
    """The sum of `values`, added one at a time in order, as a loop over them would
    add them, where `np.sum` adds them in pairs."""
    return float(np.cumsum(values)[-1]) if len(values) > 0 else 0.0


class _DiracDeltaList(list):
    """
    The list of Dirac deltas of a `DistributionalValue`. It counts the changes made
    to it, so that the value can tell when the arrays it keeps of its Dirac deltas
    may no longer hold.
    """

    def __init__(self, dirac_deltas: "Iterable[DiracDelta]" = ()) -> None:
        super().__init__(dirac_deltas)
        self.change_count = 0

    def __setitem__(self, *args: "Any") -> None:
        super().__setitem__(*args)
        self.change_count += 1

    def __delitem__(self, *args: "Any") -> None:
        super().__delitem__(*args)
        self.change_count += 1

    # `list` itself takes any iterable for `+=` but only lists for `+`, which mypy
    # rejects for a subclass that keeps both.
    def __iadd__(  # type: ignore[misc]
        self, dirac_deltas: "Iterable[Any]"
    ) -> "_DiracDeltaList":
        super().__iadd__(dirac_deltas)
        self.change_count += 1
        return self

    def __imul__(self, count: "Any") -> "_DiracDeltaList":  # type: ignore[misc]
        super().__imul__(count)
        self.change_count += 1
        return self

    def append(self, *args: "Any") -> None:
        super().append(*args)
        self.change_count += 1

    def extend(self, *args: "Any") -> None:
        super().extend(*args)
        self.change_count += 1

    def insert(self, *args: "Any") -> None:
        super().insert(*args)
        self.change_count += 1

    def pop(self, *args: "Any") -> "Any":
        dirac_delta = super().pop(*args)
        self.change_count += 1
        return dirac_delta

    def remove(self, *args: "Any") -> None:
        super().remove(*args)
        self.change_count += 1

    def clear(self) -> None:
        super().clear()
        self.change_count += 1

    def sort(self, *args: "Any", **kwargs: "Any") -> None:
        super().sort(*args, **kwargs)
        self.change_count += 1

    def reverse(self) -> None:
        super().reverse()
        self.change_count += 1


class _ChangeState:
    """
    Identifies a list of Dirac deltas as it is, to tell later whether it, or a
    Dirac delta, has changed since. A change to any Dirac delta counts, in this list
    or another, as Dirac deltas can be in several lists and do not know which.
    """

    def __init__(self, dirac_deltas: _DiracDeltaList) -> None:
        self.dirac_deltas = dirac_deltas
        self.list_change_count = dirac_deltas.change_count
        self.dirac_delta_change_count = DiracDelta.change_count

    def holds_for(self, dirac_deltas: _DiracDeltaList) -> bool:
        """Whether `dirac_deltas` is the same list, and neither it nor any Dirac
        delta has changed since."""
        return (
            self.dirac_deltas is dirac_deltas
            and self.list_change_count == dirac_deltas.change_count
            and self.dirac_delta_change_count == DiracDelta.change_count
        )


class _Facts:
    """
    What a `DistributionalValue` has worked out about its Dirac deltas, which it
    keeps until they change. `None` is not yet known.
    """

    def __init__(self, dirac_deltas: _DiracDeltaList) -> None:
        self.state = _ChangeState(dirac_deltas)
        self.mean: float | None = None
        self.variance: float | None = None
        self.has_no_zero_mass: bool | None = None
        self.is_finite: bool | None = None
        self.is_sorted: bool | None = None
        self.is_cured: bool | None = None
        self.is_full_valid_TTR: bool | None = None


class _DiracDeltaArrays:
    """
    Arrays computed from a list of Dirac deltas, which a `DistributionalValue` keeps
    for as long as its Dirac deltas are unchanged, rather than recomputing them
    from every Dirac delta for every statistic. The arrays other than the
    positions and masses are computed when first needed.
    """

    def __init__(
        self,
        dirac_deltas: _DiracDeltaList,
        positions: np.ndarray | None = None,
        masses: np.ndarray | None = None,
    ) -> None:
        """
        Args:
            dirac_deltas: The Dirac deltas.
            positions: Their positions, as floats, when already known, which then
                need not be read from every Dirac delta. Given with `masses`.
            masses: Their masses, when already known.
        """
        self.state = _ChangeState(dirac_deltas)
        self.positions: np.ndarray = (
            positions
            if positions is not None
            else np.array([dd.position for dd in dirac_deltas], dtype=np.float64)
        )
        self.masses: np.ndarray = (
            masses
            if masses is not None
            else np.array([dd.mass for dd in dirac_deltas], dtype=np.float64)
        )
        self.raw_masses: np.ndarray | None = None
        self.total_mass: float | None = None
        self.sorted_positions: np.ndarray | None = None
        # The positions of the strictly-positive-mass Dirac deltas, sorted, and
        # their cumulative masses.
        self.sorted_support: tuple[np.ndarray, np.ndarray] | None = None
        # The cumulative masses over the total mass, with a leading 0.
        self.cdf_lookup: np.ndarray | None = None
        # The cumulative masses over their last.
        self.inverse_cdf_cumulative: np.ndarray | None = None


class DistributionalValue:
    def __init__(
        self,
        particle_value: float | None = None,
        UR_type: int = UR_TYPE_UNSPECIFIED,
        dirac_deltas: list[DiracDelta] | None = None,
        double_precision: bool = True,
    ) -> None:
        self.particle_value: float | None = particle_value
        self.UR_type: int = UR_type
        self._dirac_deltas = dirac_deltas if dirac_deltas is not None else []
        self.double_precision = double_precision

        """
        properties
        """
        # A mean set explicitly, e.g., from a Ux header, which is kept as it is.
        self._mean: float | None = None

        self.nan_dirac_delta: DiracDelta = DiracDelta(position=np.nan, mass=np.nan)
        self.neg_inf_dirac_delta: DiracDelta = DiracDelta(position=-np.inf, mass=np.nan)
        self.pos_inf_dirac_delta: DiracDelta = DiracDelta(position=np.inf, mass=np.nan)

        self._facts = _Facts(self._dirac_delta_list)
        self._array_cache: _DiracDeltaArrays | None = None

    @property
    def dirac_deltas(self) -> list[DiracDelta]:
        """The list of all the Dirac Deltas of the DistributionalValue.

        The value keeps a list of its own, so changing the list it was built from
        does not change it. Statistics are computed from arrays of the Dirac
        deltas, which are kept until this list, or a Dirac delta, changes.

        :return: The list of the Dirac Deltas.
        :rtype: list[DiracDelta]
        """
        return self._dirac_delta_list

    @property
    def _dirac_deltas(self) -> list[DiracDelta]:
        """The list of all the Dirac deltas, for use inside the class."""
        return self._dirac_delta_list

    @_dirac_deltas.setter
    def _dirac_deltas(self, dirac_deltas: list[DiracDelta]) -> None:
        """Replaces the Dirac deltas, keeping them in a list that counts changes."""
        self._dirac_delta_list = _DiracDeltaList(dirac_deltas)

    def _arrays(self) -> _DiracDeltaArrays:
        """The arrays of the Dirac deltas, computed again only once they change.
        `_ChangeState` notices any change, to the list of Dirac deltas or to a
        Dirac delta."""
        if self._array_cache is None or not self._array_cache.state.holds_for(
            self._dirac_delta_list
        ):
            self._array_cache = _DiracDeltaArrays(self._dirac_delta_list)

        return self._array_cache

    def _cached_arrays(self) -> _DiracDeltaArrays | None:
        """The arrays of the Dirac deltas if they are already computed, for work
        that is quicker to do on the Dirac deltas themselves than to compute them
        for."""
        if self._array_cache is None or not self._array_cache.state.holds_for(
            self._dirac_delta_list
        ):
            return None

        return self._array_cache

    def _set_arrays(self, positions: np.ndarray, masses: np.ndarray) -> None:
        """Keeps the positions and masses of the Dirac deltas, as floats, that a
        method has just worked out, so that they need not be read from every Dirac
        delta. They must be exactly those of the Dirac deltas as they now are."""
        self._array_cache = _DiracDeltaArrays(self._dirac_delta_list, positions, masses)

    def _known(self) -> _Facts:
        """What the value has worked out about its Dirac deltas, which it forgets
        once they change, unless the change is one of its own (see
        `_keep_facts`)."""
        if not self._facts.state.holds_for(self._dirac_delta_list):
            self._facts = _Facts(self._dirac_delta_list)

        return self._facts

    def _keep_facts(self) -> None:
        """Keeps what the value has worked out about its Dirac deltas through a
        change its own methods have just made, e.g., sorting them, which changes
        none of it but what the method sets itself."""
        self._facts.state = _ChangeState(self._dirac_delta_list)

    @property
    def finite_dirac_deltas(self) -> list[DiracDelta]:
        """The list of all the finite Dirac Deltas of the DistributionalValue.

        :return: The list of the finite Dirac Deltas.
        :rtype: list[DiracDelta]
        """
        self.sort()
        if self.has_special_values:
            return self.dirac_deltas[:-3]

        return self.dirac_deltas

    @property
    def positions(self) -> np.ndarray:
        """The list af all the Dirac Delta positions.

        :return: The list of all the Dirac Delta positions, a copy that the caller
            may change.
        :rtype: np.ndarray
        """
        return self._arrays().positions.copy()

    @property
    def masses(self) -> np.ndarray:
        """The list af all the Dirac Delta floating-point masses.

        :return: The list of all the Dirac Delta floating-point masses, a copy that
            the caller may change.
        :rtype: np.ndarray
        """
        return self._arrays().masses.copy()

    @property
    def raw_masses(self) -> np.ndarray:
        """The list af all the Dirac Delta fixed-point masses.

        :return: The list of all the Dirac Delta fixed-point masses, a copy that the
            caller may change.
        :rtype: np.ndarray
        """
        arrays = self._arrays()
        if arrays.raw_masses is None:
            arrays.raw_masses = np.array(
                [dd.raw_mass for dd in self.dirac_deltas], dtype=np.uint64
            )

        return arrays.raw_masses.copy()

    def _total_mass(self) -> float:
        """The total mass of the Dirac deltas, non-finite ones included."""
        arrays = self._arrays()
        if arrays.total_mass is None:
            arrays.total_mass = float(arrays.masses.sum())

        return arrays.total_mass

    def _sorted_positions(self) -> np.ndarray:
        """The positions of all the Dirac deltas, sorted. Not to be changed."""
        arrays = self._arrays()
        if arrays.sorted_positions is None:
            arrays.sorted_positions = np.sort(arrays.positions)

        return arrays.sorted_positions

    def _sorted_support(self) -> tuple[np.ndarray, np.ndarray]:
        """The positions of the strictly-positive-mass Dirac deltas, sorted, and
        their cumulative masses. Not to be changed."""
        arrays = self._arrays()
        if arrays.sorted_support is None:
            positions, masses = self._positive_mass_support()
            order = np.argsort(positions)
            arrays.sorted_support = (positions[order], np.cumsum(masses[order]))

        return arrays.sorted_support

    def _cdf_lookup(self) -> np.ndarray:
        """The CDF at each position of `_sorted_support`, after a leading 0 for
        positions below them all. Not to be changed."""
        arrays = self._arrays()
        if arrays.cdf_lookup is None:
            _, cumulative_masses = self._sorted_support()
            arrays.cdf_lookup = np.concatenate(
                ([0.0], cumulative_masses / self._total_mass())
            )

        return arrays.cdf_lookup

    @property
    def mean(self) -> float | None:
        """The mean position of all the Dirac Deltas. A mean set explicitly, e.g.,
        from a Ux header, is kept as it is. Otherwise, it is calculated from the
        Dirac deltas, and again once they change.

        :return: The mean position of all the Dirac Deltas.
        :rtype: float | None
        """
        if self._mean is not None:
            return self._mean

        known = self._known()
        if known.mean is None:
            return self.calculate_mean()

        return known.mean

    @mean.setter
    def mean(self, mean: float | None):
        """Sets the mean position of all the Dirac Deltas explicitly (it does
        not interfere with the Dirac Deltas list). It is kept as it is, even if
        the Dirac deltas change. Setting `None` goes back to calculating it.

        :param mean: The mean value to use.
        :type mean: float | None
        """
        self._mean = mean

    def calculate_mean(self) -> float | None:
        """Calculated the mean position of all the Dirac Deltas based on the
        Dirac Deltas list. The result replaces a mean set explicitly.

        :return: The calculated mean position of all the Dirac Deltas.
        :rtype: float
        """
        mean: float
        if self.nan_dirac_delta.mass > 0:
            mean = np.nan
        elif self.neg_inf_dirac_delta.mass > 0 and self.pos_inf_dirac_delta.mass > 0:
            mean = np.nan
        elif self.neg_inf_dirac_delta.mass > 0:
            mean = -np.inf
        elif self.pos_inf_dirac_delta.mass > 0:
            mean = np.inf
        elif self.UR_order == 0:
            mean = np.nan
        else:
            arrays = self._arrays()
            total_mass = _sum_in_order(arrays.masses)
            total_weighted_position = _sum_in_order(arrays.positions * arrays.masses)

            if total_mass == 0:
                mean = np.nan
            else:
                mean = total_weighted_position / total_mass

        self._mean = None
        self._known().mean = mean
        return mean

    @property
    def UR_order(self) -> int:
        """The number of non-zero mass Dirac Deltas.

        :return: The number of non-zero mass Dirac Deltas.
        :rtype: int
        """
        return len(self.dirac_deltas)

    @property
    def variance(self) -> float | None:
        """The positional variance of all the Dirac Deltas.

        :return: The positional variance of all the Dirac Deltas.
        :rtype: float | None
        """
        known = self._known()
        if known.variance is None:
            return self.calculate_variance()

        return known.variance

    def calculate_variance(self) -> float | None:
        """Calculates the positional variance of all the Dirac Deltas.

        :return: The calculated positional variance of all the Dirac Deltas.
        :rtype: float | None
        """
        variance: float | None
        mean = self.mean
        # Calculate weighted sample variance
        if mean is None or self.UR_order == 0 or not np.isfinite(mean):
            variance = None
        else:
            total_mass: float = 0
            total_weighted_squared_diffs: float = 0
            for dd in self.dirac_deltas:
                total_weighted_squared_diffs += ((dd.position - mean) ** 2) * dd.mass
                total_mass += dd.mass

            variance = total_weighted_squared_diffs / total_mass

        self._known().variance = variance
        return variance

    def _positive_mass_support(self) -> tuple[np.ndarray, np.ndarray]:
        """Positions and masses of the strictly-positive-mass Dirac deltas.

        Zero-mass deltas carry no probability, so the support-shape
        statistics ignore them: a zero-mass outlier must not widen the
        reported range, and a zero-mass placeholder below the support
        must not shift a quantile or CDF lookup.

        Returns:
            ``(positions, masses)`` restricted to deltas whose mass is
            strictly greater than zero.
        """
        arrays = self._arrays()
        mask = arrays.masses > 0.0
        return arrays.positions[mask], arrays.masses[mask]

    @property
    def range(self) -> float:
        """Max finite position minus min finite position.

        Returns:
            0.0 for a single-Dirac distribution.
            NaN when any NaN-position Dirac has positive mass.
            +Inf when any ±Inf-position Dirac has positive mass.
            NaN for empty / all-zero-mass distributions.
        """
        self.sort()
        if self.nan_dirac_delta.mass > 0:
            return float("nan")
        if self.neg_inf_dirac_delta.mass > 0 or self.pos_inf_dirac_delta.mass > 0:
            return float("inf")
        # After the special-value early returns the only remaining
        # deltas are finite. Drop zero-mass ones so they cannot widen
        # the reported support.
        finite_positions, _ = self._positive_mass_support()
        if finite_positions.size == 0:
            return float("nan")
        return float(finite_positions.max() - finite_positions.min())

    def quantile(self, t: float) -> float:
        """Empirical inverse CDF at ``t`` (non-interpolating step lookup).

        Returns an actual support position (the smallest whose cumulative mass
        reaches ``t``) and takes a single scalar ``t``. For an interpolating,
        array-capable variant that can return values between support points,
        see `inverse_cdf`.

        Args:
            t: Quantile parameter in [0, 1].

        Returns:
            Smallest position whose cumulative normalised mass is at
            least ``t``. NaN if ``t`` is NaN or if the distribution
            has zero total mass.

        Raises:
            ValueError: if ``t`` is outside [0, 1].
        """
        if np.isnan(t):
            return float("nan")
        if t < 0.0 or t > 1.0:
            raise ValueError(f"quantile parameter t must be in [0, 1]; got {t}.")
        if self._total_mass() == 0.0:
            return float("nan")
        sorted_positions, _ = self._sorted_support()
        cumulative = self._cdf_lookup()[1:]
        index = int(np.searchsorted(cumulative, t, side="left"))
        index = min(index, len(sorted_positions) - 1)
        return float(sorted_positions[index])

    @property
    def median(self) -> float:
        """The median position — shorthand for ``quantile(0.5)``."""
        return self.quantile(0.5)

    @overload
    def cdf(self, x: float, *, treat_as_samples: bool = False) -> float: ...

    @overload
    def cdf(self, x: np.ndarray, *, treat_as_samples: bool = False) -> np.ndarray: ...

    def cdf(self, x, *, treat_as_samples: bool = False):
        """Empirical right-continuous step-function CDF at ``x``.

        Args:
            x: Evaluation point(s). Scalar or 1-D array.
            treat_as_samples: When True, ignore masses and treat the
                positions as equally-weighted samples, so the CDF is
                ``(count of positions <= x) / N``. When False (default),
                uses the mass-weighted step CDF.

        Returns:
            CDF value(s) in [0, 1]. ``float`` for scalar input,
            ``np.ndarray`` for array input. NaN for NaN inputs and
            for empty / all-zero-mass distributions.
        """
        if treat_as_samples:
            sample_positions = self._sorted_positions()
            if sample_positions.size == 0:
                if np.ndim(x) == 0:
                    return float("nan")
                return np.full(np.shape(x), np.nan, dtype=float)
            result = (
                np.searchsorted(sample_positions, x, side="right")
                / sample_positions.size
            )
        else:
            if self._total_mass() == 0.0:
                if np.ndim(x) == 0:
                    return float("nan")
                return np.full(np.shape(x), np.nan, dtype=float)
            sorted_positions, _ = self._sorted_support()
            indices = np.searchsorted(sorted_positions, x, side="right")
            result = self._cdf_lookup()[indices]
        if np.ndim(x) == 0:
            if np.isnan(x):
                return float("nan")
            return float(result)
        nan_mask = np.isnan(np.asarray(x))
        if np.any(nan_mask):
            result = np.where(nan_mask, np.nan, result)
        return result

    @overload
    def inverse_cdf(self, p: float, *, treat_as_samples: bool = False) -> float: ...

    @overload
    def inverse_cdf(
        self, p: np.ndarray, *, treat_as_samples: bool = False
    ) -> np.ndarray: ...

    def inverse_cdf(self, p, *, treat_as_samples: bool = False):
        """Interpolated inverse CDF (quantile function), array-capable.

        Unlike `quantile` (a scalar, non-interpolating step lookup that always
        returns an actual support position), this linearly interpolates the
        support positions against their cumulative mass (weighted branch) or
        via ``np.quantile`` over the raw positions (samples branch). The result
        is therefore not snapped to a support point: for ``p`` between two
        support points it can return a value strictly between them (within the
        support range ``[min, max]``). It also accepts array ``p``.

        Args:
            p: Probability level(s) in [0, 1]. Scalar or 1-D array.
            treat_as_samples: When True, ignore masses and treat the
                positions as equally-weighted samples (``np.quantile``).
                When False (default), interpolate against the
                mass-weighted cumulative distribution.

        Returns:
            Interpolated position(s). ``float`` for scalar ``p``,
            ``np.ndarray`` for array ``p``. NaN when the distribution has
            no positions (samples) or no positive mass (weighted).
        """
        if treat_as_samples:
            # `np.quantile` does not change the array, so the cached one will do.
            sample_positions = self._arrays().positions
            if sample_positions.size == 0:
                if np.ndim(p) == 0:
                    return float("nan")
                return np.full(np.shape(p), np.nan, dtype=float)
            values = np.quantile(sample_positions, p)
        else:
            sorted_positions, cumulative_masses = self._sorted_support()
            if sorted_positions.size == 0:
                if np.ndim(p) == 0:
                    return float("nan")
                return np.full(np.shape(p), np.nan, dtype=float)
            arrays = self._arrays()
            if arrays.inverse_cdf_cumulative is None:
                arrays.inverse_cdf_cumulative = (
                    cumulative_masses / cumulative_masses[-1]
                )
            values = np.interp(p, arrays.inverse_cdf_cumulative, sorted_positions)
        if np.ndim(p) == 0:
            return float(values)
        return np.asarray(values)

    def raw_moments(self, order: int) -> float:
        """The ``order``-th raw moment ``Σ p_i · x_i^order``.

        Args:
            order: Moment order (any non-negative integer).

        Returns:
            ``order == 0`` returns 1.0 for any non-empty distribution.
            NaN when any NaN-position Dirac has positive mass.
            +Inf when only +Inf-position mass is present.
            For -Inf-position mass: +Inf for even ``order``,
            -Inf for odd ``order``.
            For both ±Inf-position masses: +Inf for even ``order``,
            NaN for odd ``order``.
            NaN for empty / all-zero-mass distributions.
        """
        # Sorting first builds the arrays of the Dirac deltas once, in sorted order.
        self.sort()
        if self._total_mass() == 0.0:
            return float("nan")
        if order == 0:
            return 1.0
        if self.nan_dirac_delta.mass > 0:
            return float("nan")
        has_neg_inf = self.neg_inf_dirac_delta.mass > 0
        has_pos_inf = self.pos_inf_dirac_delta.mass > 0
        if has_neg_inf and has_pos_inf:
            return float("inf") if order % 2 == 0 else float("nan")
        if has_pos_inf:
            return float("inf")
        if has_neg_inf:
            return float("inf") if order % 2 == 0 else float("-inf")
        positions, masses = self._positive_mass_support()
        total_mass = float(masses.sum())
        if total_mass == 0.0:
            return float("nan")
        return float(np.sum(masses * positions**order) / total_mass)

    def central_moments(self, order: int) -> float:
        """The ``order``-th central moment ``Σ p_i · (x_i − mean)^order``.

        Args:
            order: Moment order (any non-negative integer).

        Returns:
            ``order == 0`` returns 1.0 for any non-empty distribution.
            NaN whenever any NaN- or ±Inf-position Dirac has positive
            mass (the mean is non-finite).
            NaN for empty / all-zero-mass distributions.
        """
        # Sorting first builds the arrays of the Dirac deltas once, in sorted order.
        self.sort()
        if self._total_mass() == 0.0:
            return float("nan")
        if order == 0:
            return 1.0
        if (
            self.nan_dirac_delta.mass > 0
            or self.neg_inf_dirac_delta.mass > 0
            or self.pos_inf_dirac_delta.mass > 0
        ):
            return float("nan")
        positions, masses = self._positive_mass_support()
        total_mass = float(masses.sum())
        if total_mass == 0.0:
            return float("nan")
        mean = float(np.sum(masses * positions) / total_mass)
        return float(np.sum(masses * (positions - mean) ** order) / total_mass)

    @property
    def has_special_values(self) -> bool:
        """Property that identifies if there are non-zero mass Dirac Deltas,
        with non-finite position, i.e. `NaN`, `-Inf`, `Inf`.

        :return: `True` if there are non-zero mass, non-finite position Dirac Deltas,
        `False` otherwise.
        :rtype: bool
        """
        return bool(
            self.nan_dirac_delta.mass > 0
            or self.neg_inf_dirac_delta.mass > 0
            or self.pos_inf_dirac_delta.mass > 0
        )

    @classmethod
    def from_samples(cls, samples: np.ndarray | list[float]) -> "DistributionalValue":
        """
        Construct a DistributionalValue from an array of float samples.

        Each sample becomes a Dirac delta with equal mass (1 / n_total).
        Non-finite values (NaN, -Inf, +Inf) are included and will be
        separated by the sort() method when the DistributionalValue is
        processed.

        To reduce the samples to a TTR, `PlotData.ttr_from_samples` is much
        faster than reducing the value this returns, because it does not build
        a Dirac delta per sample.

        Args:
            samples: 1-D array of float samples (may contain NaN/Inf).

        Returns:
            A DistributionalValue instance.

        Raises:
            ValueError: If the samples array is not 1-D, or is empty.
        """
        samples = np.asarray(samples, dtype=np.float64)
        if np.ndim(samples) != 1:
            raise ValueError(
                f"samples must be a 1-D array, got {np.ndim(samples)} dimensions."
            )
        n_total = len(samples)
        if n_total == 0:
            raise ValueError("samples array must not be empty.")

        mass_per_sample = 1.0 / n_total
        # `tolist` converts every sample to a float at once.
        dirac_deltas = DiracDelta.many(samples.tolist(), mass_per_sample)

        dist = cls(dirac_deltas=dirac_deltas)
        # The positions and masses are known already.
        dist._set_arrays(samples.copy(), np.full(n_total, mass_per_sample))
        return dist

    @classmethod
    def from_data_out(cls, path: str | os.PathLike[str]) -> "DistributionalValue":
        """
        Construct a DistributionalValue from a Monte Carlo `data.out` file.

        A `data.out` file, as written by `saveMonteCarloFloatDataToDataDotOutFile`
        or `saveMonteCarloDoubleDataToDataDotOutFile` in
        https://github.com/signaloid/Signaloid-Demo-CommonUtilityRoutines/blob/main/common.c,
        holds the run's CPU time in microseconds (an unsigned integer) on its
        first line, then one unweighted Monte Carlo sample per line, every line
        ending in a newline. The first line is skipped, blank lines are
        ignored, and every other line must be a single float (`nan`, `inf` and
        `-inf` included). The `...MultidimensionalDataToDataDotOutFile`
        writers put comma-separated samples of several variables on each line,
        which is not supported.

        A first line that is not an integer means the file lacks it (for
        example, the output of `signaloid-uxdata-toolkit sample -o`), and a
        last line with no newline means the file was cut off mid-line; both
        raise. A file cut off exactly at the end of a line cannot be detected.

        Args:
            path: Path to the `data.out` file.

        Returns:
            A DistributionalValue with one equal-mass Dirac delta per sample.

        Raises:
            OSError: If the file cannot be read.
            UnicodeDecodeError: If the file is not UTF-8 text.
            ValueError: If the first line is not an integer, a line after it
                is not a float, the last line has no newline, or the file
                holds no samples after its first line.
        """
        samples: list[float] = []
        with open(path, "r", encoding="utf-8") as file:
            # The first line is the CPU time in microseconds, not a sample.
            cpu_time = file.readline().strip()
            if not cpu_time.isdecimal():
                raise ValueError(
                    f"{path}:1: expected the CPU time in microseconds, got "
                    f"{cpu_time!r}; the file may be missing its first line"
                )
            for line_number, line in enumerate(file, start=2):
                # The writer ends every line, the last included, in a newline.
                if not line.endswith("\n"):
                    raise ValueError(
                        f"{path}:{line_number}: last line has no newline, so "
                        f"the file looks truncated"
                    )
                line = line.strip()
                if not line:
                    continue
                try:
                    samples.append(float(line))
                except ValueError:
                    raise ValueError(
                        f"{path}:{line_number}: expected one float sample per "
                        f"line, got {line!r}"
                    ) from None

        if not samples:
            raise ValueError(f"{path} holds no samples after its first line.")

        return cls.from_samples(samples)

    @classmethod
    def from_weighted_samples(
        cls,
        positions: list[float] | np.ndarray,
        masses: list[float] | np.ndarray,
    ) -> "DistributionalValue":
        """
        Construct a DistributionalValue from weighted samples.

        Validates the inputs (non-empty, equal-length, numeric, non-negative,
        finite and non-zero total mass), normalises the masses to sum to 1.0,
        and builds one Dirac delta per (position, normalised-mass) pair.

        Args:
            positions: 1-D array of sample positions.
            masses: 1-D array of (unnormalised) sample masses, one per position.

        Returns:
            A DistributionalValue instance with normalised masses.

        Raises:
            ValueError: If positions or masses are empty, have differing
                lengths, contain non-numerical values, contain negative
                masses, or the total mass is not finite and non-zero.
        """
        positions_arr = np.array(positions)
        masses_arr = np.array(masses)
        if len(positions_arr) == 0:
            raise ValueError(f"positions {positions_arr} is empty")
        if len(masses_arr) == 0:
            raise ValueError(f"masses {masses_arr} is empty")
        if len(positions_arr) != len(masses_arr):
            raise ValueError(
                f"positions {positions_arr} and masses {masses_arr} "
                f"have differing lengths"
            )
        if not all(
            isinstance(pos, (float, int, np.integer, np.floating))
            for pos in positions_arr
        ):
            raise ValueError(f"{positions_arr} contain non numerical values")
        if not all(
            isinstance(mass, (float, int, np.integer, np.floating))
            for mass in masses_arr
        ):
            raise ValueError(f"{masses_arr} contain non numerical values")
        if np.any(masses_arr < 0.0):
            raise ValueError(f"masses must be non-negative (got {masses_arr})")

        total_mass = float(np.sum(masses_arr))
        if not np.isfinite(total_mass) or total_mass == 0.0:
            raise ValueError(
                f"sum of masses must be finite and non-zero (got {total_mass})"
            )
        normalized = list(masses_arr / total_mass)
        # Positions are kept as floats, as from `from_samples`. Integer positions,
        # e.g., NumPy unsigned integers, would overflow or wrap around in arithmetic.
        dirac_deltas = [
            DiracDelta(position=float(pos), mass=float(m))
            for pos, m in zip(positions_arr, normalized)
        ]
        return cls(dirac_deltas=dirac_deltas)

    def __repr__(self) -> str:
        """Constructs the representation type for the `DistributionalValue`.

        :return: The representation type for the `DistributionalValue`.
        :rtype: str
        """
        return f"{id(self)}:{self.UR_type}-{self.UR_order}"

    def __str__(self) -> str:
        """Constructs the Ux-string representation of the `DistributionalValue`.

        :return: The `DistributionalValue` Ux-string.
        :rtype: str
        """
        result = self.export(to_str=True)
        return result if isinstance(result, str) else ""

    def __bytes__(self) -> bytes:
        """Constructs the byte array for the `DistributionalValue`. Uses either
        single or double precision for support positions based on `self.double_precision`.

        :return: The byte array for the `DistributionalValue`.
        :rtype: bytes
        """

        result = self.export(to_str=False)
        return result if isinstance(result, bytes) else bytes()

    def export(self, to_str: bool = True) -> str | bytes:
        """
        Constructs the Ux String/Ux Binary Data with particle value for the `DistributionalValue`.

        Args:
            to_str: Whether to export a Ux String or Ux Binary Data
                    - True: Export a Ux String
                    - False: Export Ux Binary Data

        Returns:
            The Ux String or the Ux Binary Data with particle value for the `DistributionalValue`.

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
            The Ux Binary Data layout has a 3-byte marker (0xF00000) between
            the particle value and the representation type. The legacy format
            does not have this marker. The `export` function always produces the
            Ux Binary layout and not the legacy. The `parse` function accepts
            both (see `parse`).
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
        """

        # Create byte representation
        buffer: bytes = bytes()
        UxString: str = ""

        if to_str:
            # Particle value (double in string format)
            UxString += (
                str(self.particle_value) if self.particle_value is not None else ""
            )
            UxString += "Ux"

            fmt = STRUCT_FORMATS["str"]

            # # Representation type (uint8_t)                     (1 byte)
            buffer += struct.pack(fmt["UR_type"], self.UR_type)
        else:
            # Particle value (double)                           (8 bytes)
            particle_value = (
                self.particle_value if self.particle_value is not None else 0
            )
            fmt = STRUCT_FORMATS["bytes"]
            buffer += struct.pack(fmt["particle"], particle_value)

            # Ux Binary Data format specifies:
            # - start byte (0xF0),
            # - followed by 2 padding bytes (0x0000).
            # These are inserted between the particle value and the
            # representation type.
            buffer += bytes([UX_BINARY_FORMAT_MARKER, 0x00, 0x00])

            # Representation type (uint8_t)                     (1 byte)
            buffer += struct.pack(
                fmt["UR_type"],
                UR_TYPE_ATHENS if self.UR_type == UR_TYPE_UNSPECIFIED else self.UR_type,
            )

        # Number of samples (uint64_t)                      (8 bytes)
        buffer += struct.pack(fmt["sample_count"], self.UR_order)

        # Mean value of distribution (double)               (8 bytes)
        # Mean is always double precision regardless of self.double_precision
        buffer += struct.pack(fmt["mean"], self.mean)

        # Number of non-zero mass Dirac deltas (uint32_t)   (4 bytes)
        buffer += struct.pack(fmt["UR_order"], self.UR_order)

        # Choose the format based on double_precision flag
        position_format = fmt[
            "position_double" if self.double_precision else "position_single"
        ]

        # Pairs of:
        # - Support position (double or float)              (8 or 4 bytes)
        # - Probability mass (uint64_t)                     (8 bytes)
        # The pairs are joined once at the end. Adding each to `buffer` would copy
        # all of it every time, which takes time quadratic in the number of pairs.
        parts = [buffer]
        for dd in self.dirac_deltas:
            # Pack the position using either double or float precision
            parts.append(struct.pack(position_format, dd.position))

            # Probability mass is always uint64_t
            parts.append(struct.pack(fmt["mass"], dd.raw_mass))
        buffer = b"".join(parts)

        if to_str:
            return UxString + buffer.hex().upper()

        return buffer

    @staticmethod
    def parse(
        dist: str | bytes, double_precision: bool | None = None
    ) -> "DistributionalValue | None":
        """
        Constructs a `DistributionalValue` after parsing an input that can be
        a Ux String or a byte array.

        Args:
            dist: The input Ux String or byte array.
            double_precision: The floating point representation precision.
                - true: double precision
                - false: single precision
        Returns:
            The constructed `DistributionalValue` or None if parsing fails.

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
            The Ux Binary Data layout has a 3-byte marker (0xF00000) between
            the particle value and the representation type. The legacy format
            does not have this marker. The `export` function always produces the
            Ux Binary layout and not the legacy. The `parse` function accepts
            both (see `parse`).
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
        """

        if not dist:
            return None

        # If input is a hex string without 'Ux', convert it to bytes and try to
        # parse it as Ux Binary Data
        if isinstance(dist, str) and "Ux" not in dist:
            dist = bytes.fromhex(dist)

        buffer: bytes = bytes()
        offset = 0

        double_precision_is_known = double_precision is not None
        if double_precision is None:
            double_precision = True

        dist_value = DistributionalValue(double_precision=double_precision)

        if isinstance(dist, str):
            # Parse Ux String from string of hexadecimal characters
            fmt = STRUCT_FORMATS["str"]

            # Match an optional floating-point or integer number, followed by 'Ux',
            # and then hexadecimal characters
            if sys.implementation.name == "circuitpython":
                parts = dist.split("Ux")
                index = 0
                if len(parts) == 2:
                    dist_value.particle_value = float(parts[index])
                    index += 1

                buffer = bytes.fromhex(parts[index])
            else:
                pattern = (
                    r"^([-+]?\d*\.?\d+(?:e[-+]\d+)?|nan|[-+]?inf)?Ux([0-9A-Fa-f]+)$"
                )
                match = re.match(pattern, dist)
                if not match:
                    return None

                dist_value.particle_value = (
                    float(match.group(1)) if match.group(1) else None
                )

                buffer = bytes.fromhex(match.group(2))
        elif isinstance(dist, (bytes, bytearray)):
            # Parse byte array
            fmt = STRUCT_FORMATS["bytes"]

            buffer = dist

            # Need 8 bytes for the particle value plus at least 1 more byte
            # to detect the layout (see below).
            if len(buffer) < 9:
                return None

            dist_value.particle_value = struct.unpack(
                fmt["particle"], buffer[offset : offset + 8]
            )[0]
            offset += 8

            if buffer[offset] == UX_BINARY_FORMAT_MARKER:
                # If offset has the Ux Binary data format marker then we are not
                # in the legacy format. Skip the 3-byte marker (0xF00000).
                offset += 3
        else:
            print("Error: ", type(dist))
            return None

        # Check if the buffer has the minimum required length (in bytes).
        # Minimum header after the current offset = 1 (repr) + 8 (samples) +
        # 8 (mean) + 4 (count) = 21. The offset already accounts for the
        # particle value and, for the Ux Binary layout, the 3-byte start
        # marker.
        min_length = offset + 21
        if len(buffer) < min_length:
            return None

        dist_value.UR_type = struct.unpack(fmt["UR_type"], buffer[offset : offset + 1])[
            0
        ]
        offset += 1

        # Not used, uncomment if needed
        # number_of_samples = struct.unpack(fmt["sample_count"], buffer[offset:offset + 8])[0]
        offset += 8

        dist_value.mean = struct.unpack(fmt["mean"], buffer[offset : offset + 8])[0]
        offset += 8

        UR_order = struct.unpack(fmt["UR_order"], buffer[offset : offset + 4])[0]
        offset += 4

        # Validate UR_order - reasonable upper limit to prevent processing
        # extremely large inputs that might be malicious
        if UR_order is None or UR_order < 0 or UR_order > 10000:
            return None

        # Calculate expected length based on UR_order
        # 4 or 8 for position + 8 for mass
        bytes_per_position = 8 if double_precision else 4
        bytes_per_dirac_delta = bytes_per_position + 8
        expected_length = min_length + (UR_order * bytes_per_dirac_delta)
        if len(buffer) < expected_length:
            if double_precision_is_known:
                return None

            # Check if it can be parsed with the other precision
            double_precision = False
            dist_value.double_precision = double_precision
            bytes_per_position = 4
            bytes_per_dirac_delta = bytes_per_position + 8
            expected_length = min_length + (UR_order * bytes_per_dirac_delta)
            if len(buffer) < expected_length:
                return None

        position_format = fmt[
            "position_double" if double_precision else "position_single"
        ]

        for _ in range(UR_order):
            support_position_bytes = buffer[offset : offset + bytes_per_position]
            offset += bytes_per_position
            position = struct.unpack(position_format, support_position_bytes)[0]

            mass_bytes = buffer[offset : offset + 8]
            offset += 8
            raw_mass = struct.unpack(fmt["mass"], mass_bytes)[0]

            dist_value.dirac_deltas.append(DiracDelta(position, raw_mass=raw_mass))

        return dist_value

    def mean_distance(self, other: "DistributionalValue") -> float:
        """
        Calculates the distance between the mean values of this `DistributionalValue`
        and another `DistributionalValue`.

        Args:
            other: The other `DistributionalValue`.
        Returns:
            The distance between the mean values.
        """

        if self.mean is None or other.mean is None:
            return math.nan

        return abs(self.mean - other.mean)

    def mean_relative_diff(self, other: "DistributionalValue") -> float:
        """
        Calculates the relative difference between the mean values of
        this `DistributionalValue` and another `DistributionalValue`.

        Args:
            other: The other `DistributionalValue`.
        Returns:
            The relative difference between the mean values (normalized by
            the mean value of the other `DistributionalValue`).
        """
        if self.mean is None or other.mean is None or other.mean == 0:
            return math.nan

        return abs((self.mean - other.mean) / other.mean)

    @property
    def has_no_zero_mass(self) -> bool | None:
        """
        The property that no Dirac delta of the `DistributionalValue` has a zero mass.

        Returns:
            `True` if `DistributionalValue` has a zero mass Dirac delta, `False` else.
        """
        known = self._known()
        if known.has_no_zero_mass is None:
            known.has_no_zero_mass = self.check_has_no_zero_mass()

        return known.has_no_zero_mass

    def check_has_no_zero_mass(self) -> bool | None:
        """
        Checks the property that no Dirac delta of the `DistributionalValue` has a zero
        mass.

        Returns:
            `True` if `DistributionalValue` has a zero mass Dirac delta, `False` else.
        """
        if self.UR_order == 0:
            return None

        for dd in self.dirac_deltas:
            if dd.mass == 0:
                return False

        return True

    def drop_zero_mass_positions(self) -> None:
        """
        Drops (removes) the Dirac deltas of the `DistributionalValue` that has a zero
        mass.
        """
        if self._known().has_no_zero_mass:
            return

        if self.UR_order == 0:
            return

        arrays = self._cached_arrays()
        if arrays is not None and bool(np.all(arrays.masses > 0)):
            # There is nothing to drop.
            self._known().has_no_zero_mass = True
            return

        # The value's own NaN, -Inf, and +Inf Dirac deltas stay, even with no mass,
        # as `finite_dirac_deltas` relies on finding all three at the end once they
        # are there.
        nan_dirac_delta = self.nan_dirac_delta
        neg_inf_dirac_delta = self.neg_inf_dirac_delta
        pos_inf_dirac_delta = self.pos_inf_dirac_delta
        self._dirac_deltas = [
            dd
            for dd in self._dirac_deltas
            if dd.mass > 0
            or dd is nan_dirac_delta
            or dd is neg_inf_dirac_delta
            or dd is pos_inf_dirac_delta
        ]
        self._keep_facts()
        self._known().has_no_zero_mass = True

    @property
    def is_finite(self) -> bool | None:
        """
        The property that all Dirac deltas of the `DistributionalValue` have finite
        positions, i.e., no NaN, -Inf, or Inf values.

        Returns:
            `True` if `DistributionalValue` has a Dirac delta with non-finite position,
            `False` else.
        """
        known = self._known()
        if known.is_finite is None:
            known.is_finite = self.check_is_finite()

        return known.is_finite

    def check_is_finite(self) -> bool | None:
        """
        Checks the property that all Dirac deltas of the `DistributionalValue` have
        finite positions, i.e., no NaN, -Inf, or Inf values.

        Returns:
            `None` if `DistributionalValue` has no Dirac deltas. Else, `False` if
            `DistributionalValue` has a Dirac delta with non-finite position, `True` else.
        """
        if self.UR_order == 0:
            return None

        arrays = self._cached_arrays()
        if arrays is not None:
            return bool(np.all(np.isfinite(arrays.positions)))

        for dd in self.dirac_deltas:
            if not dd.isFinite:
                return False

        return True

    @property
    def is_sorted(self) -> bool | None:
        """
        The property that the Dirac deltas of the `DistributionalValue` are sorted
        according to their positions. The NaN, -Inf, and Inf positional values
        are cured and sorted to the end in the order [NaN, -Inf, Inf].

        Returns:
            `True` if the Dirac deltas of the `DistributionalValue` are sorted according
            to their positions, `False` else.
        """
        known = self._known()
        if known.is_sorted is None:
            known.is_sorted = self.check_is_sorted()

        return known.is_sorted

    def check_is_sorted(self) -> bool | None:
        """
        Checks The property that the Dirac deltas of the `DistributionalValue` are
        sorted according to their positions. The NaN, -Inf, and Inf positional values
        are cured and sorted to the end in the order [NaN, -Inf, Inf].

        Returns:
            `None` if the `DistributionalValue` has no Dirac deltas. Else, `True` if
            the Dirac deltas of the `DistributionalValue` are sorted according to
            their positions, `False` else.
        """
        if self.UR_order == 0:
            return None

        if self.UR_order == 1:
            return True

        for i in range(0, self.UR_order - 1):
            if not self.dirac_deltas[i].isFinite:
                return False

            if self.dirac_deltas[i] > self.dirac_deltas[i + 1]:
                return False

        return True

    def sort(self) -> None:
        """
        Sorts the positions and the masses of a `DistributionalValue` according to
        the positions. Also, cures multiple entries for NaN, -Inf, and Inf and
        places them at the end of positions and masses with the order [NaN, -Inf, Inf].
        """
        if self._known().is_sorted:
            return

        # The value's own NaN, -Inf, and +Inf Dirac deltas, there from sorting
        # before, count like any other.
        nan_mass = neg_inf_mass = pos_inf_mass = 0.0
        finite_dirac_deltas: list[DiracDelta]
        sorted_arrays: tuple[np.ndarray, np.ndarray] | None = None
        arrays = self._cached_arrays()
        # The ulab NumPy of CircuitPython has neither a stable sort nor
        # integer-array indexing.
        if arrays is not None and sys.implementation.name != "circuitpython":
            # With the arrays of the Dirac deltas at hand, sorting them is much
            # faster than sorting the Dirac deltas, and gives their arrays sorted.
            # A stable sort orders equal positions as `list.sort` does, and the
            # masses are summed in order, as the loop below does.
            positions, masses = arrays.positions, arrays.masses
            nan_mass = _sum_in_order(masses[np.isnan(positions)])
            neg_inf_mass = _sum_in_order(masses[positions == -np.inf])
            pos_inf_mass = _sum_in_order(masses[positions == np.inf])
            finite = np.nonzero(np.isfinite(positions))[0]
            order = finite[np.argsort(positions[finite], kind="stable")]
            dirac_deltas = self._dirac_deltas
            finite_dirac_deltas = [dirac_deltas[index] for index in order.tolist()]
            sorted_arrays = (positions[order], masses[order])
        else:
            # This visits every Dirac delta, so it tests each position with
            # `math`, which is much faster than NumPy on a single float.
            finite_dirac_deltas = []
            for dd in self.dirac_deltas:
                if math.isfinite(dd.position):
                    finite_dirac_deltas.append(dd)
                elif math.isnan(dd.position):
                    nan_mass += dd.mass
                elif dd.position < 0:
                    neg_inf_mass += dd.mass
                else:
                    pos_inf_mass += dd.mass
            # Sorting by the position compares floats, where the default ordering
            # would call `DiracDelta.__lt__` for every comparison. The order is the
            # same.
            finite_dirac_deltas.sort(key=lambda dd: dd.position)

        # New Dirac deltas rather than changed ones, as the old ones may also be in
        # another list, which a change to them would change too.
        self.nan_dirac_delta = DiracDelta(position=np.nan, mass=nan_mass)
        self.neg_inf_dirac_delta = DiracDelta(position=-np.inf, mass=neg_inf_mass)
        self.pos_inf_dirac_delta = DiracDelta(position=np.inf, mass=pos_inf_mass)

        self._dirac_deltas = finite_dirac_deltas

        if self.has_special_values:
            self.dirac_deltas.append(self.nan_dirac_delta)
            self.dirac_deltas.append(self.neg_inf_dirac_delta)
            self.dirac_deltas.append(self.pos_inf_dirac_delta)
            if sorted_arrays is not None:
                sorted_arrays = (
                    np.concatenate(
                        (sorted_arrays[0], np.array([np.nan, -np.inf, np.inf]))
                    ),
                    np.concatenate(
                        (
                            sorted_arrays[1],
                            np.array([nan_mass, neg_inf_mass, pos_inf_mass]),
                        )
                    ),
                )

        self._keep_facts()
        if sorted_arrays is not None:
            self._set_arrays(*sorted_arrays)
        self._known().is_sorted = True

    @property
    def is_cured(self) -> bool | None:
        """
        The property that no two Dirac deltas of the `DistributionalValue` have
        the same positional value, including NaN, -Inf, and Inf.

        Returns:
            `True` if no two Dirac deltas of the `DistributionalValue` have the same
            positional value, including NaN, -Inf, and Inf, `False` else.
        """
        known = self._known()
        if known.is_cured is None:
            known.is_cured = self.check_is_cured()

        return known.is_cured

    def check_is_cured(self) -> bool | None:
        """
        Checks the property that no two Dirac deltas of the `DistributionalValue` have
        the same positional value, including NaN, -Inf, and Inf.

        Returns:
            `None` if the `DistributionalValue` has no Dirac deltas. Else, `True` if
            no two Dirac deltas of the `DistributionalValue` have the same positional
            value, including NaN, -Inf, and Inf, `False` else.
        """

        if self.UR_order == 0:
            return None

        return self.UR_order == len(set(self.positions))

    def cure(self) -> None:
        """
        Cures the positions and masses of the `DistributionalVariable` from multiple
        entries of the same positional value, including NaN, -Inf, and Inf.
        """
        if self._known().is_cured:
            return

        self.combine_dirac_deltas(0, 0)

    def combine_dirac_deltas(
        self,
        relative_mean_threshold: float = COMBINE_RELATIVE_MEAN_THRESHOLD,
        relative_range_threshold: float = COMBINE_RELATIVE_RANGE_THRESHOLD,
    ) -> None:
        """
        Combine Dirac deltas with same, very-close-relative-to-range,
        very-close-relative-to-mean-value positions, as `combine_sorted_arrays`
        describes. Dirac deltas that do not combine with another are kept as they
        are.

        :param relative_mean_threshold: The threshold multiplier for the relative
        mean.
        :type relative_mean_threshold: float
        :param relative_range_threshold: The threshold multiplier for the relative
        range.
        :type relative_range_threshold: float
        """
        self.sort()

        finite_dirac_deltas = self.finite_dirac_deltas
        count = len(finite_dirac_deltas)
        if count <= 1:
            self._known().is_cured = True
            return

        # Sorting puts the finite Dirac deltas first.
        arrays = self._arrays()
        positions, masses, runs = DistributionalValue.combine_sorted_arrays(
            arrays.positions[:count],
            arrays.masses[:count],
            relative_mean_threshold,
            relative_range_threshold,
        )

        dirac_deltas: list[DiracDelta] = []
        kept_from = 0
        for start, end in runs:
            dirac_deltas.extend(finite_dirac_deltas[kept_from:start])
            # Each Dirac delta kept, and each run, gives one combined Dirac delta.
            combined = len(dirac_deltas)
            dirac_deltas.append(
                DiracDelta(float(positions[combined]), mass=float(masses[combined]))
            )
            kept_from = end
        dirac_deltas.extend(finite_dirac_deltas[kept_from:])

        self._dirac_deltas = dirac_deltas

        if self.has_special_values:
            self.dirac_deltas.append(self.nan_dirac_delta)
            self.dirac_deltas.append(self.neg_inf_dirac_delta)
            self.dirac_deltas.append(self.pos_inf_dirac_delta)
            positions = np.concatenate((positions, np.array([np.nan, -np.inf, np.inf])))
            masses = np.concatenate(
                (
                    masses,
                    np.array(
                        [
                            self.nan_dirac_delta.mass,
                            self.neg_inf_dirac_delta.mass,
                            self.pos_inf_dirac_delta.mass,
                        ]
                    ),
                )
            )

        self._keep_facts()
        # The combined positions and masses are those of the new Dirac deltas.
        self._set_arrays(positions, masses)
        self._known().is_cured = True

    @staticmethod
    def combine_sorted_arrays(
        positions: np.ndarray,
        masses: np.ndarray,
        relative_mean_threshold: float = COMBINE_RELATIVE_MEAN_THRESHOLD,
        relative_range_threshold: float = COMBINE_RELATIVE_RANGE_THRESHOLD,
    ) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]]]:
        """
        Combines Dirac deltas given as arrays of their positions and masses, which
        `combine_dirac_deltas` uses, and which `PlotData.ttr_from_samples` uses to
        combine samples without building a Dirac delta for each.

        Taken in order, each Dirac delta combines into the last one kept when the
        two are within the threshold of each other, as `DiracDelta.similar` decides,
        into the Dirac delta that `DiracDelta` addition gives. The last one kept is
        the Dirac delta before it, unless that one combined into an earlier one, in
        which case it is their combination. Dirac deltas at equal positions combine
        into one at that position, so each run of them combines at once. Between
        the different positions left, the comparisons are made for all of them at
        once, and only neighbours within the threshold of each other are combined
        one at a time.

        Args:
            positions: The finite positions of the Dirac deltas, sorted. At least
                two are required.
            masses: The masses of the Dirac deltas.
            relative_mean_threshold: The threshold multiplier for the relative
                mean.
            relative_range_threshold: The threshold multiplier for the relative
                range.
        Returns:
            (positions, masses, runs): New arrays of the positions and masses of
                the Dirac deltas after combining, and, in order, a (start, end) pair
                for each one that combines several: those from `start` up to but
                excluding `end`.
        """

        positions = np.asarray(positions, dtype=np.float64)
        masses = np.asarray(masses, dtype=np.float64)

        threshold: float = 0.0
        if relative_mean_threshold > 0 and relative_range_threshold > 0:
            total_mass = float(np.sum(masses))
            # With no mass at all, the positions count equally.
            finite_mean = (
                float(np.sum(positions * masses)) / total_mass
                if total_mass > 0
                else float(np.mean(positions))
            )
            # The threshold is a distance, so it uses the size of the mean. A
            # negative mean would otherwise give a negative threshold, and
            # coincident Dirac deltas would never be combined.
            mean_threshold = abs(finite_mean) * relative_mean_threshold

            range_threshold = (
                float(positions[-1]) - float(positions[0])
            ) * relative_range_threshold

            threshold = max(mean_threshold, range_threshold)

        # Group the runs of equal positions, each into one at that position. Boolean
        # masks and slices select them, as the ulab NumPy of CircuitPython does not
        # support integer-array indexing.
        count = len(positions)
        is_group_start = np.concatenate(
            (np.array([True]), positions[1:] != positions[:-1])
        )
        group_starts = np.nonzero(is_group_start)[0]
        group_ends = np.concatenate((group_starts[1:], np.array([count])))
        group_positions = positions[is_group_start]
        group_masses = masses[is_group_start]
        equal_position_groups = [
            int(group) for group in np.nonzero(group_ends - group_starts > 1)[0]
        ]
        for group in equal_position_groups:
            group_masses[group] = np.sum(
                masses[group_starts[group] : group_ends[group]]
            )

        # Combine neighbouring groups within the threshold of each other into
        # chains, each group with the combination of those before it in the chain.
        chains: list[tuple[int, int, float, float]] = []
        # The first group that has not yet been compared with the last one kept.
        next_group = 1
        within_threshold = (
            np.abs(group_positions[:-1] - group_positions[1:]) <= threshold
        )
        for index in np.nonzero(within_threshold)[0]:
            first = int(index)
            if first + 1 < next_group:
                # Already compared with the combination of the chain before it.
                continue

            position = float(group_positions[first])
            mass = float(group_masses[first])
            end = first + 1
            while end < len(group_positions):
                next_position = float(group_positions[end])
                if not abs(position - next_position) <= threshold:
                    break
                position, mass = combined_position_and_mass(
                    position, mass, next_position, float(group_masses[end])
                )
                end += 1

            chains.append((first, end, position, mass))
            next_group = end + 1

        keep = np.full(len(group_positions), True)
        for first, end, position, mass in chains:
            group_positions[first] = position
            group_masses[first] = mass
            keep[first + 1 : end] = False

        # The runs of Dirac deltas that combine: the chains, and the groups of equal
        # positions outside them, in order.
        group_runs: list[tuple[int, int]] = []
        chain = 0
        for group in equal_position_groups:
            while chain < len(chains) and chains[chain][1] <= group:
                group_runs.append((chains[chain][0], chains[chain][1]))
                chain += 1
            if chain < len(chains) and chains[chain][0] <= group:
                continue
            group_runs.append((group, group + 1))
        group_runs += [(first, end) for first, end, _, _ in chains[chain:]]
        runs = [
            (int(group_starts[first]), int(group_ends[end - 1]))
            for first, end in group_runs
        ]

        return group_positions[keep], group_masses[keep], runs

    @property
    def is_full_valid_TTR(self) -> bool | None:
        """
        The property that the Dirac deltas of the `DistributionalValue` form
        a full valid TTR. "Full" means that there are 2^n Dirac deltas (after
        dropping zero mass Dirac deltas and curing to combine same position
        Dirac deltas). "Valid" means that there is a distribution whose TTR
        exactly contains the Dirac deltas of the `DistributionalValue`.

        Returns:
            `True` if the Dirac deltas of the `DistributionalValue` form a full
            and valid TTR, `False` else.
        """
        known = self._known()
        if known.is_full_valid_TTR is None:
            known.is_full_valid_TTR = self.check_is_full_valid_TTR()

        return self._known().is_full_valid_TTR

    def check_is_full_valid_TTR(self) -> bool | None:
        """
        Checks the property that the Dirac deltas of the `DistributionalValue`
        form a full and valid TTR. "Full" means that there are 2^n Dirac deltas
        (after dropping zero mass Dirac deltas and curing to combine same position
        Dirac deltas). "Valid" means that there is a distribution whose TTR
        exactly contains the Dirac deltas of the `DistributionalValue`.

        Returns:
            `None` if the `DistributionalValue` has no Dirac deltas. Else, `True` if
            the Dirac deltas of the `DistributionalValue` form a full and valid TTR,
            `False` else.
        """

        self.drop_zero_mass_positions()
        self.cure()

        if not self.is_finite:
            return False

        if self.UR_order == 0:
            return None

        if self.UR_order == 1:
            return True

        # Check UR_order is power of 2
        if self.UR_order & (self.UR_order - 1) != 0:
            return False

        ttr_order = int(np.log2(self.UR_order))
        number_of_boundaries = 2 * self.UR_order - 1
        boundary_positions = np.array([np.nan] * number_of_boundaries)
        boundary_probabilities = np.array([np.nan] * number_of_boundaries)
        boundary_positions[::2] = self.positions
        boundary_probabilities[::2] = self.masses

        for n in range(ttr_order):
            step = 2**n
            for i in range(2 ** (n + 1) - 1, number_of_boundaries, 2 ** (n + 2)):
                boundary_probabilities[i] = (
                    boundary_probabilities[i - step] + boundary_probabilities[i + step]
                )
                boundary_positions[i] = (
                    boundary_probabilities[i - step] * boundary_positions[i - step]
                    + boundary_probabilities[i + step] * boundary_positions[i + step]
                ) / boundary_probabilities[i]

        return bool(np.all(boundary_positions[:-1] < boundary_positions[1:]))
