# CircuitPython check

The `circuitpython` job of the workflow uses these files to check that the
CircuitPython code of this package still works: `signaloid.circuitpython`, and
the `distributional` and `distributional_information_plotting` modules it uses.
The [Cardputer demo](https://github.com/signaloid/Signaloid-Compute-Module-Demo-CardPuter)
runs them on CircuitPython 10.3.1. There, NumPy is ulab, which has only part of
NumPy, and modules such as `functools`, `operator` and `typing` do not exist.

- `setup-emulator.sh <work dir>`: builds a stand-in for CircuitPython 10.3.1
  (the MicroPython unix port with CircuitPython's ulab and single-precision
  floats), and downloads CircuitPython 10.3.1's `mpy-cross`.
- `check_plotting.py <src dir>`: runs the plotting path of the Cardputer demo in
  the emulator, from Ux bytes to `plot()`, on one real result and 200 random
  distributions. The display is stubbed out.

To run the check locally, on Linux or an Apple silicon Mac, from the top of
the repository:

```
bash .github/circuitpython/setup-emulator.sh ~/circuitpython-emulator
~/circuitpython-emulator/bin/micropython -X heapsize=64M .github/circuitpython/check_plotting.py src
```

The emulator is not CircuitPython. It catches imports that CircuitPython lacks,
syntax its compiler rejects, and NumPy that ulab does not support. It does not
check memory use or the display, so changes to the CircuitPython code still need
a run on the Cardputer.
