#!/bin/bash
#
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
#
# Builds a stand-in for CircuitPython 10.3.1, for running the CircuitPython code
# of this package off-hardware, and downloads CircuitPython 10.3.1's mpy-cross:
#   - the MicroPython unix port, of the version CircuitPython 10.3.1 is close to;
#   - ulab at the commit CircuitPython 10.3.1 ships, with CircuitPython's ulab
#     options (no complex, no random, reverse special methods);
#   - single-precision floats, as on the ESP32-S3 of the Cardputer;
#   - the CircuitPython behaviour this package relies on, patched in:
#     sys.implementation.name == "circuitpython", int.bit_length() and the
#     slice() constructor.
# The display (displayio, uplot) is not emulated.
#
# Usage: setup-emulator.sh <work dir>
# Leaves <work dir>/bin/micropython and <work dir>/bin/mpy-cross. Does nothing
# if both are already there, e.g., restored from a cache.
#
# Upgrade all three versions together, to those of the CircuitPython release
# that the Cardputer demo uses.
set -e

CIRCUITPYTHON_VERSION=10.3.1
MICROPYTHON_VERSION=v1.27.0
ULAB_COMMIT=1d3ddd8f5228e2c7335dac71f4846e6abb0ec9f9

WORK=${1:?usage: $0 <work dir>}
mkdir -p "$WORK/bin"
cd "$WORK"
if [ -x bin/micropython ] && [ -x bin/mpy-cross ]; then
	echo "CircuitPython emulator already in $WORK/bin"
	exit 0
fi

MPY_CROSS_URL_PREFIX=https://adafruit-circuit-python.s3.amazonaws.com/bin/mpy-cross
case "$(uname -s)-$(uname -m)" in
	Linux-x86_64) MPY_CROSS_URL=$MPY_CROSS_URL_PREFIX/linux-amd64/mpy-cross-linux-amd64-$CIRCUITPYTHON_VERSION.static ;;
	Linux-aarch64) MPY_CROSS_URL=$MPY_CROSS_URL_PREFIX/linux-aarch64/mpy-cross-linux-aarch64-$CIRCUITPYTHON_VERSION.static-aarch64 ;;
	Darwin-arm64) MPY_CROSS_URL=$MPY_CROSS_URL_PREFIX/macos/mpy-cross-macos-$CIRCUITPYTHON_VERSION-arm64 ;;
	*) echo "No CircuitPython mpy-cross for $(uname -s)-$(uname -m)"; exit 1 ;;
esac
curl -fsSL -o bin/mpy-cross "$MPY_CROSS_URL"
chmod +x bin/mpy-cross

[ -d micropython ] || git clone -q --depth 1 --branch "$MICROPYTHON_VERSION" https://github.com/micropython/micropython.git micropython
[ -d ulab ] || { git clone -q https://github.com/v923z/micropython-ulab.git ulab && git -C ulab checkout -q "$ULAB_COMMIT"; }

python3 - <<'PY'
def patch(path, old, new):
    text = open(path).read()
    if new in text:
        return
    assert text.count(old) == 1, (path, old[:50])
    open(path, "w").write(text.replace(old, new))

patch("micropython/py/modsys.c", "MP_ROM_QSTR(MP_QSTR_micropython), \\", "MP_ROM_QSTR(MP_QSTR_circuitpython), \\")
patch("micropython/py/objint.c", "static const mp_rom_map_elem_t int_locals_dict_table[] = {\n", """static mp_obj_t int_bit_length(mp_obj_t self_in) {
    if (!mp_obj_is_small_int(self_in)) {
        mp_raise_NotImplementedError(MP_ERROR_TEXT("emulator: bit_length of a long int"));
    }
    mp_int_t value = MP_OBJ_SMALL_INT_VALUE(self_in);
    mp_uint_t magnitude = value < 0 ? -value : value, bits = 0;
    while (magnitude) { bits++; magnitude >>= 1; }
    return mp_obj_new_int_from_uint(bits);
}
static MP_DEFINE_CONST_FUN_OBJ_1(int_bit_length_obj, int_bit_length);

static const mp_rom_map_elem_t int_locals_dict_table[] = {
    { MP_ROM_QSTR(MP_QSTR_bit_length), MP_ROM_PTR(&int_bit_length_obj) },
""")
patch("micropython/py/objslice.c", "#if MICROPY_PY_BUILTINS_SLICE_INDICES && !MICROPY_PY_BUILTINS_SLICE_ATTRS\nstatic const", """static mp_obj_t slice_make_new(const mp_obj_type_t *type, size_t n_args, size_t n_kw, const mp_obj_t *args) {
    mp_arg_check_num(n_args, n_kw, 1, 3, false);
    mp_obj_t start = mp_const_none, stop = mp_const_none, step = mp_const_none;
    if (n_args == 1) { stop = args[0]; } else { start = args[0]; stop = args[1]; if (n_args == 3) { step = args[2]; } }
    return mp_obj_new_slice(start, stop, step);
}

#if MICROPY_PY_BUILTINS_SLICE_INDICES && !MICROPY_PY_BUILTINS_SLICE_ATTRS
static const""")
patch("micropython/py/objslice.c", "    unary_op, slice_unary_op,\n    SLICE_TYPE_ATTR_OR_LOCALS_DICT", "    make_new, slice_make_new,\n    unary_op, slice_unary_op,\n    SLICE_TYPE_ATTR_OR_LOCALS_DICT")
PY

JOBS=$(getconf _NPROCESSORS_ONLN)
make -C micropython/ports/unix submodules VARIANT=standard > /dev/null
make -C micropython/mpy-cross -j"$JOBS" > /dev/null
make -C micropython/ports/unix -j"$JOBS" VARIANT=standard USER_C_MODULES="$PWD/ulab" \
	MICROPY_PY_FFI=0 MICROPY_PY_SSL=0 MICROPY_PY_BTREE=0 \
	CFLAGS_EXTRA="-DCIRCUITPY_ULAB=1 -DULAB_SUPPORTS_COMPLEX=0 -DULAB_NUMPY_HAS_RANDOM_MODULE=0 -DMICROPY_PY_REVERSE_SPECIAL_METHODS=1 -DMICROPY_FLOAT_IMPL=MICROPY_FLOAT_IMPL_FLOAT -DMICROPY_PY_BUILTINS_SLICE_ATTRS=1 -DMICROPY_PY_BUILTINS_SLICE_INDICES=1" \
	> build.log 2>&1 || { tail -30 build.log; exit 1; }
cp micropython/ports/unix/build-standard/micropython bin/micropython

bin/micropython -c "import sys, ulab; print(sys.implementation.name, 'ulab', ulab.__version__)"
