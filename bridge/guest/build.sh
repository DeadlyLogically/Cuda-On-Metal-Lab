#!/bin/sh
set -eu
out=${1:?pass output directory}
src=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
mkdir -p "$out"
exports="$src/export_refusal.c"
if test -n "${BRIDGE_PRIVATE_EXPORTS:-}"; then exports="$BRIDGE_PRIVATE_EXPORTS"; fi
cc -std=c11 -O2 -g -Wall -Wextra -Werror -fPIC -shared -Wl,-soname,libcuda.so.1 -o "$out/libcuda.so.1" "$src/wire.c" "$src/driver.c" "$exports" -lpthread -ldl
ln -sf libcuda.so.1 "$out/libcuda.so"
cc -std=c11 -O2 -g -Wall -Wextra -Werror -o "$out/probe" "$src/probe.c" -L"$out" -lcuda -ldl -Wl,-rpath,'$ORIGIN'
