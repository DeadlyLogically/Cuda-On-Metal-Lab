#!/bin/sh
set -eu
cd "$(dirname "$0")"
xcrun clang++ -O2 -std=c++17 -fobjc-arc -framework Foundation -framework Metal -framework MetalPerformanceShaders probe.mm -o probe
./probe original-translated.metal
