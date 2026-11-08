# Matching a binary to its own PTX

Some CUDA applications compile GPU code during execution with NVIDIA's NVRTC compiler. This Linux component intercepts `nvrtcGetCUBIN` and returns the original binary to the caller. When the same compilation also supplies PTX, the component retains both in process memory. The driver can later ask for that PTX using an exact byte match on the binary. This route and the [T4 instruction translator](../sass/) solve different cases.

The original NVIDIA vector-add application passed its 50,000-element check in a separate Linux bridge experiment and issued a Metal dispatch on an M2. The same executable stopped before dispatch when companion collection was disabled. This directory contains the matching component and a native Linux test. The [recorded end-to-end results](sample-results.json) came from the fuller NVRTC bridge.

## Run the component check

On Linux ARM64, install a C compiler and an NVRTC development package that targets `sm_80`. The experiment used NVRTC 13.0.88. From this directory, with `CUDA_HOME` pointing at your installation, run:

```sh
cc -O2 -fPIC -shared nvrtc_image.c -o companion.so -ldl -pthread
cc -O2 check_companion.c -I"$CUDA_HOME/include" -L"$CUDA_HOME/lib64" -lnvrtc -ldl -o check_companion
LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" LD_PRELOAD="$PWD/companion.so" ./check_companion
```

The check compiles a CUDA kernel with the real NVRTC library, compares the cached PTX with the compiler's own output and changes one byte to verify that the lookup rejects the altered image. This check exercises the matching logic and requires no physical GPU.
