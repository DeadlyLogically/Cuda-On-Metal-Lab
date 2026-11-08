# GPT-2 output projection on M2

This benchmark compares two implementations of the same FP32 projection on one Apple M2. The left matrix is 128 × 768. The weight matrix is 50,304 × 768 and is read transposed. There is no bias. Both implementations read the same generated inputs, already resident in shared GPU buffers, and write separate outputs.

`original-translated.metal` contains the emitted shader from an actual CuMetal run of the original `llm.c` training kernel. `probe.mm` encodes either that shader or Apple's `MPSMatrixMultiplication` and uses Metal command-buffer timestamps. Preparation and compilation are outside the timed region. Run it on an Apple silicon Mac with the Xcode command line tools:

```sh
cd benchmarks/matmul
./run.sh
```

The output includes three warm-up pairs and twelve measured pairs, alternating which implementation runs first. The final line reports the number of values checked, failures, maximum and RMS differences, and the largest errors at 128 points checked in CPU double precision. The published training benchmark and this isolated probe use different inputs. [Recorded medians and tolerances](comparison.json) · [Raw paired times](raw.csv)
