#define main llmc_original_main_not_run
#include "upstream/softmax_forward.cu"
#undef main

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

static size_t total_values = 0;
static int total_launches = 0;
static double max_abs_error = 0.0;
static double max_row_error = 0.0;

static bool check_case(int n, int c, int block, int pattern) {
    const size_t count = size_t(n) * size_t(c);
    constexpr size_t guard = 32;
    constexpr float sentinel = -12345.25f;
    std::vector<float> input(count + guard * 2, sentinel);
    std::vector<float> output(count + guard * 2, sentinel);
    std::vector<double> reference(count);
    uint32_t state = 0x9e3779b9u;
    for (int row = 0; row < n; ++row) {
        for (int col = 0; col < c; ++col) {
            state = state * 1664525u + 1013904223u;
            float x = float(int(state >> 8) % 20001 - 10000) / 1000.f;
            if (pattern == 1) x = -10000.f;
            if (pattern == 2) x = 10000.f + float((col + row) % 31 - 15);
            if (pattern == 3) x = -1.0e35f;
            input[guard + size_t(row) * c + col] = x;
        }
        double maximum = -std::numeric_limits<double>::infinity();
        for (int col = 0; col < c; ++col)
            maximum = std::max(maximum, double(input[guard + size_t(row) * c + col]));
        double sum = 0;
        for (int col = 0; col < c; ++col) {
            double x = std::exp(double(input[guard + size_t(row) * c + col]) - maximum);
            reference[size_t(row) * c + col] = x;
            sum += x;
        }
        for (int col = 0; col < c; ++col) reference[size_t(row) * c + col] /= sum;
    }
    std::fill(output.begin() + guard, output.end() - guard,
              std::numeric_limits<float>::quiet_NaN());
    float *device_in = nullptr, *device_out = nullptr;
    const size_t bytes = input.size() * sizeof(float);
    cudaCheck(cudaMalloc(&device_in, bytes));
    cudaCheck(cudaMalloc(&device_out, bytes));
    cudaCheck(cudaMemcpy(device_in, input.data(), bytes, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(device_out, output.data(), bytes, cudaMemcpyHostToDevice));
    printf("CASE n=%d c=%d block=%d pattern=%d\n", n, c, block, pattern);
    fflush(stdout);
    softmax_forward2(device_out + guard, device_in + guard, n, c, block);
    cudaCheck(cudaGetLastError());
    cudaCheck(cudaDeviceSynchronize());
    cudaCheck(cudaMemcpy(output.data(), device_out, bytes, cudaMemcpyDeviceToHost));
    std::vector<float> input_after(input.size());
    cudaCheck(cudaMemcpy(input_after.data(), device_in, bytes, cudaMemcpyDeviceToHost));
    cudaCheck(cudaFree(device_out));
    cudaCheck(cudaFree(device_in));
    if (std::memcmp(input.data(), input_after.data(), bytes) != 0) {
        fprintf(stderr, "FAIL: input or input guards modified\n"); return false;
    }
    for (size_t i = 0; i < guard; ++i) {
        if (output[i] != sentinel || output[guard + count + i] != sentinel) {
            fprintf(stderr, "FAIL: output guard modified\n"); return false;
        }
    }
    for (int row = 0; row < n; ++row) {
        double sum = 0;
        for (int col = 0; col < c; ++col) {
            const size_t i = size_t(row) * c + col;
            const double actual = output[guard + i], expected = reference[i];
            const double error = std::fabs(actual - expected);
            max_abs_error = std::max(max_abs_error, error);
            if (!std::isfinite(actual) || actual < 0 || error > 3e-6 + 4e-6 * std::fabs(expected)) {
                fprintf(stderr, "FAIL: row=%d col=%d got=%.12g expected=%.12g\n", row, col, actual, expected);
                return false;
            }
            sum += actual;
        }
        max_row_error = std::max(max_row_error, std::fabs(sum - 1.0));
        if (std::fabs(sum - 1.0) > 3e-5) {
            fprintf(stderr, "FAIL: row=%d sum=%.12g\n", row, sum); return false;
        }
    }
    total_values += count;
    ++total_launches;
    return true;
}

int main(int argc, char** argv) {
    cudaCheck(cudaSetDevice(0));
    if (argc > 1 && std::strcmp(argv[1], "smoke") == 0) {
        if (!check_case(3, 33, 64, 0)) return 1;
    } else {
        for (int c : {1, 7, 31, 32, 33, 127, 257, 1031, 4099, 50257}) {
            const int n = c == 50257 ? 2 : (c % 3 == 0 ? 3 : 7);
            for (int block : {32, 64, 128, 256, 512, 1024}) {
                for (int pattern = 0; pattern < 4; ++pattern) {
                    if (!check_case(n, c, block, pattern)) return 1;
                }
            }
        }
    }
    printf("AUTO_SOFTMAX_OK launches=%d values=%zu max_abs_error=%.12g max_row_sum_error=%.12g\n",
           total_launches, total_values, max_abs_error, max_row_error);
    return 0;
}
