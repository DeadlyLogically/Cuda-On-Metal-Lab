#define main llmc_original_attention_main_not_run
#include "upstream/attention_forward.cu"
#undef main

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

namespace {
constexpr size_t kGuard = 32;
constexpr float kSentinel = -12345.25f;
size_t total_values = 0;
int total_cases = 0;
double maximum_score_error = 0;
double maximum_weight_error = 0;
double maximum_output_error = 0;
double maximum_row_sum_error = 0;

struct GuardedBuffer {
    std::vector<float> host;
    float* device = nullptr;
    size_t count;

    explicit GuardedBuffer(size_t n) : host(n + 2 * kGuard, kSentinel), count(n) {
        std::fill(host.begin() + kGuard, host.end() - kGuard,
                  std::numeric_limits<float>::quiet_NaN());
        cudaCheck(cudaMalloc(&device, host.size() * sizeof(float)));
    }
    ~GuardedBuffer() { if (device) cudaFree(device); }
    float* data() { return device + kGuard; }
    void upload() {
        cudaCheck(cudaMemcpy(device, host.data(), host.size() * sizeof(float),
                             cudaMemcpyHostToDevice));
    }
    void download() {
        cudaCheck(cudaMemcpy(host.data(), device, host.size() * sizeof(float),
                             cudaMemcpyDeviceToHost));
    }
    bool guards_ok(const char* name) const {
        for (size_t i = 0; i < kGuard; ++i) {
            if (host[i] != kSentinel || host[kGuard + count + i] != kSentinel) {
                std::fprintf(stderr, "FAIL: %s guard changed at %zu\n", name, i);
                return false;
            }
        }
        return true;
    }
};

void reference_attention(const float* input, int b_count, int t_count, int c_count,
                         int heads, std::vector<double>& scores,
                         std::vector<double>& weights, std::vector<double>& output) {
    const int head_size = c_count / heads;
    const double scale = 1.0 / std::sqrt(double(head_size));
    const auto qkv_index = [=](int b, int t, int part, int h, int channel) {
        return (size_t(b) * t_count + t) * (3 * c_count)
             + part * c_count + h * head_size + channel;
    };
    for (int b = 0; b < b_count; ++b) {
        for (int h = 0; h < heads; ++h) {
            for (int t = 0; t < t_count; ++t) {
                const size_t row = ((size_t(b) * heads + h) * t_count + t) * t_count;
                double maximum = -std::numeric_limits<double>::infinity();
                for (int key = 0; key <= t; ++key) {
                    double score = 0;
                    for (int channel = 0; channel < head_size; ++channel) {
                        score += double(input[qkv_index(b, t, 0, h, channel)])
                               * double(input[qkv_index(b, key, 1, h, channel)]);
                    }
                    scores[row + key] = score * scale;
                    maximum = std::max(maximum, scores[row + key]);
                }
                double denominator = 0;
                for (int key = 0; key <= t; ++key) {
                    weights[row + key] = std::exp(scores[row + key] - maximum);
                    denominator += weights[row + key];
                }
                for (int key = 0; key <= t; ++key) weights[row + key] /= denominator;
                for (int channel = 0; channel < head_size; ++channel) {
                    double value = 0;
                    for (int key = 0; key <= t; ++key) {
                        value += weights[row + key]
                               * double(input[qkv_index(b, key, 2, h, channel)]);
                    }
                    output[(size_t(b) * t_count + t) * c_count
                           + h * head_size + channel] = value;
                }
            }
        }
    }
}

bool close_value(const char* what, size_t index, double actual, double expected,
                 double absolute_tolerance, double relative_tolerance,
                 double& maximum_error) {
    const double error = std::fabs(actual - expected);
    if (std::isfinite(error)) maximum_error = std::max(maximum_error, error);
    if (!std::isfinite(actual) ||
        error > absolute_tolerance + relative_tolerance * std::fabs(expected)) {
        std::fprintf(stderr, "FAIL: %s[%zu] got=%.12g expected=%.12g error=%.12g\n",
                     what, index, actual, expected, error);
        return false;
    }
    return true;
}

bool run_case(int b, int t, int c, int heads, int block, int pattern) {
    const size_t input_count = size_t(b) * t * 3 * c;
    const size_t output_count = size_t(b) * t * c;
    const size_t matrix_count = size_t(b) * heads * t * t;
    GuardedBuffer input(input_count), scores(matrix_count), weights(matrix_count), output(output_count);
    uint32_t state = 0x9e3779b9u;
    for (size_t i = 0; i < input_count; ++i) {
        state = state * 1664525u + 1013904223u;
        float value = float(int(state >> 8) % 20001 - 10000) / 10000.f;
        const bool is_query_or_key = (i % size_t(3 * c)) < size_t(2 * c);
        if (pattern == 1 && is_query_or_key) value = 0;
        if (pattern == 2 && is_query_or_key) value *= 4.f;
        input.host[kGuard + i] = value;
    }
    const std::vector<float> original_input = input.host;
    std::vector<double> expected_scores(matrix_count, -std::numeric_limits<double>::infinity());
    std::vector<double> expected_weights(matrix_count, 0);
    std::vector<double> expected_output(output_count, 0);
    reference_attention(input.host.data() + kGuard, b, t, c, heads,
                        expected_scores, expected_weights, expected_output);
    input.upload(); scores.upload(); weights.upload(); output.upload();
    std::printf("CASE attention b=%d t=%d c=%d heads=%d block=%d pattern=%d\n",
                b, t, c, heads, block, pattern);
    std::fflush(stdout);

    attention_forward1(output.data(), scores.data(), weights.data(), input.data(),
                       b, t, c, heads, block);
    cudaCheck(cudaGetLastError());
    cudaCheck(cudaDeviceSynchronize());
    input.download(); scores.download(); weights.download(); output.download();
    if (std::memcmp(input.host.data(), original_input.data(),
                    original_input.size() * sizeof(float)) != 0) {
        std::fprintf(stderr, "FAIL: attention input changed\n"); return false;
    }
    if (!input.guards_ok("input") || !scores.guards_ok("scores") ||
        !weights.guards_ok("weights") || !output.guards_ok("output")) return false;
    for (size_t i = 0; i < matrix_count; ++i) {
        const int query = int((i / t) % t), key = int(i % t);
        const double actual_score = scores.host[kGuard + i];
        const double actual_weight = weights.host[kGuard + i];
        if (key > query) {
            if (!(std::isinf(actual_score) && actual_score < 0) || actual_weight != 0) {
                std::fprintf(stderr, "FAIL: causal mask at %zu score=%g weight=%g\n",
                             i, actual_score, actual_weight);
                return false;
            }
        } else {

            if (!close_value("score", i, actual_score, expected_scores[i],
                             1e-5, 1e-5, maximum_score_error)) return false;
            if (actual_weight < 0 ||
                !close_value("weight", i, actual_weight, expected_weights[i],
                             3e-6, 2e-5, maximum_weight_error)) return false;
        }
    }
    for (size_t row = 0; row < size_t(b) * heads * t; ++row) {
        double sum = 0;
        for (int key = 0; key < t; ++key) sum += weights.host[kGuard + row * t + key];
        const double error = std::fabs(sum - 1);
        maximum_row_sum_error = std::max(maximum_row_sum_error, error);
        if (!std::isfinite(sum) || error > 5e-6) {
            std::fprintf(stderr, "FAIL: attention row %zu sum %.12g\n", row, sum);
            return false;
        }
    }
    for (size_t i = 0; i < output_count; ++i) {
        if (!close_value("output", i, output.host[kGuard + i], expected_output[i],
                         1e-5, 2e-5, maximum_output_error)) return false;
    }
    ++total_cases;
    total_values += output_count + 2 * matrix_count;
    return true;
}
}   

int main(int argc, char** argv) {
    cudaCheck(cudaSetDevice(0));
    if (argc > 1 && std::strcmp(argv[1], "smoke") == 0) {
        if (!run_case(1, 3, 15, 3, 32, 0)) return 1;
    } else {
        struct Shape { int b, t, c, heads; };
        const Shape shapes[] = {
            {1, 1, 1, 1}, {1, 3, 15, 3}, {2, 17, 28, 4},
            {1, 33, 15, 3}, {2, 33, 28, 4}, {1, 65, 64, 4},
        };
        for (const auto shape : shapes) {
            for (int block : {32, 64, 128, 256}) {
                for (int pattern = 0; pattern < 3; ++pattern) {
                    if (!run_case(shape.b, shape.t, shape.c, shape.heads, block, pattern)) return 1;
                }
            }
        }
    }
    std::printf("AUTO_ATTENTION_OK cases=%d launches=%d values=%zu max_score_error=%.12g max_weight_error=%.12g max_output_error=%.12g max_row_sum_error=%.12g\n",
                total_cases, total_cases * 3, total_values, maximum_score_error,
                maximum_weight_error, maximum_output_error, maximum_row_sum_error);
    return 0;
}
