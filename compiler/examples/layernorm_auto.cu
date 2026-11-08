#define main llmc_original_main_not_run
#include "upstream/layernorm_forward.cu"
#undef main
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

namespace {
constexpr size_t guard = 32;
constexpr float sentinel = -12345.25f;
size_t checked_values = 0;
int cases = 0;
double max_out_error = 0, max_mean_error = 0, max_rstd_error = 0;

struct Buffer {
    size_t count;
    std::vector<float> host;
    float* allocation = nullptr;
    explicit Buffer(size_t n): count(n), host(n + 2 * guard, sentinel) {
        cudaCheck(cudaMalloc(&allocation, host.size() * sizeof(float)));
    }
    ~Buffer() { cudaFree(allocation); }
    float* device() { return allocation + guard; }
    float& operator[](size_t i) { return host[guard + i]; }
    void upload() { cudaCheck(cudaMemcpy(allocation, host.data(), host.size() * sizeof(float), cudaMemcpyHostToDevice)); }
    void poison() { std::fill(host.begin() + guard, host.end() - guard, std::numeric_limits<float>::quiet_NaN()); upload(); }
    void read() { cudaCheck(cudaMemcpy(host.data(), allocation, host.size() * sizeof(float), cudaMemcpyDeviceToHost)); }
    bool guards() const {
        for (size_t i = 0; i < guard; ++i)
            if (host[i] != sentinel || host[guard + count + i] != sentinel) return false;
        return true;
    }
    bool unchanged() {
        auto expected = host;
        read();
        return std::memcmp(expected.data(), host.data(), host.size() * sizeof(float)) == 0;
    }
};

bool compare(double actual, double expected, double& largest, const char* name, size_t index) {
    const double error = std::fabs(actual - expected);
    largest = std::max(largest, error);
    if (!std::isfinite(actual) || error > 2e-5 + 3e-5 * std::fabs(expected)) {
        std::fprintf(stderr, "FAIL %s[%zu]: got=%.12g expected=%.12g error=%.12g\n", name, index, actual, expected, error);
        return false;
    }
    ++checked_values;
    return true;
}

bool check_case(int version, int b, int t, int c, int block, int pattern, int absent = 0) {
    const int n = b * t;
    Buffer input(size_t(n) * c), weight(c), bias(c), out(size_t(n) * c), mean(n), rstd(n);
    uint32_t state = 0x12345678u;
    for (int i = 0; i < n * c; ++i) {
        state = state * 1664525u + 1013904223u;
        float x = float(int(state >> 8) % 6145 - 3072) / 1024.f;
        if (pattern == 1) x = 3.25f;
        if (pattern == 2) x = float(i % 17 - 8) / 16384.f;
        input[i] = x;
    }
    for (int i = 0; i < c; ++i) {
        weight[i] = float(i % 13 - 6) / 4.f;
        bias[i] = float(i % 11 - 5) / 8.f;
    }
    std::vector<double> ref_out(size_t(n) * c), ref_mean(n), ref_rstd(n);
    for (int row = 0; row < n; ++row) {
        double m = 0, variance = 0;
        for (int i = 0; i < c; ++i) m += input[size_t(row) * c + i];
        m /= c;
        for (int i = 0; i < c; ++i) {
            double d = double(input[size_t(row) * c + i]) - m;
            variance += d * d;
        }
        double s = 1.0 / std::sqrt(variance / c + double(1e-5f));
        ref_mean[row] = m; ref_rstd[row] = s;
        for (int i = 0; i < c; ++i)
            ref_out[size_t(row) * c + i] = (double(input[size_t(row) * c + i]) - m) * s * weight[i] + bias[i];
    }
    input.upload(); weight.upload(); bias.upload(); out.poison(); mean.poison(); rstd.poison();
    std::printf("CASE layernorm v=%d B=%d T=%d C=%d block=%d pattern=%d absent=%d\n", version, b, t, c, block, pattern, absent);
    std::fflush(stdout);
    float* mean_ptr = absent & 1 ? nullptr : mean.device();
    float* rstd_ptr = absent & 2 ? nullptr : rstd.device();
    if (version == 1)
        layernorm_forward1(out.device(), mean_ptr, rstd_ptr, input.device(), weight.device(), bias.device(), b, t, c, block);
    else
        layernorm_forward3(out.device(), mean_ptr, rstd_ptr, input.device(), weight.device(), bias.device(), b, t, c, block);
    cudaCheck(cudaGetLastError());
    cudaCheck(cudaDeviceSynchronize());
    out.read(); mean.read(); rstd.read();
    if (!input.unchanged() || !weight.unchanged() || !bias.unchanged() ||
        !out.guards() || !mean.guards() || !rstd.guards()) {
        std::fprintf(stderr, "FAIL: input/parameters/guards changed\n"); return false;
    }
    for (size_t i = 0; i < ref_out.size(); ++i)
        if (!compare(out[i], ref_out[i], max_out_error, "output", i)) return false;
    for (int i = 0; i < n; ++i) {
        if (!(absent & 1) && !compare(mean[i], ref_mean[i], max_mean_error, "mean", i)) return false;
        if (!(absent & 2) && !compare(rstd[i], ref_rstd[i], max_rstd_error, "rstd", i)) return false;
        if (((absent & 1) && !std::isnan(mean[i])) || ((absent & 2) && !std::isnan(rstd[i]))) {
            std::fprintf(stderr, "FAIL: omitted output changed\n"); return false;
        }
    }
    ++cases;
    return true;
}
}

int main(int argc, char** argv) {
    cudaCheck(cudaSetDevice(0));
    if (argc > 1 && std::strcmp(argv[1], "smoke") == 0) {
        if (!check_case(1, 1, 3, 33, 64, 0)) return 1;
    } else {
        for (int c : {1, 7, 31, 32, 33, 127, 257, 768, 1031}) {
            int b = c % 2 ? 1 : 3, t = c % 2 ? 7 : 3;
            for (int block : {32, 64, 128, 256})
                for (int pattern = 0; pattern < 3; ++pattern)
                    if (!check_case(3, b, t, c, block, pattern)) return 1;
        }
        for (int absent = 1; absent <= 3; ++absent)
            if (!check_case(3, 1, 3, 33, 128, 0, absent)) return 1;
    }
    std::printf("AUTO_LAYERNORM_OK cases=%d launches=%d values=%zu max_out_error=%.12g max_mean_error=%.12g max_rstd_error=%.12g\n",
                cases, cases, checked_values, max_out_error, max_mean_error, max_rstd_error);
}
