extern "C" __global__ void tripleSum(const float *a, const float *b, float *out, int n) {
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        out[i] = __fadd_rn(__fadd_rn(a[i], b[i]), b[i]);
    }
}
