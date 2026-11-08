extern "C" __global__ void elementMultiply(const float *a, const float *b, float *out, int n) {
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        out[i] = a[i] * b[i];
    }
}
