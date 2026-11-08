extern "C" __global__ void atomicSum(const float *a, float *out, int n) {
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        atomicAdd(out + i, a[i]);
    }
}
