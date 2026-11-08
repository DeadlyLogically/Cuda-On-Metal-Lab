#include <dlfcn.h>
#include <nvrtc.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void require_nvrtc(nvrtcResult result) {
    if (result != NVRTC_SUCCESS) {
        fprintf(stderr, "%s\n", nvrtcGetErrorString(result));
        exit(1);
    }
}

int main(void) {
    char *(*lookup)(const void *) = dlsym(RTLD_DEFAULT, "bridge_nvrtc_image_ptx");
    if (!lookup) {
        fputs("Load the companion library with LD_PRELOAD.\n", stderr);
        return 1;
    }
    const char source[] = "extern \"C\" __global__ void twice(float *x) { x[0] *= 2; }";
    const char *options[] = {"--gpu-architecture=sm_80"};
    nvrtcProgram program;
    require_nvrtc(nvrtcCreateProgram(&program, source, "twice.cu", 0, NULL, NULL));
    require_nvrtc(nvrtcCompileProgram(program, 1, options));
    size_t image_size, ptx_size;
    require_nvrtc(nvrtcGetCUBINSize(program, &image_size));
    require_nvrtc(nvrtcGetPTXSize(program, &ptx_size));
    char *image = malloc(image_size);
    char *expected = malloc(ptx_size);
    if (!image || !expected || image_size == 0) return 1;
    require_nvrtc(nvrtcGetCUBIN(program, image));
    require_nvrtc(nvrtcGetPTX(program, expected));
    char *matched = lookup(image);
    int equal = matched && strcmp(matched, expected) == 0;
    image[image_size - 1] ^= 1;
    char *unknown = lookup(image);
    int rejected = unknown == NULL;
    free(image);
    free(expected);
    free(matched);
    free(unknown);
    require_nvrtc(nvrtcDestroyProgram(&program));
    if (!equal || !rejected) return 1;
    puts("Matched the compiler's PTX and rejected the changed image.");
    return 0;
}
