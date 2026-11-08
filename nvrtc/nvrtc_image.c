#define _GNU_SOURCE
#include "nvrtc_image.h"
#include <dlfcn.h>
#include <elf.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/uio.h>
#include <unistd.h>

typedef int (*nvrtc_get_size)(void *,size_t *);
typedef int (*nvrtc_get_code)(void *,char *);

enum { CACHE_LIMIT = 32, BYTE_BUDGET = 32u << 20, PTX_LIMIT = 1u << 20 };
struct image_pair {
    char *cubin, *ptx;
    size_t cubin_size, ptx_size;
};
static struct image_pair cache[CACHE_LIMIT];
static size_t cached_bytes, next_slot;
static pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;

static void forget(struct image_pair *slot) {
    cached_bytes -= slot->cubin_size + slot->ptx_size;
    free(slot->cubin);
    free(slot->ptx);
    *slot = (struct image_pair){0};
}

static void remember(struct image_pair pair) {
    pthread_mutex_lock(&lock);
    for (size_t i=0; i<CACHE_LIMIT; i++) {
        struct image_pair *old = &cache[i];
        if (old->cubin_size != pair.cubin_size || memcmp(old->cubin,pair.cubin,pair.cubin_size)) continue;
        
        if (old->ptx && (old->ptx_size != pair.ptx_size || memcmp(old->ptx,pair.ptx,old->ptx_size))) {
            cached_bytes -= old->ptx_size;
            free(old->ptx);
            old->ptx = NULL;
            old->ptx_size = 0;
        }
        free(pair.cubin);
        free(pair.ptx);
        pthread_mutex_unlock(&lock);
        return;
    }
    do {
        forget(&cache[next_slot]);
        if (cached_bytes + pair.cubin_size + pair.ptx_size <= BYTE_BUDGET) break;
        next_slot = (next_slot+1) % CACHE_LIMIT;
    } while (1);
    cache[next_slot] = pair;
    cached_bytes += pair.cubin_size + pair.ptx_size;
    next_slot = (next_slot+1) % CACHE_LIMIT;
    pthread_mutex_unlock(&lock);
}

int nvrtcGetCUBIN(void *program,char *cubin) {
    nvrtc_get_code get_cubin = (nvrtc_get_code)dlsym(RTLD_NEXT,"nvrtcGetCUBIN");
    if (!get_cubin) return 11; 
    int status = get_cubin(program,cubin);
    if (status != 0 || getenv("CUMETAL_NVRTC_DISABLE_COMPANIONS")) return status;
    nvrtc_get_size cubin_size = (nvrtc_get_size)dlsym(RTLD_NEXT,"nvrtcGetCUBINSize");
    nvrtc_get_size ptx_size = (nvrtc_get_size)dlsym(RTLD_NEXT,"nvrtcGetPTXSize");
    nvrtc_get_code get_ptx = (nvrtc_get_code)dlsym(RTLD_NEXT,"nvrtcGetPTX");
    struct image_pair pair = {0};
    if (!cubin_size || !ptx_size || !get_ptx || cubin_size(program,&pair.cubin_size) ||
        ptx_size(program,&pair.ptx_size) || pair.cubin_size < sizeof(Elf64_Ehdr) ||
        pair.cubin_size > BYTE_BUDGET || pair.ptx_size < 2 || pair.ptx_size > PTX_LIMIT ||
        pair.cubin_size > BYTE_BUDGET-pair.ptx_size) return status;
    if (memcmp(cubin,ELFMAG,SELFMAG)) return status;
    pair.ptx = malloc(pair.ptx_size);
    if (!pair.ptx) return status;
    if (get_ptx(program,pair.ptx) || strnlen(pair.ptx,pair.ptx_size) != pair.ptx_size-1) {
        free(pair.ptx);
        return status;
    }
    pair.cubin = malloc(pair.cubin_size);
    if (!pair.cubin) { free(pair.ptx); return status; }
    memcpy(pair.cubin,cubin,pair.cubin_size);
    if (getenv("CUMETAL_BENCH_TRACE")) fprintf(stderr,"NVRTC emitted cubin=%zu ptx=%zu\n",pair.cubin_size,pair.ptx_size);
    remember(pair);
    return status;
}

static int readable_copy(void *dest,const void *source,size_t n) {
    struct iovec local = {dest,n}, remote = {(void*)source,n};
    return process_vm_readv(getpid(),&local,1,&remote,1,0) == (ssize_t)n;
}
char *bridge_nvrtc_image_ptx(const void *image) {
    Elf64_Ehdr header;
    if (!image || !readable_copy(&header,image,sizeof(header)) ||
        memcmp(header.e_ident,ELFMAG,SELFMAG) || header.e_ident[EI_CLASS] != ELFCLASS64 ||
        header.e_ident[EI_DATA] != ELFDATA2LSB || header.e_machine != EM_CUDA) return NULL;
    char *ptx = NULL;
    pthread_mutex_lock(&lock);
    for (size_t i=0; i<CACHE_LIMIT; i++) {
        const struct image_pair *pair = &cache[i];
        if (!pair->ptx || memcmp(&header,pair->cubin,sizeof(header))) continue;
        char *bytes = malloc(pair->cubin_size);
        if (!bytes) break;
        int equal = readable_copy(bytes,image,pair->cubin_size) && !memcmp(bytes,pair->cubin,pair->cubin_size);
        free(bytes);
        if (!equal) continue;
        ptx = strdup(pair->ptx);
        if (ptx && getenv("CUMETAL_BENCH_TRACE"))
            fprintf(stderr,"NVRTC matched exact cubin=%zu to ptx=%zu\n",pair->cubin_size,pair->ptx_size);
        break;
    }
    pthread_mutex_unlock(&lock);
    return ptx;
}
