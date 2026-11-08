#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef unsigned long long deviceptr;
extern int cuInit(unsigned);
extern int cuDeviceGetCount(int *);
extern int cuDevicePrimaryCtxRetain(void **,int);
extern int cuDevicePrimaryCtxRelease_v2(int);
extern int cuCtxSetCurrent(void *);
extern int cuCtxSynchronize(void);
extern int cuMemAlloc_v2(deviceptr *,size_t);
extern int cuMemFree_v2(deviceptr);
extern int cuMemcpyHtoD_v2(deviceptr,const void *,size_t);
extern int cuMemcpyDtoH_v2(void *,deviceptr,size_t);
extern int cuModuleLoadData(void **,const void *);
extern int cuModuleGetFunction(void **,void *,const char *);
extern int cuLaunchKernel(void *,unsigned,unsigned,unsigned,unsigned,unsigned,unsigned,unsigned,void *,void **,void **);
extern int cuModuleUnload(void *);
static const char ptx[]=
".version 7.0\n"
".target sm_80\n"
".address_size 64\n"
".visible .entry bridge_add(\n"
" .param .u64 .ptr .global a,\n"
" .param .u64 .ptr .global b,\n"
" .param .u64 .ptr .global out\n"
") {\n"
" .reg .b32 %r<5>;\n"
" .reg .b64 %d<8>;\n"
" ld.param.u64 %d0, [a];\n"
" ld.param.u64 %d1, [b];\n"
" ld.param.u64 %d2, [out];\n"
" mov.u32 %r0, %tid.x;\n"
" mul.wide.u32 %d3, %r0, 4;\n"
" add.u64 %d4, %d0, %d3;\n"
" add.u64 %d5, %d1, %d3;\n"
" add.u64 %d6, %d2, %d3;\n"
" ld.global.u32 %r1,[%d4];\n"
" ld.global.u32 %r2,[%d5];\n"
" add.u32 %r3,%r1,%r2;\n"
" st.global.u32 [%d6],%r3;\n"
" ret;\n"
"}\n";
#define CHECK(call) do {int r=(call); printf("%s => %d\n",#call,r); if(r) return 1;}while(0)

static int launch(deviceptr a,deviceptr b,deviceptr c) {
    void *module=NULL,*fn=NULL;
    CHECK(cuModuleLoadData(&module,ptx));
    CHECK(cuModuleGetFunction(&fn,module,"bridge_add"));
    void *params[]={&a,&b,&c};
    CHECK(cuLaunchKernel(fn,1,1,1,3,1,1,0,NULL,params,NULL));
    CHECK(cuCtxSynchronize());
    CHECK(cuModuleUnload(module));
    return 0;
}
static int verify(const int *values) {
    printf("value=[%d,%d,%d]\n",values[0],values[1],values[2]);
    return !(values[0]==5 && values[1]==7 && values[2]==9);
}
static int run_driver(void) {
    const int a[3]={1,2,3}, b[3]={4,5,6}; int values[3]={0}, count=0;
    deviceptr da=0,db=0,dc=0; void *ctx=NULL;
    CHECK(cuInit(0));
    CHECK(cuDeviceGetCount(&count));
    if(count!=1) return 1;
    CHECK(cuDevicePrimaryCtxRetain(&ctx,0)); CHECK(cuCtxSetCurrent(ctx));
    CHECK(cuMemAlloc_v2(&da,sizeof a)); CHECK(cuMemAlloc_v2(&db,sizeof b));
    CHECK(cuMemAlloc_v2(&dc,sizeof values));
    CHECK(cuMemcpyHtoD_v2(da,a,sizeof a)); CHECK(cuMemcpyHtoD_v2(db,b,sizeof b));
    if(launch(da,db,dc)) return 1;
    CHECK(cuMemcpyDtoH_v2(values,dc,sizeof values)); int result=verify(values);
    CHECK(cuMemFree_v2(dc)); CHECK(cuMemFree_v2(db)); CHECK(cuMemFree_v2(da));
    CHECK(cuCtxSetCurrent(NULL)); CHECK(cuDevicePrimaryCtxRelease_v2(0));
    return result;
}
static int run_runtime(const char *path,int memory_only) {
    void *lib=dlopen(path,RTLD_NOW|RTLD_LOCAL);
    if(!lib){fprintf(stderr,"dlopen: %s\n",dlerror());return 1;}
    int (*runtime_version)(int*)=dlsym(lib,"cudaRuntimeGetVersion");
    int (*runtime_alloc)(void**,size_t)=dlsym(lib,"cudaMalloc");
    int (*runtime_copy)(void*,const void*,size_t,int)=dlsym(lib,"cudaMemcpy");
    int (*runtime_free)(void*)=dlsym(lib,"cudaFree");
    if(!runtime_version || !runtime_alloc || !runtime_copy || !runtime_free) return 1;
    int version=0; CHECK(runtime_version(&version)); printf("original-runtime-version=%d\n",version);
    Dl_info resolved;
    if(!dladdr((void*)runtime_alloc,&resolved))return 1;
    printf("cudaMalloc-source=%s\n",resolved.dli_fname);
    const int a[]={1,2,3},b[]={4,5,6}; int values[3]={0}; void *da=NULL,*db=NULL,*dc=NULL;
    CHECK(runtime_alloc(&da,sizeof a));
    CHECK(runtime_copy(da,a,sizeof a,1)); CHECK(runtime_copy(values,da,sizeof a,2));
    if(memcmp(a,values,sizeof a))return 1;
    printf("original-runtime-copy-verified\n");
    if(memory_only) {CHECK(runtime_free(da)); return 0;}
    CHECK(runtime_alloc(&db,sizeof b)); CHECK(runtime_alloc(&dc,sizeof values));
    CHECK(runtime_copy(db,b,sizeof b,1));

    if(launch((uintptr_t)da,(uintptr_t)db,(uintptr_t)dc)) return 1;
    CHECK(runtime_copy(values,dc,sizeof values,2)); int result=verify(values);
    CHECK(runtime_free(dc));CHECK(runtime_free(db));CHECK(runtime_free(da));
    return result;
}

int main(int argc,char **argv) {
    setvbuf(stdout,NULL,_IONBF,0);
    if(argc==2 && !strcmp(argv[1],"driver")) return run_driver();
    if(argc==3 && !strcmp(argv[1],"runtime")) return run_runtime(argv[2],0);
    if(argc==3 && !strcmp(argv[1],"runtime-mem")) return run_runtime(argv[2],1);
    fprintf(stderr,"usage: probe driver | probe runtime[-mem] /path/libcudart.so\n");return 2;
}
