#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include "bridge_md2.h"

extern int cuDriverGetVersion(int *);
extern int cuDeviceGetCount(int *);
extern int cuDeviceGet(int *, int);
extern int cuDeviceGetAttribute(int *,int,int);
extern int cuDeviceGetUuid(void *,int);
extern int cuDevicePrimaryCtxRetain(void **,int);
extern int cuDevicePrimaryCtxSetFlags_v2(int,unsigned);
extern int cuCtxGetCurrent(void **);
enum { INVALID=1, INVALID_DEVICE=101, INVALID_CONTEXT=201, INVALID_HANDLE=400, UNSUPPORTED=801 };

static const uintptr_t cudart_interface[], integrity_interface[];
static int unsupported(void){return UNSUPPORTED;}
static pthread_mutex_t context_lock=PTHREAD_MUTEX_INITIALIZER;
static void *runtime_context;
static int get_primary_context(void **out,int device) {
    if (!out) return INVALID;
    if (device!=0) return INVALID_DEVICE;
    pthread_mutex_lock(&context_lock);
    int status=runtime_context?0:cuDevicePrimaryCtxRetain(&runtime_context,device);
    if (!status) *out=runtime_context;
    pthread_mutex_unlock(&context_lock);
    return status;
}


static uint32_t hook_flags[1024], hook_state[14];
static void hook_buffer1(void **out,size_t *entries) {
    *out=hook_flags; *entries=sizeof(hook_flags)/sizeof(hook_flags[0]);
}
static void hook_buffer2(void **out,size_t *entries) {
    *out=hook_state; *entries=sizeof(hook_state)/sizeof(hook_state[0]);
}
static unsigned context_check_feature(void) { return 0; }


typedef void (*destroy_cb)(void *,void *,void *);
struct cls_record {void *ctx,*key,*value; destroy_cb destructor;};
static struct cls_record records[64];
static pthread_mutex_t cls_lock=PTHREAD_MUTEX_INITIALIZER;
static int resolve_context(void **ctx) {
    if (!*ctx) {
        int err=cuCtxGetCurrent(ctx);
        if(err) return err;
    }
    return *ctx && *ctx==runtime_context ? 0 : INVALID_CONTEXT;
}

static int check_context_alias(void *ctx, unsigned *remapped, void **alternative) {
    if (!remapped || !alternative) return INVALID;
    int status=resolve_context(&ctx);
    if(status) return status;
    *remapped=0; *alternative=NULL;
    return 0;
}

static int cls_put(void *ctx,void *key,void *value,destroy_cb destructor) {
    int err=resolve_context(&ctx);
    if(err) return err;
    if(!key) return INVALID;
    pthread_mutex_lock(&cls_lock);
    struct cls_record *empty=NULL;
    for(size_t i=0;i<sizeof(records)/sizeof(*records);++i){
        if(records[i].ctx==ctx && records[i].key==key) {
            pthread_mutex_unlock(&cls_lock);
            return INVALID_HANDLE;
        }
        if(!records[i].ctx && !empty) empty=&records[i];
    }
    if(empty) *empty=(struct cls_record){ctx,key,value,destructor};
    pthread_mutex_unlock(&cls_lock);
    return empty?0:2;
}
static int cls_get(void **out,void *ctx,void *key) {
    int err=resolve_context(&ctx);
    if(err) return err;
    if(!out) return INVALID;
    *out=NULL;
    pthread_mutex_lock(&cls_lock);
    for(size_t i=0;i<sizeof(records)/sizeof(*records);++i){
        if(records[i].ctx==ctx && records[i].key==key) {
            *out=records[i].value;
            pthread_mutex_unlock(&cls_lock); return 0;
        }
    }
    pthread_mutex_unlock(&cls_lock); return INVALID_HANDLE;
}
static int cls_del(void *ctx,void *key) {
    int err=resolve_context(&ctx);
    if(err) return err;
    pthread_mutex_lock(&cls_lock);
    for(size_t i=0;i<sizeof(records)/sizeof(*records);++i) {
        if(records[i].ctx==ctx && records[i].key==key) {
            records[i]=(struct cls_record){0};
            pthread_mutex_unlock(&cls_lock); return 0;
        }
    }
    pthread_mutex_unlock(&cls_lock); return INVALID_HANDLE;
}

static int compatibility_digest(uint32_t version,uint64_t seconds,uint64_t *out) {
    if (!out) return INVALID;
    if (version==13000 || version==13001) {
        out[0]=(version==13000)?UINT64_C(0x3341181c03cb675c):UINT64_C(0x1841181c03cb675c);
        out[1]=UINT64_C(0x8ed383aa1f4cd1e8); return 0;
    }
    if(version!=13002) return UNSUPPORTED;
    struct hash_input {
        uint32_t driver,version,pid,thread;
        uintptr_t cudart_table,compatibility_table,function;
        uint64_t seconds;
    } header={0};
    _Static_assert(sizeof(header)==48,"Only LP64 Linux ABI is supported");
    int driver,count;
    int err=cuDriverGetVersion(&driver);
    if(err) return err;
    header.driver=(uint32_t)driver;
    header.version=version;
    header.pid=(uint32_t)getpid();
    header.thread=(uint32_t)(uintptr_t)pthread_self();
    header.cudart_table=(uintptr_t)cudart_interface;
    header.compatibility_table=(uintptr_t)integrity_interface;
    header.function=(uintptr_t)compatibility_digest;
    header.seconds=seconds;
    struct md2 ctx;
    compatibility_key(&ctx,0x36);
    md2_update(&ctx,&header,sizeof(header));
    if ((err=cuDeviceGetCount(&count))) return err;
    for(int i=0;i<count;i++) {
        int device;
        struct { uint8_t uuid[16]; int domain,bus,slot; } gpu={0};
        _Static_assert(sizeof(gpu)==28,"Unexpected device properties layout");
        if((err=cuDeviceGet(&device,i)) ||
           (err=cuDeviceGetUuid(gpu.uuid,device)) ||
           (err=cuDeviceGetAttribute(&gpu.domain,50,device)) ||
           (err=cuDeviceGetAttribute(&gpu.bus,33,device)) ||
           (err=cuDeviceGetAttribute(&gpu.slot,34,device))) return err;
        md2_update(&ctx,&gpu,sizeof(gpu));
    }
    uint8_t inner[16]; md2_final(&ctx,inner);
    compatibility_key(&ctx,0x5c);
    md2_update(&ctx,inner,sizeof(inner)); md2_final(&ctx,(uint8_t *)out);
    return 0;
}
#define FP(f) ((uintptr_t)(f))
static const uintptr_t cudart_interface[13]={
    13*sizeof(uintptr_t), FP(unsupported), FP(get_primary_context), FP(unsupported),
    FP(cuDevicePrimaryCtxSetFlags_v2), FP(unsupported),FP(unsupported),FP(unsupported),
    FP(unsupported),FP(unsupported),FP(unsupported),FP(unsupported),FP(unsupported)
};
static const uintptr_t hook_interface[7]={
    7*sizeof(uintptr_t),0,FP(hook_buffer1),0,0,0,FP(hook_buffer2)
};

static const uintptr_t tls_interface[1]={sizeof(uintptr_t)};
static const uintptr_t cls_interface[4]={FP(cls_put),FP(cls_del),FP(cls_get),FP(unsupported)};
static const uintptr_t checks_interface[4]={4*sizeof(uintptr_t),FP(unsupported),FP(check_context_alias),FP(context_check_feature)};
static const uintptr_t integrity_interface[3]={3*sizeof(uintptr_t),FP(compatibility_digest),FP(unsupported)};
static const struct {
    uint8_t id[16];
    const uintptr_t *entries;
} tables[]={
    {{0x6b,0xd5,0xfb,0x6c,0x5b,0xf4,0xe7,0x4a,0x89,0x87,0xd9,0x39,0x12,0xfd,0x9d,0xf9},cudart_interface},
    {{0xa0,0x94,0x79,0x8c,0x2e,0x74,0x2e,0x74,0x93,0xf2,0x08,0x00,0x20,0x0c,0x0a,0x66},hook_interface},
    {{0x42,0xd8,0x5a,0x81,0x23,0xf6,0xcb,0x47,0x82,0x98,0xf6,0xe7,0x8a,0x3a,0xec,0xdc},tls_interface},
    {{0xc6,0x93,0x33,0x6e,0x11,0x21,0xdf,0x11,0xa8,0xc3,0x68,0xf3,0x55,0xd8,0x95,0x93},cls_interface},
    {{0x26,0x3e,0x88,0x60,0x7c,0xd2,0x61,0x43,0x92,0xf6,0xbb,0xd5,0x00,0x6d,0xfa,0x7e},checks_interface},
    {{0xd4,0x08,0x20,0x55,0xbd,0xe6,0x70,0x4b,0x8d,0x34,0xba,0x12,0x3c,0x66,0xe1,0xf2},integrity_interface}
};
int cuGetExportTable(const void **out,const void *id) {
    if(!out || !id) return INVALID;
    *out=NULL;
    const uint8_t *uuid=id;
    int status=UNSUPPORTED;
    for(size_t i=0;i<sizeof(tables)/sizeof(*tables);i++) {
        if(!memcmp(id,tables[i].id,16)) {
            *out=tables[i].entries;
            status=0;break;
        }
    }
    fprintf(stderr,"cuGetExportTable ");
    for(int i=0;i<16;i++) fprintf(stderr,"%02x",uuid[i]);
    fprintf(stderr," -> %d\n",status);
    return status;
}
