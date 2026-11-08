#define _GNU_SOURCE
#include <ctype.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "wire.h"

enum { INVALID=1, NOMEM=2, NOT_INIT=3, INVALID_DEVICE=101, INVALID_IMAGE=200,
       INVALID_CONTEXT=201, INVALID_HANDLE=400, NOT_FOUND=500, NOT_SUPPORTED=801 };
typedef unsigned long long deviceptr;
static int initialized;
static _Thread_local void *current;
static char primary_context;
static unsigned int primary_refcount;
void *bridge_primary_handle(void) {return &primary_context;}

struct module { unsigned long long id; char *ptx; };
struct function { unsigned long long id; struct module *module; size_t nargs; };
static struct module modules[16];
static struct function functions[64];

int cuInit(unsigned flags) {
    if(flags) return INVALID;
    if(initialized) return 0;
    int status=bridge_call(NULL,"INIT");
    if(!status) initialized=1;
    return status;
}
int cuDriverGetVersion(int *version) {
    if(!version) return INVALID;
    *version=13000;
    fprintf(stderr,"DRIVER experimental_guest_abi=%d\n",*version);
    return 0;
}
int cuDeviceGetCount(int *count) {
    if(!initialized) return NOT_INIT;
    if(!count) return INVALID;
    *count=1;
    return 0;
}
int cuDeviceGet(int *device,int ordinal) {
    if(!initialized) return NOT_INIT;
    if(!device) return INVALID;
    if(ordinal) return INVALID_DEVICE;
    *device=0;
    return 0;
}
int cuDeviceGetName(char *name,int len,int dev) {
    if(!initialized) return NOT_INIT;
    if(dev) return INVALID_DEVICE;
    if(!name || len<1) return INVALID;
    char *hex=NULL; int status=bridge_call(&hex,"NAME");
    if(status) return status;
    size_t n=strlen(hex)/2; char *bytes=malloc(n+1);
    if(!bytes) {free(hex);return NOMEM;}
    status=hex_decode(hex,bytes,n);
    if(!status) {
        size_t copy=n<(size_t)len-1?n:(size_t)len-1;
        memcpy(name,bytes,copy); name[copy]=0;
    }
    free(bytes); free(hex); return status;
}
int cuDeviceTotalMem_v2(size_t *bytes,int dev) {
    if(!initialized) return NOT_INIT;
    if(dev) return INVALID_DEVICE;
    if(!bytes) return INVALID;
    unsigned long long total=0;
    int status=bridge_number(&total,"TOTAL");
    if(!status) *bytes=(size_t)total;
    return status;
}

int cuDeviceGetAttribute(int *out,int attr,int dev) {
    if(!initialized) return NOT_INIT;
    if(!out) return INVALID;
    if(dev) return INVALID_DEVICE;

    int value_override=-3;
    if(attr==3 || attr==4 || attr==6 || attr==7) value_override=1;
    if(attr==18) value_override=1;
    if(attr==131 || attr==134) value_override=-1;
    if((attr>=21 && attr<=32) || (attr>=42 && attr<=49) || (attr>=51 && attr<=74) ||
       attr==9 || attr==11 || attr==14 || attr==19 || attr==35 || attr==37 ||
       attr==38 || attr==40 || attr==41 || attr==77 || attr==78 || attr==79 ||
       attr==80 || attr==83 || attr==84 || attr==85 || attr==86 || attr==88 ||
       attr==89 || attr==90 || attr==91 || attr==95 || attr==99 || attr==100 ||
       attr==101 || attr==108 || attr==109 || attr==111 || (attr>=112 && attr<=121) ||
       attr==125 || attr==129 || attr==130 || attr==133 || attr==139 || attr==140 || attr==143)
       value_override=0;
    if(value_override!=-3) {
        *out=value_override; fprintf(stderr,"GUEST ATTR %d -> %d virtual capability\n",attr,*out);
        return 0;
    }
    unsigned long long value=0; int status=bridge_number(&value,"ATTR %d",attr);
    if(!status && value<=2147483647U) *out=(int)value;
    else if(!status) return NOT_SUPPORTED;
    return status;
}
int cuDeviceGetUuid(void *out,int dev) {
    if(!initialized) return NOT_INIT;
    if(!out) return INVALID;
    if(dev) return INVALID_DEVICE;
    char *hex=NULL; int status=bridge_call(&hex,"UUID");
    if(!status) status=hex_decode(hex,out,16);
    free(hex); return status;
}
int cuDeviceGetUuid_v2(void *out,int dev) {return cuDeviceGetUuid(out,dev);}
int cuDevicePrimaryCtxSetFlags_v2(int device,unsigned flags) {
    if(!initialized) return NOT_INIT;
    if(device) return INVALID_DEVICE;
    if(flags) return NOT_SUPPORTED;
    if(primary_refcount) return 708;
    return 0;
}
int cuDevicePrimaryCtxGetState(int device,unsigned *flags,int *active) {
    if(!initialized) return NOT_INIT;
    if(device) return INVALID_DEVICE;
    if(!flags || !active) return INVALID;
    *flags=0; *active=primary_refcount>0; return 0;
}
int cuDevicePrimaryCtxRetain(void **ctx,int device) {
    if(!initialized) return NOT_INIT;
    if(!ctx) return INVALID;
    if(device) return INVALID_DEVICE;
    primary_refcount++;
    *ctx=&primary_context;
    return 0;
}
int cuDevicePrimaryCtxRelease_v2(int device) {
    if(!initialized) return NOT_INIT;
    if(device) return INVALID_DEVICE;
    if(!primary_refcount) return INVALID_CONTEXT;
    primary_refcount--;
    if(!primary_refcount) current=NULL;
    return 0;
}
int cuDevicePrimaryCtxRelease(int device) {return cuDevicePrimaryCtxRelease_v2(device);}
int cuCtxSetCurrent(void *ctx) {
    if(!initialized) return NOT_INIT;
    if(ctx && (ctx!=&primary_context || !primary_refcount)) return INVALID_CONTEXT;
    current=ctx; return 0;
}
int cuCtxGetCurrent(void **ctx) {
    if(!initialized) return NOT_INIT;
    if(!ctx) return INVALID;
    *ctx=current; return 0;
}
int cuCtxGetDevice(int *device) {
    if(!current) return INVALID_CONTEXT;
    if(!device) return INVALID;
    *device=0;return 0;
}
int cuCtxSynchronize(void) {
    if(!current) return INVALID_CONTEXT;
    return bridge_call(NULL,"SYNC");
}
static int valid_execution_context(void *ctx) {
    if(!initialized) return NOT_INIT;
    if(!ctx) ctx=current;
    return ctx==&primary_context && primary_refcount ? 0 : INVALID_CONTEXT;
}
int cuCtxGetDevice_v2(int *device,void *ctx) {
    if(!device) return INVALID;
    int status=valid_execution_context(ctx);
    if(status) return status;
    *device=0; return 0;
}
int cuCtxSynchronize_v2(void *ctx) {
    int status=valid_execution_context(ctx);
    return status ? status : bridge_call(NULL,"SYNC");
}
int cuModuleGetLoadingMode(int *mode) {
    if(!initialized) return NOT_INIT;
    if(!mode) return INVALID;
    *mode=2;
    return 0;
}
int cuMemAlloc_v2(deviceptr *ptr,size_t n) {
    if(!current) return INVALID_CONTEXT;
    if(!ptr || !n) return INVALID;
    return bridge_number(ptr,"ALLOC %zu",n);
}
int cuMemFree_v2(deviceptr ptr) {
    if(!current) return INVALID_CONTEXT;
    return bridge_call(NULL,"FREE %llu",ptr);
}
int cuMemcpyHtoD_v2(deviceptr dest,const void *src,size_t n) {
    if(!current) return INVALID_CONTEXT;
    if(!n) return 0;
    if(!dest || !src) return INVALID;
    char *hex=hex_encode(src,n);
    if(!hex) return NOT_SUPPORTED;
    int status=bridge_call(NULL,"H2D %llu %s",dest,hex);
    free(hex); return status;
}
int cuMemcpyDtoH_v2(void *dest,deviceptr src,size_t n) {
    if(!current) return INVALID_CONTEXT;
    if(!n) return 0;
    if(!dest || !src || n>((16<<20)-128)/2) return INVALID;
    char *reply=NULL;
    int status=bridge_call(&reply,"D2H %llu %zu",src,n);
    if(!status) status=hex_decode(reply,dest,n);
    free(reply); return status;
}

static struct module *find_module(void *handle) {
    for(size_t i=0;i<16;i++) if(handle==modules+i && modules[i].id) return modules+i;
    return NULL;
}
int cuModuleLoadData(void **handle,const void *image) {
    if(!current) return INVALID_CONTEXT;
    if(!handle || !image) return INVALID;
    const unsigned char *first=image;
    while(*first && isspace(*first)) first++;

    if(*first!='.' || strncmp((const char*)first,".version",8)) return INVALID_IMAGE;
    size_t n=strnlen((const char*)image,1<<20);
    if(!n || n==(1<<20)) return INVALID_IMAGE;
    struct module *m=NULL;
    for(size_t i=0;i<16;i++) if(!modules[i].id){m=modules+i;break;}
    if(!m) return NOMEM;
    char *ptx=strndup(image,n), *hex=hex_encode(image,n);
    if(!ptx || !hex) {free(ptx);free(hex);return NOMEM;}
    unsigned long long id=0;
    int status=bridge_number(&id,"LOAD %s",hex);
    free(hex);
    if(status) {free(ptx);return status;}
    m->id=id; m->ptx=ptx; *handle=m; return 0;
}

static const char *skip_space(const char *p) {
    while(*p && isspace((unsigned char)*p)) p++;
    return p;
}
static int identifier(char c) {return isalnum((unsigned char)c) || c=='_' || c=='$';}
static int consume(const char **p,const char *token) {
    const char *s=skip_space(*p); size_t n=strlen(token);
    if(strncmp(s,token,n) || identifier(s[n])) return 0;
    *p=s+n; return 1;
}

static int arity(const char *ptx,const char *name,size_t *count) {
    const char *p=ptx; size_t n=strlen(name);
    if(!n || n>200) return INVALID;
    for(size_t i=0;i<n;i++) if(!identifier(name[i])) return INVALID;
    while((p=strstr(p,".entry"))!=NULL) {
        p=skip_space(p+6);
        if(strncmp(p,name,n) || identifier(p[n])) continue;
        p=skip_space(p+n);if(*p++!='(') return INVALID_IMAGE;
        *count=0;
        for (;;) {
            p=skip_space(p);
            if(*p==')') return 0;
            if(*count==16 || !consume(&p,".param") || !consume(&p,".u64") || !consume(&p,".ptr")) return NOT_SUPPORTED;
            (void)consume(&p,".global"); p=skip_space(p);
            if(!identifier(*p)) return INVALID_IMAGE;
            while(identifier(*p))p++;
            (*count)++; p=skip_space(p);
            if(*p==')') return 0;
            if(*p++!=',') return INVALID_IMAGE;
        }
    }
    return NOT_FOUND;
}
int cuModuleGetFunction(void **handle,void *mod_handle,const char *name) {
    if(!current) return INVALID_CONTEXT;
    struct module *m=find_module(mod_handle);
    if(!m) return INVALID_HANDLE;
    if(!handle || !name) return INVALID;
    size_t count; int status=arity(m->ptx,name,&count);
    if(status) return status;
    struct function *f=NULL;
    for(size_t i=0;i<64;i++) if(!functions[i].id){f=functions+i;break;}
    if(!f) return NOMEM;
    unsigned long long id=0;
    status=bridge_number(&id,"GET %llu %s",m->id,name);
    if(status) return status;
    f->id=id; f->module=m; f->nargs=count; *handle=f; return 0;
}
int cuLaunchKernel(void *handle,unsigned gx,unsigned gy,unsigned gz,unsigned bx,unsigned by,unsigned bz,
                   unsigned shared,void *stream,void **params,void **extra) {
    if(!current) return INVALID_CONTEXT;
    struct function *f=NULL;
    for(size_t i=0;i<64;i++) if(handle==functions+i && functions[i].id){f=functions+i;break;}
    if(!f) return INVALID_HANDLE;
    if(!gx || !bx || (!params && f->nargs)) return INVALID;
    if(gy!=1 || gz!=1 || by!=1 || bz!=1 || shared || stream || extra) return NOT_SUPPORTED;
    char args[16*28+1]=""; size_t used=0;
    for(size_t i=0;i<f->nargs;i++) {
        if(!params[i]) return INVALID;
        deviceptr ptr;
        memcpy(&ptr,params[i],sizeof(ptr));
        used+=(size_t)snprintf(args+used,sizeof(args)-used," p:%llu",ptr);
    }
    return bridge_call(NULL,"LAUNCH %llu %u %u %zu%s",f->id,gx,bx,f->nargs,args);
}
int cuModuleUnload(void *handle) {
    struct module *m=find_module(handle);
    if(!m) return INVALID_HANDLE;
    int status=bridge_call(NULL,"UNLOAD %llu",m->id); if(status) return status;
    for(size_t i=0;i<64;i++) if(functions[i].module==m) memset(functions+i,0,sizeof(functions[i]));
    free(m->ptx); memset(m,0,sizeof(*m)); return 0;
}

int cuGetExportTable(const void **table,const unsigned char *uuid);
int cuGetProcAddress(const char *sym,void **out,int version,unsigned long long flags);
int cuGetProcAddress_v2(const char *sym,void **out,int version,unsigned long long flags,int *query);
struct entry {const char *name;void *fn;int since;};
#define ENTRY(name,version) {#name,(void *)(name),version}
#define VERSIONED(name,version) {#name,(void *)(name##_v2),version}
static const struct entry entries[]={
    ENTRY(cuInit,2000),ENTRY(cuDriverGetVersion,2020),ENTRY(cuDeviceGetCount,2000),ENTRY(cuDeviceGet,2000),
    ENTRY(cuDeviceGetName,2000), VERSIONED(cuDeviceTotalMem,3020),
    ENTRY(cuDeviceGetAttribute,2000),ENTRY(cuDeviceGetUuid,9020),
    ENTRY(cuDevicePrimaryCtxGetState,7000),VERSIONED(cuDevicePrimaryCtxSetFlags,11000),
    ENTRY(cuDevicePrimaryCtxRetain,7000),VERSIONED(cuDevicePrimaryCtxRelease,11000),
    ENTRY(cuCtxSetCurrent,4000),ENTRY(cuCtxGetCurrent,4000),ENTRY(cuCtxGetDevice,2000),ENTRY(cuCtxSynchronize,2000),
    VERSIONED(cuMemAlloc,3020),VERSIONED(cuMemFree,3020),VERSIONED(cuMemcpyHtoD,3020),VERSIONED(cuMemcpyDtoH,3020),
    ENTRY(cuModuleGetLoadingMode,11070),ENTRY(cuModuleLoadData,2000),ENTRY(cuModuleGetFunction,2000),ENTRY(cuModuleUnload,2000),ENTRY(cuLaunchKernel,4000),
    ENTRY(cuGetExportTable,3000)
};
int cuGetProcAddress_v2(const char *sym,void **out,int version,unsigned long long flags,int *query) {
    if(!sym || !out || version<0 || version>13000 || flags>2) return INVALID;
    *out=NULL; int result=1;
    if(!strcmp(sym,"cuDeviceGetUuid") && version>=11040) {
        *out=(void*)cuDeviceGetUuid_v2;
    } else if(!strcmp(sym,"cuGetProcAddress")) {
        if(version>=12000) *out=(void*)cuGetProcAddress_v2;
        else if(version>=11030) *out=(void*)cuGetProcAddress;
        else result=2;
    } else if(!strcmp(sym,"cuCtxSynchronize") && version>=13000) {
        *out=(void*)cuCtxSynchronize_v2;
    } else if(!strcmp(sym,"cuCtxGetDevice") && version>=13000) {
        *out=(void*)cuCtxGetDevice_v2;
    } else if(!strcmp(sym,"cuDevicePrimaryCtxRelease") && version>=7000 && version<11000) {
        *out=(void*)cuDevicePrimaryCtxRelease;
    } else for(size_t i=0;i<sizeof(entries)/sizeof(entries[0]);i++) {
        if(strcmp(sym,entries[i].name)) continue;
        if(version>=entries[i].since) *out=entries[i].fn; else result=2;
        break;
    }
    if(*out) result=0;
    if(query) *query=result;
    fprintf(stderr,"LOOKUP %s version=%d flags=%llu -> %s\n",sym,version,flags,*out?"found":"missing");
    return 0;
}
int cuGetProcAddress(const char *sym,void **out,int version,unsigned long long flags) {
    return cuGetProcAddress_v2(sym,out,version,flags,NULL);
}
