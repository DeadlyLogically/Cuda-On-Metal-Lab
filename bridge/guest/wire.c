#define _GNU_SOURCE
#include <errno.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include "wire.h"

static pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
static int fd = -1;
static FILE *input;
static int dead;
enum { MAX_LINE = 16 << 20 };

static int connect_host(void) {
    const char *path = getenv("CUMETAL_BRIDGE_SOCKET");
    if (!path || !*path) { errno=ENOENT; return -1; }
    struct sockaddr_un address = {.sun_family = AF_UNIX};
    if (strlen(path) >= sizeof(address.sun_path)) { errno=ENAMETOOLONG; return -1; }
    strcpy(address.sun_path, path);
    int result=socket(AF_UNIX,SOCK_STREAM,0);
    if(result>=0 && connect(result,(struct sockaddr *)&address,sizeof(address))<0) {
        close(result); return -1;
    }
    return result;
}

static int command(char **reply, const char *request) {
    if (reply) *reply = NULL;
    if (!request || strchr(request,'\n') || strlen(request)>=MAX_LINE) return 1;
    pthread_mutex_lock(&lock);
    int status = 999;
    if (dead) goto done;
    if (fd<0) {
        fd=connect_host();
        if (fd<0) { perror("bridge connect"); goto done; }
        struct timeval timeout = {.tv_sec=25};
        setsockopt(fd,SOL_SOCKET,SO_RCVTIMEO,&timeout,sizeof(timeout));
        setsockopt(fd,SOL_SOCKET,SO_SNDTIMEO,&timeout,sizeof(timeout));
        input=fdopen(fd,"r");
        if (!input) { close(fd); fd=-1; goto done; }
    }
    const char *part=request;
    size_t left=strlen(request);
    while (left) {
        ssize_t sent=send(fd,part,left,MSG_NOSIGNAL);
        if (sent<0 && errno==EINTR) continue;
        if (sent<=0) goto failed;
        part+=sent; left-=(size_t)sent;
    }
    if (send(fd,"\n",1,MSG_NOSIGNAL)!=1) goto failed;
    size_t max = 256;
    if (!strncmp(request,"D2H ",4)) {
        const char *last=strrchr(request,' ');
        unsigned long long count = last ? strtoull(last+1,NULL,10) : 0;
        if (count>((MAX_LINE-32)/2)) goto failed;
        max = (size_t)count*2+32;
    }
    char *line=malloc(max);
    if (!line) goto failed;
    if (!fgets(line,(int)max,input) || !strchr(line,'\n')) { free(line); goto failed; }
    line[strcspn(line,"\r\n")]=0;
    if (!strcmp(line,"OK") || !strncmp(line,"OK ",3)) {
        const char *value=line[2]==' '?line+3:"";
        status = reply ? ((*reply=strdup(value))?0:2) : (*value?999:0);
    } else if (!strncmp(line,"ERR ",4)) {
        char *end=NULL;
        long result=strtol(line+4,&end,10);
        status= end && *end=='\0' && end!=line+4 && result>0 && result<=999 ? (int)result : 999;
    }
    fprintf(stderr,"BRIDGE %.*s status=%d\n",(int)strcspn(request," "),request,status);
    free(line); goto done;
failed:
    dead=1; fclose(input); input=NULL; fd=-1;
    fprintf(stderr,"BRIDGE transport/protocol failure\n");
done:
    pthread_mutex_unlock(&lock);
    return status;
}

int bridge_call(char **reply,const char *format,...) {
    char *request=NULL; va_list args; va_start(args,format);
    int size=vasprintf(&request,format,args); va_end(args);
    if (size<0) return 2;
    int status=command(reply,request); free(request); return status;
}

int bridge_number(unsigned long long *number,const char *format,...) {
    if (!number) return 1;
    char *request=NULL, *reply=NULL; va_list args; va_start(args,format);
    int size=vasprintf(&request,format,args); va_end(args);
    if (size<0) return 2;
    int status=command(&reply,request); free(request);
    if (status==0) {
        errno=0; char *end=NULL;
        unsigned long long value=strtoull(reply,&end,10);
        if (errno || !*reply || *end || *reply=='-') status=999; else *number=value;
    }
    free(reply); return status;
}

char *hex_encode(const void *data,size_t count) {
    if ((!data && count) || count>(MAX_LINE-128)/2) return NULL;
    char *out=malloc(count*2+1);
    if (!out) return NULL;
    const unsigned char *p=data;
    const char digits[]="0123456789abcdef";
    for (size_t i=0;i<count;i++) {out[2*i]=digits[p[i]>>4]; out[2*i+1]=digits[p[i]&15];}
    out[count*2]=0; return out;
}

static int nibble(unsigned char c) {
    if(c>='0' && c<='9') return c-'0';
    if(c>='a' && c<='f') return c-'a'+10;
    if(c>='A' && c<='F') return c-'A'+10;
    return -1;
}
int hex_decode(const char *hex,void *bytes,size_t count) {
    if(!hex || strlen(hex)!=count*2) return 999;
    unsigned char *p=bytes;
    for(size_t i=0;i<count;i++) {
        int hi=nibble(hex[2*i]),lo=nibble(hex[2*i+1]);
        if(hi<0 || lo<0) return 999;
        p[i]=(unsigned char)(hi*16+lo);
    }
    return 0;
}
