#include <stdlib.h>
#include <stdio.h>
#include "wire.h"
int cuGetExportTable(const void **table,const unsigned char *uuid) {
    if(!table || !uuid) return 1;
    *table=NULL; char *hex=hex_encode(uuid,16);
    fprintf(stderr,"DRIVER cuGetExportTable uuid=%s -> NOT_SUPPORTED\n",hex?hex:"?");
    free(hex); return 801;
}
