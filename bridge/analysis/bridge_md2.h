#include <stdint.h>
#include <string.h>
static const uint8_t permutation[256]={
    41, 46, 67, 201, 162, 216, 124, 1, 61, 54, 84, 161, 236, 240, 6, 19,
    98, 167, 5, 243, 192, 199, 115, 140, 152, 147, 43, 217, 188, 76, 130, 202,
    30, 155, 87, 60, 253, 212, 224, 22, 103, 66, 111, 24, 138, 23, 229, 18,
    190, 78, 196, 214, 218, 158, 222, 73, 160, 251, 245, 142, 187, 47, 238, 122,
    169, 104, 121, 145, 21, 178, 7, 63, 148, 194, 16, 137, 11, 34, 95, 33,
    128, 127, 93, 154, 90, 144, 50, 39, 53, 62, 204, 231, 191, 247, 151, 3,
    255, 25, 48, 179, 72, 165, 181, 209, 215, 94, 146, 42, 172, 86, 170, 198,
    79, 184, 56, 210, 150, 164, 125, 182, 118, 252, 107, 226, 156, 116, 4, 241,
    69, 157, 112, 89, 100, 113, 135, 32, 134, 91, 207, 101, 230, 45, 168, 2,
    27, 96, 37, 173, 174, 176, 185, 246, 28, 70, 97, 105, 52, 64, 126, 15,
    85, 71, 163, 35, 221, 81, 175, 58, 195, 92, 249, 206, 186, 197, 234, 38,
    44, 83, 13, 110, 133, 40, 132, 9, 211, 223, 205, 244, 65, 129, 77, 82,
    106, 220, 55, 200, 108, 193, 171, 250, 36, 225, 123, 8, 12, 189, 177, 74,
    120, 136, 149, 139, 227, 99, 232, 109, 233, 203, 213, 254, 59, 0, 29, 57,
    242, 239, 183, 14, 102, 88, 208, 228, 166, 119, 114, 248, 235, 117, 75, 10,
    49, 68, 80, 180, 143, 237, 31, 26, 219, 153, 141, 51, 159, 17, 131, 20,
};


struct md2 {
    uint8_t state[48], checksum[16], pending[16];
    unsigned used;
};

static void md2_block(struct md2 *ctx, const uint8_t block[16]) {
    uint8_t last = ctx->checksum[15], t = 0;
    for (int i = 0; i < 16; ++i) {
        ctx->state[16+i] = block[i];
        ctx->state[32+i] = block[i] ^ ctx->state[i];
        ctx->checksum[i] ^= permutation[block[i] ^ last];
        last = ctx->checksum[i];
    }
    for (int pass = 0; pass < 18; ++pass) {
        for (int i = 0; i < 48; ++i) {
            ctx->state[i] ^= permutation[t];
            t = ctx->state[i];
        }
        t = (uint8_t)(t + pass);
    }
}

static void md2_update(struct md2 *ctx, const void *buf, size_t n) {
    const uint8_t *p = buf;
    while (n) {
        size_t take = 16 - ctx->used;
        if (take > n) take = n;
        memcpy(ctx->pending + ctx->used, p, take);
        ctx->used += (unsigned)take;
        n -= take;
        p += take;
        if (ctx->used == 16) {
            md2_block(ctx, ctx->pending);
            ctx->used = 0;
        }
    }
}

static void md2_final(struct md2 *ctx, uint8_t out[16]) {
    uint8_t pad[16], check[16];
    uint8_t n = (uint8_t)(16-ctx->used);
    memset(pad,n,n);
    md2_update(ctx,pad,n);
    memcpy(check,ctx->checksum,16);
    md2_update(ctx,check,16);
    memcpy(out,ctx->state,16);
}

static void compatibility_key(struct md2 *ctx, uint8_t domain_byte) {
    static const uint8_t key[16]={0x14,0x6a,0xdd,0xae,0x53,0xa9,0xa7,0x52,0xaa,0x08,0x41,0x36,0x0b,0xf5,0x5a,0x9f};
    uint8_t pad[16];
    memset(ctx,0,sizeof(*ctx));
    for (size_t i=0;i<sizeof(pad);i++) pad[i]=key[i]^domain_byte;
    md2_update(ctx,pad,sizeof(pad));
}
