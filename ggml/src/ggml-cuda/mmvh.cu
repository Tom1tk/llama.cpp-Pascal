#include "mmvh.cuh"
#include "vecdotq.cuh"

// How it works (D2 in docs/pascal/fork-plan-2026-10-14.md):
//
// GP100 has no dp4a, so mmvq's int8 dot products are emulated (BFE + XMAD per
// byte) and the matvec is ALU-bound at ~27% of HBM bandwidth. Here the weights
// never become ints. A byte b placed in the low bits of a half with a zero
// exponent *is* the half b * 2^-24 (a subnormal), so `b & 0x00FF00FF` is two
// exact fp16 weights in one LOP. They go straight into __hfma2.
//
// The activations are converted once per call to half, scaled per 32-block to
// |x| <= 2^14 so the products stay in normal range, and permuted [0,2,1,3]
// within every group of 4 so they line up with the (byte0,byte2)/(byte1,byte3)
// lanes. Per 32-block we also keep the scale and the sums of each 16-half, which
// pay for the K-quant minimums and IQ4_XS's +128 table bias.
//
// Work split: one warp per R rows (sharing the activation loads), 8 lanes per
// 256-weight superblock, each lane two units of 16 weights (one 16-half of a
// 32-block each).

#define MMVH_NWARPS 4

// H41: Q6_K with 4 lanes per superblock (64 weights each) instead of 8, opt-in.
static bool mmvh_q6k_wide_enabled() {
    static const bool enabled = [] {
        const char * env = getenv("GGML_PASCAL_MMVH_Q6K_WIDE");
        return env != nullptr && atoi(env) != 0;
    }();
    return enabled;
}

// Q3_K, Q2_K and the IQ2 family on mmvh (otherwise mmvq), opt-in.
static bool mmvh_lowbit_enabled() {
    static const bool enabled = [] {
        const char * env = getenv("GGML_PASCAL_MMVH_LOWBIT");
        return env != nullptr && atoi(env) != 0;
    }();
    return enabled;
}

static bool mmvh_env_enabled() {
    static const bool enabled = [] {
        const char * env = getenv("GGML_CUDA_PASCAL_MMVH");
        return env != nullptr && atoi(env) != 0;
    }();
    return enabled;
}

bool ggml_cuda_mmvh_can_use(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst) {
    if (!mmvh_env_enabled()) {
        return false;
    }
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    if (!GGML_CUDA_CC_IS_NVIDIA(cc) || cc < GGML_CUDA_CC_PASCAL || cc >= GGML_CUDA_CC_VOLTA) {
        return false;
    }
    switch (src0->type) {
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_IQ3_S:
        case GGML_TYPE_IQ3_XXS:
            break;
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ2_XXS:
            if (!mmvh_lowbit_enabled()) {
                return false;
            }
            break;
        default:
            return false;
    }
    return src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 &&
        src1->ne[1] >= 1 && src1->ne[1] <= 8 &&
        src0->ne[2] == 1 && src0->ne[3] == 1 && src1->ne[2] == 1 && src1->ne[3] == 1 &&
        src1->nb[0] == sizeof(float) && src0->ne[0] % QK_K == 0;
}

// --- activations ------------------------------------------------------------

// one warp per 32-block: xh = x * 2^14/amax (permuted), meta = {amax*2^10, sum lo16, sum hi16, 0}
static __global__ void mmvh_convert_y(const float * __restrict__ x, half * __restrict__ xh, float4 * __restrict__ meta,
        const int64_t s11, const int64_t ncols_padded) {
    const int     lane = threadIdx.x;
    const int64_t col  = blockIdx.y;
    const int64_t i    = 32*int64_t(blockIdx.x) + lane;

    const float v = x[col*s11 + i];

    float amax = fabsf(v);
    float sum  = v;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, off, 32));
    }
#pragma unroll
    for (int off = 8; off > 0; off >>= 1) {
        sum += __shfl_xor_sync(0xFFFFFFFF, sum, off, 32);
    }
    const float hi_sum = __shfl_sync(0xFFFFFFFF, sum, 16, 32);

    const float scale = amax > 0.0f ? 16384.0f / amax : 0.0f;
    const int   dst_i = (lane & ~3) | ((lane & 1) << 1) | ((lane >> 1) & 1);
    xh[col*ncols_padded + 32*int64_t(blockIdx.x) + dst_i] = __float2half_rn(v * scale);

    if (lane == 0) {
        meta[col*(ncols_padded/32) + blockIdx.x] = make_float4(amax * 1024.0f, sum, hi_sum, 0.0f);
    }
}

// --- weights -----------------------------------------------------------------

// Two units of 16 weights each, as bytes (unsigned, < 256) in 4 ints per unit.
struct mmvh_units {
    int   b[2][4];
    int   blk[2];   // 32-block within the superblock
    int   hf[2];    // which 16-half of that 32-block
    float ws[2];    // multiplies sum(b*x)
    float mt[2];    // multiplies sum(x) over the unit, subtracted
};

template <ggml_type type>
static __device__ __forceinline__ void mmvh_load(const char * __restrict__ row, const int sb, const int sub, mmvh_units & u);

// Q4_K / Q5_K 6-bit scale and min for 32-block is, branch-free (is < 4 vs >= 4 diverges across lanes).
// scales[] sits at offset 4 of both blocks, so it is 4-byte aligned.
static __device__ __forceinline__ void mmvh_scale_min_k4(const uint8_t * scales, const int is, int & sc, int & m) {
    const int * s4 = (const int *) scales;
    const int sh = 8*(is & 3);
    const int b0 = (s4[0] >> sh) & 0xFF;
    const int b1 = (s4[1] >> sh) & 0xFF;
    const int b2 = (s4[2] >> sh) & 0xFF;
    sc = is < 4 ? b0 & 63 : (b2 & 0xF) | ((b0 >> 6) << 4);
    m  = is < 4 ? b1 & 63 : (b2 >>  4) | ((b1 >> 6) << 4);
}

template <>
__device__ __forceinline__ void mmvh_load<GGML_TYPE_Q4_K>(const char * __restrict__ row, const int sb, const int sub, mmvh_units & u) {
    const block_q4_K * bq = (const block_q4_K *) row + sb;
    const int j  = sub >> 1;
    const int l0 = 16*(sub & 1);
    const int4 q = *(const int4 *) (bq->qs + 32*j + l0);
    const int qa[4] = {q.x, q.y, q.z, q.w};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        u.b[0][i] =  qa[i]       & 0x0F0F0F0F;
        u.b[1][i] = (qa[i] >> 4) & 0x0F0F0F0F;
    }
    const float2 dm = __half22float2(bq->dm);
#pragma unroll
    for (int k = 0; k < 2; ++k) {
        const int is = 2*j + k;
        int sc, m;
        mmvh_scale_min_k4(bq->scales, is, sc, m);
        u.blk[k] = is;
        u.hf[k]  = sub & 1;
        u.ws[k]  = dm.x * sc;
        u.mt[k]  = dm.y * m;
    }
}

template <>
__device__ __forceinline__ void mmvh_load<GGML_TYPE_Q5_K>(const char * __restrict__ row, const int sb, const int sub, mmvh_units & u) {
    const block_q5_K * bq = (const block_q5_K *) row + sb;
    const int j  = sub >> 1;
    const int l0 = 16*(sub & 1);
    const int4 q = *(const int4 *) (bq->qs + 32*j + l0);
    const int4 h = *(const int4 *) (bq->qh + l0);
    const int qa[4] = {q.x, q.y, q.z, q.w};
    const int ha[4] = {h.x, h.y, h.z, h.w};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        u.b[0][i] = ( qa[i]       & 0x0F0F0F0F) | (((ha[i] >> (2*j + 0)) & 0x01010101) << 4);
        u.b[1][i] = ((qa[i] >> 4) & 0x0F0F0F0F) | (((ha[i] >> (2*j + 1)) & 0x01010101) << 4);
    }
    const float2 dm = __half22float2(bq->dm);
#pragma unroll
    for (int k = 0; k < 2; ++k) {
        const int is = 2*j + k;
        int sc, m;
        mmvh_scale_min_k4(bq->scales, is, sc, m);
        u.blk[k] = is;
        u.hf[k]  = sub & 1;
        u.ws[k]  = dm.x * sc;
        u.mt[k]  = dm.y * m;
    }
}

template <>
__device__ __forceinline__ void mmvh_load<GGML_TYPE_Q6_K>(const char * __restrict__ row, const int sb, const int sub, mmvh_units & u) {
    const block_q6_K * bq = (const block_q6_K *) row + sb;
    // 32-block sub = 128-half n, quadrant qd: low/high nibble of ql[l] / ql[l+32], qh bits 2*qd
    const int n  = sub >> 2;
    const int qd = sub & 3;
    const float d = __half2float(bq->d);
#pragma unroll
    for (int k = 0; k < 2; ++k) {
        // block_q6_K is 210 bytes: only 2-byte alignment
        const int ql0 = (64*n + 32*(qd & 1) + 16*k) / 4;
        const int qh0 = (32*n + 16*k) / 4;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int ql = get_int_b2(bq->ql, ql0 + i) >> (4*(qd >> 1));
            const int qh = get_int_b2(bq->qh, qh0 + i) >> (2*qd);
            u.b[k][i] = (ql & 0x0F0F0F0F) | ((qh & 0x03030303) << 4);
        }
        u.blk[k] = sub;
        u.hf[k]  = k;
        u.ws[k]  = d * bq->scales[8*n + 2*qd + k];
        u.mt[k]  = 32.0f * u.ws[k];
    }
}

// 4 units of 16 weights per lane (H41). Q6_K: the lane owns ql[l0..l0+15], ql[l0+32..l0+47] and
// qh[l0..l0+15] of one 128-half, so no byte is fetched by two lanes (the 8-lane split reads
// every qh byte 4x and every ql byte 2x, with 2-byte loads).
struct mmvh_units4 {
    int   b[4][4];
    int   blk[4];
    int   hf[4];
    float ws[4];
    float mt[4];
};

// 16 bytes at a 2-byte aligned address: 5 aligned 4-byte loads + funnel shifts, branch-free.
// Reads up to 4 bytes past the end; every caller stays inside its block_q6_K.
static __device__ __forceinline__ void mmvh_load16_b2(const uint8_t * p, int * out) {
    const uintptr_t a  = (uintptr_t) p;
    const int     * q  = (const int *) (a & ~uintptr_t(3));
    const int       sh = (a & 2) * 8;
    int w[5];
#pragma unroll
    for (int i = 0; i < 5; ++i) {
        w[i] = q[i];
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        out[i] = __funnelshift_r(w[i], w[i + 1], sh);
    }
}

template <ggml_type type>
static __device__ __forceinline__ void mmvh_load4(const char * __restrict__ row, const int sb, const int sub, mmvh_units4 & u);

template <>
__device__ __forceinline__ void mmvh_load4<GGML_TYPE_Q6_K>(const char * __restrict__ row, const int sb, const int sub, mmvh_units4 & u) {
    const block_q6_K * bq = (const block_q6_K *) row + sb;
    const int n  = sub >> 1;
    const int hf = sub & 1;
    const int l0 = 16*hf;
    int qa[4], qb[4], qh[4];
    mmvh_load16_b2(bq->ql + 64*n + l0,      qa);
    mmvh_load16_b2(bq->ql + 64*n + 32 + l0, qb);
    mmvh_load16_b2(bq->qh + 32*n + l0,      qh);
    const float d = __half2float(bq->d);
#pragma unroll
    for (int qd = 0; qd < 4; ++qd) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int ql = (qd & 1 ? qb[i] : qa[i]) >> (4*(qd >> 1));
            u.b[qd][i] = (ql & 0x0F0F0F0F) | (((qh[i] >> (2*qd)) & 0x03030303) << 4);
        }
        u.blk[qd] = 4*n + qd;
        u.hf[qd]  = hf;
        u.ws[qd]  = d * bq->scales[8*n + 2*qd + hf];
        u.mt[qd]  = 32.0f * u.ws[qd];
    }
}

// Q3_K / Q2_K with the same 4-lane split: lane (n, h) owns qs[32n+16h..+15]; unit j takes bits 2j..2j+1.
template <>
__device__ __forceinline__ void mmvh_load4<GGML_TYPE_Q3_K>(const char * __restrict__ row, const int sb, const int sub, mmvh_units4 & u) {
    const block_q3_K * bq = (const block_q3_K *) row + sb;
    const int n = sub >> 1;
    const int h = sub & 1;
    int qs[4], hm[4];
    mmvh_load16_b2(bq->qs + 32*n + 16*h, qs);
    mmvh_load16_b2(bq->hmask + 16*h,     hm);
    const float d = __half2float(bq->d);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            u.b[j][i] = ((qs[i] >> (2*j)) & 0x03030303) | (((hm[i] >> (4*n + j)) & 0x01010101) << 2); // q + 4, in 0..7
        }
        const int is = 8*n + 2*j + h;
        const int us = (is < 8 ? bq->scales[is] & 0xF : bq->scales[is - 8] >> 4) | (((bq->scales[8 + is%4] >> (2*(is/4))) & 3) << 4);
        u.blk[j] = 4*n + j;
        u.hf[j]  = h;
        u.ws[j]  = d * (us - 32);
        u.mt[j]  = 4.0f * u.ws[j];
    }
}

template <>
__device__ __forceinline__ void mmvh_load4<GGML_TYPE_Q2_K>(const char * __restrict__ row, const int sb, const int sub, mmvh_units4 & u) {
    const block_q2_K * bq = (const block_q2_K *) row + sb;
    const int n = sub >> 1;
    const int h = sub & 1;
    int qs[4];
    mmvh_load16_b2(bq->qs + 32*n + 16*h, qs);
    const float2 dm = __half22float2(bq->dm);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            u.b[j][i] = (qs[i] >> (2*j)) & 0x03030303;
        }
        const int sc = bq->scales[8*n + 2*j + h];
        u.blk[j] = 4*n + j;
        u.hf[j]  = h;
        u.ws[j]  = dm.x * (sc & 0xF);
        u.mt[j]  = dm.y * (sc >> 4);
    }
}

// kvalues_iq4nl + 128, as bytes: every table value becomes a non-negative byte
static __device__ __forceinline__ int2 mmvh_iq4nl_lookup(const int q4) {
    const uint32_t t0 = 0x3f2d1801, t1 = 0x766a5d4f, t2 = 0xa6998d81, t3 = 0xf1d9c5b5;
    const uint32_t sel = 0x32103210 | ((q4 & 0x88888888) >> 1);
    uint32_t tmp[2];
#pragma unroll
    for (uint32_t i = 0; i < 2; ++i) {
        const uint32_t shift = 16 * i;
        const uint32_t lo = __byte_perm(t0, t1, q4 >> shift);
        const uint32_t hi = __byte_perm(t2, t3, q4 >> shift);
        tmp[i] = __byte_perm(lo, hi, sel >> shift);
    }
    return make_int2(__byte_perm(tmp[0], tmp[1], 0x6420), __byte_perm(tmp[0], tmp[1], 0x7531));
}

template <>
__device__ __forceinline__ void mmvh_load<GGML_TYPE_IQ4_XS>(const char * __restrict__ row, const int sb, const int sub, mmvh_units & u) {
    const block_iq4_xs * bq = (const block_iq4_xs *) row + sb;
    const int ib = sub;
    // block_iq4_xs is 136 bytes: only 8-byte alignment for qs
    const int2 qa = *(const int2 *) (bq->qs + 16*ib + 0);
    const int2 qb = *(const int2 *) (bq->qs + 16*ib + 8);
    const int q[4] = {qa.x, qa.y, qb.x, qb.y};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int2 v = mmvh_iq4nl_lookup(q[i]);
        u.b[0][i] = v.x;
        u.b[1][i] = v.y;
    }
    const int ls = ((bq->scales_l[ib/2] >> (4*(ib & 1))) & 0x0F) | (((bq->scales_h >> (2*ib)) & 0x03) << 4);
    const float d = __half2float(bq->d) * (ls - 32);
#pragma unroll
    for (int k = 0; k < 2; ++k) {
        u.blk[k] = ib;
        u.hf[k]  = k;
        u.ws[k]  = d;
        u.mt[k]  = 128.0f * d;
    }
}

// The IQ3 grids live in shared memory: random lookups into global memory are not L1-cached on GP100.
static __device__ __forceinline__ uint32_t * mmvh_grid_sh() {
    __shared__ uint32_t grid[512];
    return grid;
}

// Grid bytes g (unsigned) with 4 sign bits x4 -> bytes bias + (sign ? -g : g), all non-negative.
// The per-bit 0xFF mask comes from one multiply that spreads bits 0..3 to bytes 0..3.
static __device__ __forceinline__ int mmvh_signed_bias(const uint32_t g, const uint32_t x4, const uint32_t bias) {
    const uint32_t m = ((x4*0x00204081u) & 0x01010101u) * 0xFFu;
    return (g + bias) - ((g << 1) & m);
}

template <>
__device__ __forceinline__ void mmvh_load<GGML_TYPE_IQ3_S>(const char * __restrict__ row, const int sb, const int sub, mmvh_units & u) {
    const block_iq3_s * bq = (const block_iq3_s *) row + sb;
    const int ib = sub;
    // block_iq3_s is 110 bytes: only 2-byte alignment
    const int qs_packed[2] = {get_int_b2(bq->qs, 2*ib + 0), get_int_b2(bq->qs, 2*ib + 1)};
    const uint8_t * qs = (const uint8_t *) qs_packed;
    const int qh    = bq->qh[ib];
    const int signs = get_int_b2(bq->signs, ib);
#pragma unroll
    for (int l = 0; l < 8; ++l) {
        const uint32_t g = mmvh_grid_sh()[qs[l] | (((qh >> l) & 1) << 8)];
        u.b[l/4][l%4] = mmvh_signed_bias(g, (signs >> 4*l) & 0xF, 0x10101010);
    }
    const int   ls = (bq->scales[ib/2] >> 4*(ib & 1)) & 0xF;
    const float d  = __half2float(bq->d) * (1 + 2*ls);
#pragma unroll
    for (int k = 0; k < 2; ++k) {
        u.blk[k] = ib;
        u.hf[k]  = k;
        u.ws[k]  = d;
        u.mt[k]  = 16.0f * d;
    }
}

template <>
__device__ __forceinline__ void mmvh_load<GGML_TYPE_IQ3_XXS>(const char * __restrict__ row, const int sb, const int sub, mmvh_units & u) {
    const block_iq3_xxs * bq = (const block_iq3_xxs *) row + sb;
    const int ib = sub;
    // block_iq3_xxs is 98 bytes: only 2-byte alignment
    const int qs_packed[2] = {get_int_b2(bq->qs, 2*ib + 0), get_int_b2(bq->qs, 2*ib + 1)};
    const uint8_t * qs = (const uint8_t *) qs_packed;
    const uint32_t aux32 = get_int_b2(bq->qs, QK_K/16 + ib);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const uint32_t v = (aux32 >> 7*j) & 0x7F;
        const uint32_t s = v ^ ((__popc(v) & 1) << 7); // 8th sign is the parity
        u.b[j/2][2*(j%2) + 0] = mmvh_signed_bias(mmvh_grid_sh()[qs[2*j + 0]], s & 0xF, 0x40404040);
        u.b[j/2][2*(j%2) + 1] = mmvh_signed_bias(mmvh_grid_sh()[qs[2*j + 1]], s >> 4,  0x40404040);
    }
    const float d = __half2float(bq->d) * (0.5f + (aux32 >> 28)) * 0.5f;
#pragma unroll
    for (int k = 0; k < 2; ++k) {
        u.blk[k] = ib;
        u.hf[k]  = k;
        u.ws[k]  = d;
        u.mt[k]  = 64.0f * d;
    }
}

// The IQ2 grids are 8 bytes per entry (two uint32 here), up to 1024 entries.
static __device__ __forceinline__ uint32_t * mmvh_grid2_sh() {
    __shared__ uint32_t grid[2048];
    return grid;
}

// One 8-weight group: grid entry idx with 8 sign bits -> 2 ints of bytes 64 + (sign ? -g : g).
static __device__ __forceinline__ void mmvh_iq2_group(const uint32_t idx, const uint32_t signs, int * out) {
    out[0] = mmvh_signed_bias(mmvh_grid2_sh()[2*idx + 0], signs & 0xF, 0x40404040);
    out[1] = mmvh_signed_bias(mmvh_grid2_sh()[2*idx + 1], signs >> 4,  0x40404040);
}

// 7 stored sign bits, the 8th is the parity (ksigns_iq2xs)
static __device__ __forceinline__ uint32_t mmvh_ksigns(const uint32_t v) {
    return v ^ ((__popc(v) & 1) << 7);
}

template <>
__device__ __forceinline__ void mmvh_load<GGML_TYPE_IQ2_XXS>(const char * __restrict__ row, const int sb, const int sub, mmvh_units & u) {
    const block_iq2_xxs * bq = (const block_iq2_xxs *) row + sb;
    const int ib = sub;
    // block_iq2_xxs is 66 bytes: only 2-byte alignment
    const uint32_t idx   = get_int_b2(bq->qs, 2*ib + 0);
    const uint32_t aux32 = get_int_b2(bq->qs, 2*ib + 1);
#pragma unroll
    for (int l = 0; l < 4; ++l) {
        mmvh_iq2_group((idx >> 8*l) & 0xFF, mmvh_ksigns((aux32 >> 7*l) & 0x7F), &u.b[l/2][2*(l%2)]);
    }
    const float d = __half2float(bq->d) * (0.5f + (aux32 >> 28)) * 0.25f;
#pragma unroll
    for (int k = 0; k < 2; ++k) {
        u.blk[k] = ib;
        u.hf[k]  = k;
        u.ws[k]  = d;
        u.mt[k]  = 64.0f * d;
    }
}

template <>
__device__ __forceinline__ void mmvh_load<GGML_TYPE_IQ2_XS>(const char * __restrict__ row, const int sb, const int sub, mmvh_units & u) {
    const block_iq2_xs * bq = (const block_iq2_xs *) row + sb;
    const int ib = sub;
    // block_iq2_xs is 74 bytes: only 2-byte alignment
    const uint32_t q[2] = {(uint32_t) get_int_b2(bq->qs, 2*ib + 0), (uint32_t) get_int_b2(bq->qs, 2*ib + 1)};
#pragma unroll
    for (int l = 0; l < 4; ++l) {
        const uint32_t q2 = (q[l/2] >> 16*(l%2)) & 0xFFFF;
        mmvh_iq2_group(q2 & 511, mmvh_ksigns(q2 >> 9), &u.b[l/2][2*(l%2)]);
    }
    const float d = __half2float(bq->d) * 0.25f;
    const int   sc = bq->scales[ib];
#pragma unroll
    for (int k = 0; k < 2; ++k) {
        u.blk[k] = ib;
        u.hf[k]  = k;
        u.ws[k]  = d * (0.5f + ((sc >> 4*k) & 0xF));
        u.mt[k]  = 64.0f * u.ws[k];
    }
}

template <>
__device__ __forceinline__ void mmvh_load<GGML_TYPE_IQ2_S>(const char * __restrict__ row, const int sb, const int sub, mmvh_units & u) {
    const block_iq2_s * bq = (const block_iq2_s *) row + sb;
    const int ib = sub;
    // block_iq2_s is 82 bytes: only 2-byte alignment
    const uint32_t idx   = get_int_b2(bq->qs, ib);
    const uint32_t signs = get_int_b2(bq->qs, QK_K/32 + ib);
    const int      qh    = bq->qh[ib];
#pragma unroll
    for (int l = 0; l < 4; ++l) {
        mmvh_iq2_group(((idx >> 8*l) & 0xFF) | ((qh << (8 - 2*l)) & 0x300), (signs >> 8*l) & 0xFF, &u.b[l/2][2*(l%2)]);
    }
    const float d = __half2float(bq->d) * 0.25f;
    const int   sc = bq->scales[ib];
#pragma unroll
    for (int k = 0; k < 2; ++k) {
        u.blk[k] = ib;
        u.hf[k]  = k;
        u.ws[k]  = d * (0.5f + ((sc >> 4*k) & 0xF));
        u.mt[k]  = 64.0f * u.ws[k];
    }
}

// --- matvec ------------------------------------------------------------------

// R = rows per warp: the activation loads are shared across them
// NU = units of 16 weights per lane (4 only for the H41 Q6_K path)
template <ggml_type type, int ncols, int R, int NU>
__launch_bounds__(MMVH_NWARPS*WARP_SIZE, 1)
static __global__ void mul_mat_vec_h(const char * __restrict__ vx, const half * __restrict__ xh, const float4 * __restrict__ meta,
        float * __restrict__ dst, const int nsb, const int nrows, const int64_t stride_row_x,
        const int64_t ncols_padded, const int64_t stride_col_dst) {
    if constexpr (type == GGML_TYPE_IQ3_S || type == GGML_TYPE_IQ3_XXS) {
        const int n = type == GGML_TYPE_IQ3_S ? 512 : 256;
        for (int i = threadIdx.y*WARP_SIZE + threadIdx.x; i < n; i += MMVH_NWARPS*WARP_SIZE) {
            mmvh_grid_sh()[i] = type == GGML_TYPE_IQ3_S ? iq3s_grid[i] : iq3xxs_grid[i];
        }
        __syncthreads();
    }
    if constexpr (type == GGML_TYPE_IQ2_S || type == GGML_TYPE_IQ2_XS || type == GGML_TYPE_IQ2_XXS) {
        const int n = type == GGML_TYPE_IQ2_S ? 1024 : type == GGML_TYPE_IQ2_XS ? 512 : 256;
        const uint32_t * src = (const uint32_t *) (type == GGML_TYPE_IQ2_S ? iq2s_grid : type == GGML_TYPE_IQ2_XS ? iq2xs_grid : iq2xxs_grid);
        for (int i = threadIdx.y*WARP_SIZE + threadIdx.x; i < 2*n; i += MMVH_NWARPS*WARP_SIZE) {
            mmvh_grid2_sh()[i] = src[i];
        }
        __syncthreads();
    }
    const int row0 = R*(MMVH_NWARPS*blockIdx.x + threadIdx.y);
    if (row0 >= nrows) {
        return;
    }
    const int lane = threadIdx.x;
    constexpr int lps = QK_K / (16*NU); // lanes per superblock
    const int sub  = lane & (lps - 1);

    const char * xrow[R];
#pragma unroll
    for (int r = 0; r < R; ++r) {
        xrow[r] = vx + min(row0 + r, nrows - 1)*stride_row_x;
    }

    float tmp[ncols][R] = {{0.0f}};

    for (int sb = lane / lps; sb < nsb; sb += WARP_SIZE/lps) {
        std::conditional_t<NU == 4, mmvh_units4, mmvh_units> u[R];
#pragma unroll
        for (int r = 0; r < R; ++r) {
            if constexpr (NU == 4) {
                mmvh_load4<type>(xrow[r], sb, sub, u[r]);
            } else {
                mmvh_load<type>(xrow[r], sb, sub, u[r]);
            }
        }

#pragma unroll
        for (int k = 0; k < NU; ++k) {
            const int64_t i0 = int64_t(sb)*QK_K + 32*u[0].blk[k] + 16*u[0].hf[k];
#pragma unroll
            for (int c = 0; c < ncols; ++c) {
                const int4 * yp = (const int4 *) (xh + c*ncols_padded + i0);
                const int4 y0 = yp[0];
                const int4 y1 = yp[1];
                const int  ya[8] = {y0.x, y0.y, y0.z, y0.w, y1.x, y1.y, y1.z, y1.w};
                const float4 m = meta[c*(ncols_padded/32) + i0/32];
                const float msum = u[0].hf[k] ? m.z : m.y;

#pragma unroll
                for (int r = 0; r < R; ++r) {
                    half2 acc = __float2half2_rn(0.0f);
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        const int lo =  u[r].b[k][i]       & 0x00FF00FF;
                        const int hi = (u[r].b[k][i] >> 8) & 0x00FF00FF;
                        acc = __hfma2(*(const half2 *) &lo, *(const half2 *) &ya[2*i + 0], acc);
                        acc = __hfma2(*(const half2 *) &hi, *(const half2 *) &ya[2*i + 1], acc);
                    }
                    const float2 accf = __half22float2(acc);
                    tmp[c][r] += u[r].ws[k] * ((accf.x + accf.y) * m.x) - u[r].mt[k] * msum;
                }
            }
        }
    }

#pragma unroll
    for (int c = 0; c < ncols; ++c) {
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const float v = warp_reduce_sum(tmp[c][r]);
            if (lane == 0 && row0 + r < nrows) {
                dst[c*stride_col_dst + row0 + r] = v;
            }
        }
    }
}

template <ggml_type type, int ncols, int NU>
static void mul_mat_vec_h_launch(const char * vx, const half * xh, const float4 * meta, float * dst, const int nsb,
        const int64_t nrows, const int64_t stride_row_x, const int64_t ncols_padded, const int64_t stride_col_dst, cudaStream_t stream) {
    constexpr int R = ncols <= 4 ? 4 : 2;
    const dim3 grid((nrows + MMVH_NWARPS*R - 1) / (MMVH_NWARPS*R));
    const dim3 block(WARP_SIZE, MMVH_NWARPS);
    mul_mat_vec_h<type, ncols, R, NU><<<grid, block, 0, stream>>>(vx, xh, meta, dst, nsb, nrows, stride_row_x, ncols_padded, stride_col_dst);
}

template <ggml_type type, int NU = 2>
static void mul_mat_vec_h_cuda(const char * vx, const half * xh, const float4 * meta, float * dst, const int64_t ne00,
        const int64_t nrows, const int64_t ncols, const int64_t stride_row_x, const int64_t ncols_padded,
        const int64_t stride_col_dst, cudaStream_t stream) {
    const int nsb = ne00 / QK_K;
    switch (ncols) {
#define MMVH_CASE(n) case n: mul_mat_vec_h_launch<type, n, NU>(vx, xh, meta, dst, nsb, nrows, stride_row_x, ncols_padded, stride_col_dst, stream); break;
        MMVH_CASE(1) MMVH_CASE(2) MMVH_CASE(3) MMVH_CASE(4)
        MMVH_CASE(5) MMVH_CASE(6) MMVH_CASE(7) MMVH_CASE(8)
#undef MMVH_CASE
        default: GGML_ABORT("fatal error");
    }
}

void ggml_cuda_mul_mat_vec_h(ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    cudaStream_t stream = ctx.stream();

    const int64_t ne00  = src0->ne[0];
    const int64_t nrows = src0->ne[1];
    const int64_t ncols = src1->ne[1];

    ggml_cuda_pool_alloc<half>   xh(ctx.pool(), ncols*ne00);
    ggml_cuda_pool_alloc<float4> meta(ctx.pool(), ncols*ne00/32);

    {
        const dim3 grid(ne00/32, ncols);
        mmvh_convert_y<<<grid, WARP_SIZE, 0, stream>>>((const float *) src1->data, xh.get(), meta.get(),
            src1->nb[1] / sizeof(float), ne00);
    }

    const char  * vx  = (const char *) src0->data;
    float       * d   = (float *) dst->data;
    const int64_t sdc = dst->nb[1] / sizeof(float);

    switch (src0->type) {
        case GGML_TYPE_Q4_K:
            mul_mat_vec_h_cuda<GGML_TYPE_Q4_K>(vx, xh.get(), meta.get(), d, ne00, nrows, ncols, src0->nb[1], ne00, sdc, stream);
            break;
        case GGML_TYPE_Q5_K:
            mul_mat_vec_h_cuda<GGML_TYPE_Q5_K>(vx, xh.get(), meta.get(), d, ne00, nrows, ncols, src0->nb[1], ne00, sdc, stream);
            break;
        case GGML_TYPE_Q6_K:
            if (mmvh_q6k_wide_enabled()) {
                mul_mat_vec_h_cuda<GGML_TYPE_Q6_K, 4>(vx, xh.get(), meta.get(), d, ne00, nrows, ncols, src0->nb[1], ne00, sdc, stream);
            } else {
                mul_mat_vec_h_cuda<GGML_TYPE_Q6_K>(vx, xh.get(), meta.get(), d, ne00, nrows, ncols, src0->nb[1], ne00, sdc, stream);
            }
            break;
        case GGML_TYPE_IQ4_XS:
            mul_mat_vec_h_cuda<GGML_TYPE_IQ4_XS>(vx, xh.get(), meta.get(), d, ne00, nrows, ncols, src0->nb[1], ne00, sdc, stream);
            break;
        case GGML_TYPE_IQ3_S:
            mul_mat_vec_h_cuda<GGML_TYPE_IQ3_S>(vx, xh.get(), meta.get(), d, ne00, nrows, ncols, src0->nb[1], ne00, sdc, stream);
            break;
        case GGML_TYPE_Q3_K:
            mul_mat_vec_h_cuda<GGML_TYPE_Q3_K, 4>(vx, xh.get(), meta.get(), d, ne00, nrows, ncols, src0->nb[1], ne00, sdc, stream);
            break;
        case GGML_TYPE_Q2_K:
            mul_mat_vec_h_cuda<GGML_TYPE_Q2_K, 4>(vx, xh.get(), meta.get(), d, ne00, nrows, ncols, src0->nb[1], ne00, sdc, stream);
            break;
        case GGML_TYPE_IQ2_S:
            mul_mat_vec_h_cuda<GGML_TYPE_IQ2_S>(vx, xh.get(), meta.get(), d, ne00, nrows, ncols, src0->nb[1], ne00, sdc, stream);
            break;
        case GGML_TYPE_IQ2_XS:
            mul_mat_vec_h_cuda<GGML_TYPE_IQ2_XS>(vx, xh.get(), meta.get(), d, ne00, nrows, ncols, src0->nb[1], ne00, sdc, stream);
            break;
        case GGML_TYPE_IQ2_XXS:
            mul_mat_vec_h_cuda<GGML_TYPE_IQ2_XXS>(vx, xh.get(), meta.get(), d, ne00, nrows, ncols, src0->nb[1], ne00, sdc, stream);
            break;
        case GGML_TYPE_IQ3_XXS:
            mul_mat_vec_h_cuda<GGML_TYPE_IQ3_XXS>(vx, xh.get(), meta.get(), d, ne00, nrows, ncols, src0->nb[1], ne00, sdc, stream);
            break;
        default:
            GGML_ABORT("fatal error");
    }
}
