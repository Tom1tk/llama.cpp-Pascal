// Pascal fork (H46): prefill flash attention through cuBLAS, GGML_PASCAL_FA_GEMM (on by default on sm_60).
// sm_60 has no tensor cores; the tile kernel reaches ~5.5 TFLOPS on D=256 prefill,
// while cuBLAS reaches ~7.6 (Q*K, fp32 accumulate) and ~14.5 (P*V, HGEMM) on the same shapes.
// Per KV chunk of C tokens: S = K^T Q (f32), online softmax -> P (f16), O_c = V P (f16),
// O = O*alpha + O_c (f32). Precision: Q*K and softmax stay f32 as in the tile kernel;
// P*V accumulates in f16 per chunk (the tile kernel accumulates it in half2 over all of KV).
#pragma once

#include "common.cuh"
#include "convert.cuh"
#include "fattn-common.cuh"

#include <cmath>
#include <cstdlib>

static bool ggml_cuda_fattn_pascal_gemm_enabled() {
    static const int env = ggml_cuda_pascal_env("GGML_PASCAL_FA_GEMM");
    return ggml_cuda_pascal_lever(env, ggml_cuda_info().devices[ggml_cuda_get_device()].cc);
}

static bool ggml_cuda_fattn_pascal_gemm_supported(const ggml_tensor * dst) {
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];
    float max_bias, softcap;
    memcpy(&max_bias, (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&softcap,  (const float *) dst->op_params + 2, sizeof(float));
    static const int min_q  = getenv("GGML_PASCAL_FA_GEMM_MIN_Q")  ? atoi(getenv("GGML_PASCAL_FA_GEMM_MIN_Q"))  : 512;
    static const int min_kv = getenv("GGML_PASCAL_FA_GEMM_MIN_KV") ? atoi(getenv("GGML_PASCAL_FA_GEMM_MIN_KV")) : 4096;
    return ggml_cuda_fattn_pascal_gemm_enabled() &&
        Q->ne[1] >= min_q && K->ne[1] >= min_kv &&
        mask && mask->ne[2] == 1 && mask->ne[3] == 1 && !dst->src[4] &&
        max_bias == 0.0f && softcap == 0.0f &&
        Q->ne[3] == 1 && K->ne[3] == 1 && V->ne[3] == 1 &&
        Q->ne[0] == V->ne[0] && Q->ne[0] % 2 == 0 && Q->ne[2] % K->ne[2] == 0;
}

// Q [D, n_q, H] f32 (strided) -> Qh [D, n_q, H] f16 contiguous
static __global__ void k_fattn_pg_q_f16(const char * Q, half * Qh, const int D, const int n_q,
        const size_t nb1, const size_t nb2) {
    const int j = blockIdx.x, h = blockIdx.y;
    const float * q = (const float *) (Q + j*nb1 + h*nb2);
    half * o = Qh + (size_t) D*(j + (size_t) n_q*h);
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        o[d] = __float2half(q[d]);
    }
}

// One block per query j, looping over the H heads so the mask row is read once.
// S/P column for (h, j) is h*n_q + j with leading dimension C.
template <int C>
static __global__ void k_fattn_pg_softmax(const float * S, half * P, const half * mask, const size_t mask_nb1,
        float * m, float * l, float * alpha, const int Cc, const int kv0, const int n_q, const int H,
        const float scale, const bool first) {
    constexpr int nt = 256, npt = C/nt;
    const int j = blockIdx.x;
    const half * mrow = (const half *) ((const char *) mask + j*mask_nb1) + kv0;
    float mv[npt];
#pragma unroll
    for (int k = 0; k < npt; ++k) {
        const int i = threadIdx.x + k*nt;
        mv[k] = i < Cc ? __half2float(mrow[i]) : -INFINITY;
    }
    __shared__ float red[nt/WARP_SIZE];
    for (int h = 0; h < H; ++h) {
        const size_t col = (size_t) h*n_q + j;
        const float * s_col = S + col*C;
        float sv[npt];
        float smax = -INFINITY;
#pragma unroll
        for (int k = 0; k < npt; ++k) {
            const int i = threadIdx.x + k*nt;
            sv[k] = i < Cc ? s_col[i]*scale + mv[k] : -INFINITY;
            smax = fmaxf(smax, sv[k]);
        }
        smax = warp_reduce_max(smax);
        if (threadIdx.x % WARP_SIZE == 0) red[threadIdx.x/WARP_SIZE] = smax;
        __syncthreads();
        smax = red[0];
#pragma unroll
        for (int w = 1; w < nt/WARP_SIZE; ++w) smax = fmaxf(smax, red[w]);
        __syncthreads();

        const float m_old = first ? -INFINITY : m[col];
        const float m_new = fmaxf(m_old, smax);
        half * p_col = P + col*C;
        float sum = 0.0f;
#pragma unroll
        for (int k = 0; k < npt; ++k) {
            const int i = threadIdx.x + k*nt;
            const float p = m_new == -INFINITY ? 0.0f : expf(sv[k] - m_new);
            sum += p;
            if (i < Cc) p_col[i] = __float2half(p);
        }
        sum = warp_reduce_sum(sum);
        if (threadIdx.x % WARP_SIZE == 0) red[threadIdx.x/WARP_SIZE] = sum;
        __syncthreads();
        if (threadIdx.x == 0) {
            for (int w = 1; w < nt/WARP_SIZE; ++w) sum += red[w];
            const float a = m_old == -INFINITY ? 0.0f : expf(m_old - m_new);
            m[col]     = m_new;
            l[col]     = (first ? 0.0f : l[col])*a + sum;
            alpha[col] = a;
        }
        __syncthreads();
    }
}

// O[d, col] = O*alpha + Oc; on the last chunk write dst[d, h, j] = O / l instead.
static __global__ void k_fattn_pg_combine(float * O, const half * Oc, const float * alpha, const float * l,
        float * dst, const size_t dst_nb1, const size_t dst_nb2, const int D, const int n_q,
        const bool first, const bool last) {
    const int col = blockIdx.x;
    const float a = alpha[col];
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        const size_t i = (size_t) col*D + d;
        const float o = (first ? 0.0f : O[i]*a) + __half2float(Oc[i]);
        if (last) {
            const int h = col / n_q, j = col % n_q;
            ((float *) ((char *) dst + h*dst_nb1 + j*dst_nb2))[d] = o / l[col];
        } else {
            O[i] = o;
        }
    }
}

// f16 view of K or V, converting into the FA extra buffer if needed (same as launch_fattn).
static const char * ggml_cuda_fattn_pg_to_f16(const ggml_tensor * t, uintptr_t extra, size_t & nb1, size_t & nb2,
        cudaStream_t stream) {
    nb1 = t->nb[1];
    nb2 = t->nb[2];
    if (t->type == GGML_TYPE_F16) {
        return (const char *) t->data;
    }
    GGML_ASSERT(extra != 0);
    half * dst = (half *) extra;
    const size_t bs = ggml_blck_size(t->type), ts = ggml_type_size(t->type);
    if (ggml_is_contiguously_allocated(t)) {
        ggml_get_to_fp16_cuda(t->type)(t->data, dst, ggml_nelements(t), stream);
        nb1 = nb1*bs*sizeof(half)/ts;
        nb2 = nb2*bs*sizeof(half)/ts;
    } else {
        GGML_ASSERT(t->nb[0] == ts);
        ggml_get_to_fp16_nc_cuda(t->type)(t->data, dst, t->ne[0], t->ne[1], t->ne[2], t->ne[3],
            t->nb[1]/ts, t->nb[2]/ts, t->nb[3]/ts, stream);
        nb1 = t->ne[0]*sizeof(half);
        nb2 = t->ne[1]*nb1;
    }
    return (const char *) dst;
}

static void ggml_cuda_flash_attn_ext_pascal_gemm(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];
    cudaStream_t stream = ctx.stream();

    const int D = Q->ne[0], n_q = Q->ne[1], H = Q->ne[2], Hkv = K->ne[2], gqa = H/Hkv;
    const int n_kv = K->ne[1];
    float scale;
    memcpy(&scale, (const float *) dst->op_params + 0, sizeof(float));

    const bool V_is_K_view = V->view_src && (V->view_src == K || (V->view_src == K->view_src && V->view_offs == K->view_offs));
    const ggml_cuda_flash_attn_ext_f16_extra_data f16_extra = ggml_cuda_flash_attn_ext_get_f16_extra_data(dst, true, true);
    size_t nb11, nb12, nb21, nb22;
    const char * K_h = ggml_cuda_fattn_pg_to_f16(K, f16_extra.K, nb11, nb12, stream);
    const char * V_h;
    if (V_is_K_view && V->type != GGML_TYPE_F16) {
        V_h = K_h; nb21 = nb11; nb22 = nb12;
    } else {
        V_h = ggml_cuda_fattn_pg_to_f16(V, f16_extra.V, nb21, nb22, stream);
    }

    // ponytail: S buffer is C*H*n_q f32 (~400 MB at 12 heads x 2048 q); halve C for more heads per card
    const bool wide = (size_t) H*n_q <= 12*2048;
    const int  C    = wide ? 4096 : 2048;
    const size_t cols = (size_t) H*n_q, cols_g = (size_t) gqa*n_q;

    ggml_cuda_pool & pool = ctx.pool();
    ggml_cuda_pool_alloc<half>  Qh(pool, (size_t) D*cols);
    ggml_cuda_pool_alloc<float> S (pool, (size_t) C*cols);
    ggml_cuda_pool_alloc<half>  P (pool, (size_t) C*cols);
    ggml_cuda_pool_alloc<half>  Oc(pool, (size_t) D*cols);
    ggml_cuda_pool_alloc<float> O (pool, (size_t) D*cols);
    ggml_cuda_pool_alloc<float> ml(pool, 3*cols);
    float * m = ml.ptr, * l = ml.ptr + cols, * alpha = ml.ptr + 2*cols;

    k_fattn_pg_q_f16<<<dim3(n_q, H), 256, 0, stream>>>((const char *) Q->data, Qh.ptr, D, n_q, Q->nb[1], Q->nb[2]);

    cublasHandle_t handle = ctx.cublas_handle();
    CUBLAS_CHECK(cublasSetStream(handle, stream));
    const float one = 1.0f, zero = 0.0f;
    const half  h1  = 1.0f, h0   = 0.0f;

    for (int kv0 = 0; kv0 < n_kv; kv0 += C) {
        const int  Cc    = std::min(C, n_kv - kv0);
        const bool first = kv0 == 0, last = kv0 + C >= n_kv;

        // S[Cc x cols_g] = K^T Q per kv head
        CUBLAS_CHECK(cublasGemmStridedBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, Cc, cols_g, D, &one,
            K_h + kv0*nb11, CUDA_R_16F, nb11/sizeof(half), nb12/sizeof(half),
            Qh.ptr, CUDA_R_16F, D, (long long) D*cols_g, &zero,
            S.ptr, CUDA_R_32F, C, (long long) C*cols_g, Hkv, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));

        if (wide) {
            k_fattn_pg_softmax<4096><<<n_q, 256, 0, stream>>>(S.ptr, P.ptr, (const half *) mask->data, mask->nb[1],
                m, l, alpha, Cc, kv0, n_q, H, scale, first);
        } else {
            k_fattn_pg_softmax<2048><<<n_q, 256, 0, stream>>>(S.ptr, P.ptr, (const half *) mask->data, mask->nb[1],
                m, l, alpha, Cc, kv0, n_q, H, scale, first);
        }

        // Oc[D x cols_g] = V P per kv head
        CUBLAS_CHECK(cublasGemmStridedBatchedEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, D, cols_g, Cc, &h1,
            V_h + kv0*nb21, CUDA_R_16F, nb21/sizeof(half), nb22/sizeof(half),
            P.ptr, CUDA_R_16F, C, (long long) C*cols_g, &h0,
            Oc.ptr, CUDA_R_16F, D, (long long) D*cols_g, Hkv, CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT));

        k_fattn_pg_combine<<<cols, 256, 0, stream>>>(O.ptr, Oc.ptr, alpha, l, (float *) dst->data,
            dst->nb[1], dst->nb[2], D, n_q, first, last);
    }
    CUDA_CHECK(cudaGetLastError());
}
