#include "mmvq-stage.cuh"
#include "vecdotq.cuh"

// sm_60 is load-issue bound on this kernel, not bandwidth bound: block_q6_K is 210 bytes (2 mod 4),
// so reading quant words straight from global costs two 16-bit loads each. Each warp instead pulls
// its contiguous run of blocks in with aligned uint4 loads from the 16-byte boundary below it, keeps
// the misalignment in shared memory (free there) and runs the stock vec_dot on the staged copy.
// The q8_1 activation is staged the same way: its 36-byte blocks make the direct reads fan out.
// ponytail: Q6_K only; other types and the per-trip address hoist are a follow-up if this pays.

#define MMVQ_STAGE_NWARPS 2
#define MMVQ_STAGE_ROWS   2

template <int nwarps, int rows>
__launch_bounds__(nwarps*WARP_SIZE, 1)
static __global__ void mul_mat_vec_q6_K_stage(const char * __restrict__ vx, const block_q8_1 * __restrict__ y,
                                              float * __restrict__ dst, const int ncols_x, const int nrows_x, const int stride_row_x) {
    constexpr int bs       = sizeof(block_q6_K);
    constexpr int ybs      = (QK_K/QK8_1)*sizeof(block_q8_1);
    constexpr int lanes    = QI6_K/VDR_Q6_K_Q8_1_MMVQ;  // lanes per block
    constexpr int bpw      = WARP_SIZE/lanes;           // blocks per warp per trip
    constexpr int x_u4     = (bpw*bs  + 30)/16;         // + slack for the misalignment
    constexpr int y_u4     = (bpw*ybs + 30)/16;
    constexpr int x_rounds = (x_u4 + WARP_SIZE - 1)/WARP_SIZE;
    constexpr int y_rounds = (y_u4 + WARP_SIZE - 1)/WARP_SIZE;

    __shared__ uint4 xs[nwarps][rows][x_u4];
    __shared__ uint4 ys[nwarps][y_u4];
    __shared__ float red[nwarps > 1 ? nwarps - 1 : 1][rows][WARP_SIZE];

    const int lane = threadIdx.x;
    const int w    = threadIdx.y;
    const int row0 = rows*blockIdx.x;
    const int nblk = ncols_x/QK_K;
    const int sub  = lane/lanes;
    const int kqs  = VDR_Q6_K_Q8_1_MMVQ*(lane % lanes);

    float tmp[rows] = {0.0f};
    int   mis[rows];

    for (int kb = w*bpw; kb < nblk; kb += nwarps*bpw) {
        const int n = min(bpw, nblk - kb);
        __syncwarp(); // the previous trip's readers are done with the stage
#pragma unroll
        for (int i = 0; i < rows; ++i) {
            // clamp: surplus rows of the last block are computed on the last row and dropped at write-back
            const char  * g   = vx + (size_t(min(row0 + i, nrows_x - 1))*stride_row_x + kb)*bs;
            const int     m   = (int) ((uintptr_t) g & 15);
            const uint4 * g16 = (const uint4 *) (g - m);
            const int     nu4 = (m + n*bs + 15) >> 4;
            mis[i] = m;
#pragma unroll
            for (int r = 0; r < x_rounds; ++r) { // compile-time trip count keeps these LDG, not generic loads
                if (r*WARP_SIZE + lane < nu4) {
                    xs[w][i][r*WARP_SIZE + lane] = __ldg(g16 + r*WARP_SIZE + lane);
                }
            }
        }
        const char  * yg   = (const char *) (y + kb*(QK_K/QK8_1));
        const int     ym   = (int) ((uintptr_t) yg & 15);
        const uint4 * yg16 = (const uint4 *) (yg - ym);
        const int     nyu4 = (ym + n*ybs + 15) >> 4;
#pragma unroll
        for (int r = 0; r < y_rounds; ++r) {
            if (r*WARP_SIZE + lane < nyu4) {
                ys[w][r*WARP_SIZE + lane] = __ldg(yg16 + r*WARP_SIZE + lane);
            }
        }
        __syncwarp();

        if (kb + sub < nblk) {
            const block_q8_1 * yb = (const block_q8_1 *) ((const char *) ys[w] + ym + sub*ybs);
#pragma unroll
            for (int i = 0; i < rows; ++i) {
                tmp[i] += vec_dot_q6_K_q8_1((const char *) xs[w][i] + mis[i] + sub*bs, yb, 0, kqs);
            }
        }
    }

    if (w > 0) {
#pragma unroll
        for (int i = 0; i < rows; ++i) {
            red[w - 1][i][lane] = tmp[i];
        }
    }
    __syncthreads();
    if (w > 0) {
        return;
    }
#pragma unroll
    for (int i = 0; i < rows; ++i) {
#pragma unroll
        for (int l = 0; l < nwarps - 1; ++l) {
            tmp[i] += red[l][i][lane];
        }
        tmp[i] = warp_reduce_sum(tmp[i]);
        if (lane == i && row0 + i < nrows_x) {
            dst[row0 + i] = tmp[i];
        }
    }
}

bool ggml_cuda_mmvq_stage_can_use(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst) {
    static const int env = ggml_cuda_pascal_env("GGML_PASCAL_MMVQ_STAGE");
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    return ggml_cuda_pascal_lever(env, cc) && cc == GGML_CUDA_CC_PASCAL &&
        src0->type == GGML_TYPE_Q6_K && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 &&
        src1->ne[1] == 1 && src0->ne[2] == 1 && src0->ne[3] == 1 && src1->ne[2] == 1 && src1->ne[3] == 1 &&
        src0->ne[0] % QK_K == 0;
}

void ggml_cuda_mmvq_stage(const void * vx, const void * vy, float * dst,
        const int ncols_x, const int nrows_x, const int stride_row_x, cudaStream_t stream) {
    const dim3 grid((nrows_x + MMVQ_STAGE_ROWS - 1)/MMVQ_STAGE_ROWS);
    const dim3 block(WARP_SIZE, MMVQ_STAGE_NWARPS);
    mul_mat_vec_q6_K_stage<MMVQ_STAGE_NWARPS, MMVQ_STAGE_ROWS><<<grid, block, 0, stream>>>(
        (const char *) vx, (const block_q8_1 *) vy, dst, ncols_x, nrows_x, stride_row_x);
}
