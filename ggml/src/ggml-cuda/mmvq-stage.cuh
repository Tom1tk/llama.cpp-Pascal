#include "common.cuh"

// Pascal (sm_60) one-column Q6_K matvec with the weights and the q8_1 activation staged through
// shared memory in 16-byte loads. GGML_PASCAL_MMVQ_STAGE, on by default on sm_60 (H71).

bool ggml_cuda_mmvq_stage_can_use(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst);

void ggml_cuda_mmvq_stage(const void * vx, const void * vy, float * dst,
    int ncols_x, int nrows_x, int stride_row_x, cudaStream_t stream);
