#include "common.cuh"

// Pascal (sm_60) matvec for quantized weights without dp4a: weights are unpacked
// straight into half2 lanes as exact subnormals and multiplied with __hfma2.
// Opt-in via GGML_CUDA_PASCAL_MMVH=1. See docs/pascal/fork-plan-2026-10-14.md (D2).

bool ggml_cuda_mmvh_can_use(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst);

void ggml_cuda_mul_mat_vec_h(ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);

// Forget the cached converted activation (called at the start of every graph compute).
void ggml_cuda_mmvh_invalidate(ggml_backend_cuda_context & ctx);
