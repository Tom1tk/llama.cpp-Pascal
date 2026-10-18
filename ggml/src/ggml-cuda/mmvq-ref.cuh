#include "common.cuh"

// The reference build's mmvq (mmvq-ref.cu, vecdotq-ref.cuh, vecdotq-p100.cuh), vendored for its sm_60
// single-column path: P100 dot products without dp4a, staged q8_1 activations, per-type geometry (H75).

#define MMVQ_MAX_BATCH_SIZE 8

int get_mmvq_mmid_max_batch_ref(ggml_type type, int cc);

void ggml_cuda_mul_mat_vec_q_ref(ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst, const ggml_cuda_mm_fusion_args_host * fusion = nullptr);

// default on for sm_60, GGML_PASCAL_MMVQ_REF=0 disables
bool ggml_cuda_mmvq_ref_can_use(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst);

// node pointers are reused with fresh data across graph evaluations: call at the start of each
void ggml_cuda_mmvq_ref_invalidate(int device);
