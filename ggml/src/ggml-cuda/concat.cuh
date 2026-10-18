#include "common.cuh"

#define CUDA_CONCAT_BLOCK_SIZE 256

void ggml_cuda_op_concat(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// CONCAT -> CPY of its tail (the delta net conv state update) in one launch; false (nothing launched) if it does not apply
bool ggml_cuda_op_concat_cpy(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * cpy);
