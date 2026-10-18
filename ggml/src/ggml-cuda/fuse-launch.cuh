#include "common.cuh"

// Decode is host-launch bound on Pascal (~10 µs of host time per kernel), so these fold elementwise and
// norm nodes of the qwen35 decode graph into fewer launches. Default on for sm_60,
// GGML_PASCAL_FUSE_LAUNCH=0 disables. Each returns false (nothing launched) when it does not apply.

bool ggml_cuda_fuse_launch_enabled(int cc);

// residual ADD -> RMS_NORM -> MUL(weight): writes the sum and the normed row in one launch
bool ggml_cuda_fuse_add_rms_norm_mul(ggml_backend_cuda_context & ctx, ggml_tensor * add, ggml_tensor * rms, ggml_tensor * mul);

// RMS_NORM -> MUL(weight) -> MUL(silu(gate)) in one launch; check_only: only report whether it applies
bool ggml_cuda_fuse_rms_norm_mul_silu_gate(ggml_backend_cuda_context & ctx, ggml_tensor * rms, ggml_tensor * mul,
    const ggml_tensor * gate, ggml_tensor * out, bool check_only);

// two RMS_NORM -> SCALE of the same shape and parameters (the delta net's q and k l2 norms) in one launch
bool ggml_cuda_fuse_rms_norm_scale2(ggml_backend_cuda_context & ctx, ggml_tensor * n1, ggml_tensor * s1, ggml_tensor * n2, ggml_tensor * s2);

// ADD(row bias) -> SOFTPLUS -> MUL(row scale) (the delta net's gate) in one launch
bool ggml_cuda_fuse_add_softplus_mul(ggml_backend_cuda_context & ctx, ggml_tensor * add, ggml_tensor * sp, ggml_tensor * mul);

// Conv state update copies (H76, ported from the reference build). Default on for sm_60,
// GGML_PASCAL_CONV_STATE_CPY=0 disables; the concat half is ggml_cuda_op_concat_cpy (concat.cuh).
bool ggml_cuda_conv_state_cpy_enabled(int cc);

// consecutive same-shape f32 CPYs between views of the same two tensors (the delta net's K = n_rs_seq + 1
// conv state rollback snapshots) in one launch; returns the nodes to skip after i, 0 = not fused
int ggml_cuda_fuse_cpy_batch(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i);
