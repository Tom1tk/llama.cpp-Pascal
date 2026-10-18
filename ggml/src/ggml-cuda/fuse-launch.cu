#include "fuse-launch.cuh"
#include "unary.cuh"

// Ported from the reference build's graph-op fusions (H72).

bool ggml_cuda_fuse_launch_enabled(const int cc) {
    static const int env = ggml_cuda_pascal_env("GGML_PASCAL_FUSE_LAUNCH");
    return ggml_cuda_pascal_lever(env, cc);
}

bool ggml_cuda_conv_state_cpy_enabled(const int cc) {
    static const int env = ggml_cuda_pascal_env("GGML_PASCAL_CONV_STATE_CPY");
    return ggml_cuda_pascal_lever(env, cc);
}

static bool overlaps(const ggml_tensor * x, const ggml_tensor * y) {
    const char * x0 = (const char *) x->data, * y0 = (const char *) y->data;
    return x0 < y0 + ggml_nbytes(y) && y0 < x0 + ggml_nbytes(x);
}

// one block per row; the sum is written out (it is the next residual) and kept in registers for the norm
template <int block_size>
static __global__ void add_rms_norm_mul_f32(const float * a, const float * b, float * sum_out, const float * mul,
                                            float * dst, const int ncols, const int64_t sa, const int64_t sb,
                                            const int64_t ss, const int64_t sd, const float eps) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    a += row*sa; b += row*sb; sum_out += row*ss; dst += row*sd;

    constexpr int max_regs = 8;
    extern __shared__ float s_sum[];

    float xv[max_regs];
    float tmp = 0.0f;
#pragma unroll
    for (int u = 0; u < max_regs; ++u) {
        const int col = tid + u*block_size;
        xv[u] = col < ncols ? a[col] + b[col] : 0.0f;
        tmp += xv[u] * xv[u];
    }
#pragma unroll
    for (int u = 0; u < max_regs; ++u) {
        const int col = tid + u*block_size;
        if (col < ncols) {
            sum_out[col] = xv[u];
        }
    }

    tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum);

    const float scale = rsqrtf(tmp / ncols + eps);

#pragma unroll
    for (int u = 0; u < max_regs; ++u) {
        const int col = tid + u*block_size;
        if (col < ncols) {
            dst[col] = scale * xv[u] * mul[col];
        }
    }
}

bool ggml_cuda_fuse_add_rms_norm_mul(ggml_backend_cuda_context & ctx, ggml_tensor * add, ggml_tensor * rms, ggml_tensor * mul) {
    const ggml_tensor * a = add->src[0];
    const ggml_tensor * b = add->src[1];
    const ggml_tensor * w = mul->src[0] == rms ? mul->src[1] : mul->src[0];
    const int64_t ncols = add->ne[0], nrows = add->ne[1];
    auto f32_rows = [&](const ggml_tensor * t) {
        return t->type == GGML_TYPE_F32 && t->nb[0] == sizeof(float) && t->ne[0] == ncols && t->ne[1] == nrows &&
            t->ne[2] == 1 && t->ne[3] == 1 && t->nb[1] % sizeof(float) == 0;
    };
    if (!f32_rows(a) || !f32_rows(b) || !f32_rows(add) || !f32_rows(rms) || !f32_rows(mul) ||
            w->type != GGML_TYPE_F32 || w->ne[0] != ncols || ggml_nelements(w) != ncols || w->nb[0] != sizeof(float) ||
            ncols < 1024 || ncols > 1024*8 || rms->src[0] != add) {
        return false;
    }
    // a thread reads a and b at its own columns before writing add or mul there, so exact aliases are fine
    auto same = [](const ggml_tensor * x, const ggml_tensor * y) { return x->data == y->data && x->nb[1] == y->nb[1]; };
    if ((overlaps(mul, a) && !same(mul, a)) || (overlaps(mul, b) && !same(mul, b)) || overlaps(mul, add) ||
            overlaps(mul, w) || overlaps(add, w) || (overlaps(add, a) && !same(add, a)) || (overlaps(add, b) && !same(add, b))) {
        return false;
    }
    float eps;
    memcpy(&eps, rms->op_params, sizeof(float));
    const int64_t fs = sizeof(float);
    add_rms_norm_mul_f32<1024><<<(int) nrows, 1024, 32*sizeof(float), ctx.stream()>>>(
        (const float *) a->data, (const float *) b->data, (float *) add->data, (const float *) w->data, (float *) mul->data,
        (int) ncols, a->nb[1]/fs, b->nb[1]/fs, add->nb[1]/fs, mul->nb[1]/fs, eps);
    CUDA_CHECK(cudaGetLastError());
    return true;
}

// one warp per row of a short (head-sized) row. x is strided, dst and gate are contiguous; with
// x2/dst2 set, the upper half of gridDim.z normalises the second tensor (same shape and strides).
template <bool do_gate>
static __global__ void rms_norm_row_warp_f32(const float * x, float * dst, const int ncols,
                                             const int64_t s1, const int64_t s2, const int64_t s3, const float eps,
                                             const float * w, const float * gate, const float scale_out,
                                             const float * x2, float * dst2) {
    const int row = blockIdx.x, channel = blockIdx.y;
    int sample = blockIdx.z;
    if (x2 && sample >= (int) gridDim.z/2) {
        sample -= gridDim.z/2;
        x   = x2;
        dst = dst2;
    }
    x += sample*s3 + channel*s2 + row*s1;
    const int64_t o = ((int64_t) (sample*gridDim.y + channel)*gridDim.x + row)*ncols;
    dst += o;

    float sum = 0.0f;
    for (int c = threadIdx.x; c < ncols; c += WARP_SIZE) {
        sum += x[c]*x[c];
    }
    sum = warp_reduce_sum(sum);
    const float scale = rsqrtf(sum / ncols + eps);

    for (int c = threadIdx.x; c < ncols; c += WARP_SIZE) {
        if constexpr (do_gate) {
            dst[c] = (scale * x[c] * w[c]) * ggml_cuda_op_silu_single(gate[o + c]);
        } else {
            dst[c] = scale_out * (scale * x[c]);
        }
    }
}

bool ggml_cuda_fuse_rms_norm_mul_silu_gate(ggml_backend_cuda_context & ctx, ggml_tensor * rms, ggml_tensor * mul,
                                           const ggml_tensor * gate, ggml_tensor * out, const bool check_only) {
    const ggml_tensor * x = rms->src[0];
    const ggml_tensor * w = mul->src[0] == rms ? mul->src[1] : mul->src[0];
    const int64_t ncols = x->ne[0];
    if (x->type != GGML_TYPE_F32 || w->type != GGML_TYPE_F32 || gate->type != GGML_TYPE_F32 || out->type != GGML_TYPE_F32 ||
            x->nb[0] != sizeof(float) || !ggml_is_contiguous(w) || ggml_nelements(w) != ncols || w->ne[0] != ncols ||
            !ggml_is_contiguous(gate) || !ggml_is_contiguous(out) || !ggml_are_same_shape(gate, rms) ||
            !ggml_are_same_shape(out, rms) || ncols >= 1024 || x->ne[1] > INT_MAX || x->ne[2] > 65535 || x->ne[3] > 65535) {
        return false;
    }
    if (check_only) {
        return true;
    }
    float eps;
    memcpy(&eps, rms->op_params, sizeof(float));
    const int64_t fs = sizeof(float);
    const dim3 grid(x->ne[1], x->ne[2], x->ne[3]);
    rms_norm_row_warp_f32<true><<<grid, WARP_SIZE, 0, ctx.stream()>>>(
        (const float *) x->data, (float *) out->data, (int) ncols, x->nb[1]/fs, x->nb[2]/fs, x->nb[3]/fs, eps,
        (const float *) w->data, (const float *) gate->data, 1.0f, nullptr, nullptr);
    CUDA_CHECK(cudaGetLastError());
    return true;
}

bool ggml_cuda_fuse_rms_norm_scale2(ggml_backend_cuda_context & ctx, ggml_tensor * n1, ggml_tensor * s1, ggml_tensor * n2, ggml_tensor * s2) {
    const ggml_tensor * a = n1->src[0], * b = n2->src[0];
    auto ok = [](const ggml_tensor * n, const ggml_tensor * s) {
        float bias;
        memcpy(&bias, (const float *) s->op_params + 1, sizeof(float));
        return n->src[0]->type == GGML_TYPE_F32 && n->type == GGML_TYPE_F32 && s->type == GGML_TYPE_F32 &&
            n->src[0]->nb[0] == sizeof(float) && ggml_is_contiguous(s) && ggml_are_same_shape(n, s) &&
            n->src[0]->ne[0] < 1024 && bias == 0.0f;
    };
    if (!ok(n1, s1) || !ok(n2, s2) || !ggml_are_same_shape(a, b) || a->nb[1] != b->nb[1] || a->nb[2] != b->nb[2] ||
            a->nb[3] != b->nb[3] || memcmp(n1->op_params, n2->op_params, sizeof(float)) != 0 ||
            memcmp(s1->op_params, s2->op_params, sizeof(float)) != 0 || a->ne[1] > INT_MAX || a->ne[2] > 65535 ||
            2*a->ne[3] > 65535 ||
            overlaps(s1, a) || overlaps(s1, b) || overlaps(s2, a) || overlaps(s2, b)) {
        return false;
    }
    float eps, scale;
    memcpy(&eps,   n1->op_params, sizeof(float));
    memcpy(&scale, s1->op_params, sizeof(float));
    const int64_t fs = sizeof(float);
    const dim3 grid(a->ne[1], a->ne[2], 2*a->ne[3]);
    rms_norm_row_warp_f32<false><<<grid, WARP_SIZE, 0, ctx.stream()>>>(
        (const float *) a->data, (float *) s1->data, (int) a->ne[0], a->nb[1]/fs, a->nb[2]/fs, a->nb[3]/fs, eps,
        nullptr, nullptr, scale, (const float *) b->data, (float *) s2->data);
    CUDA_CHECK(cudaGetLastError());
    return true;
}

static __global__ void add_softplus_mul_f32(const float * x, const float * b, const float * m, float * dst,
                                            const int ne0, const int nrows, const int64_t sx, const int64_t sd) {
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= ne0*nrows) {
        return;
    }
    const int r = i / ne0, c = i % ne0;
    const float t = x[r*sx + c] + b[c];
    dst[r*sd + c] = ((t > 20.0f) ? t : logf(1.0f + expf(t))) * m[c]; // unary.cu's op_softplus
}

bool ggml_cuda_fuse_add_softplus_mul(ggml_backend_cuda_context & ctx, ggml_tensor * add, ggml_tensor * sp, ggml_tensor * mul) {
    if (ggml_get_unary_op(sp) != GGML_UNARY_OP_SOFTPLUS) {
        return false;
    }
    const int64_t ne0 = add->ne[0], nrows = ggml_nrows(add);
    auto is_row_vec = [&](const ggml_tensor * t) {
        return t->type == GGML_TYPE_F32 && t->ne[0] == ne0 && ggml_nelements(t) == ne0 && ggml_is_contiguous(t);
    };
    const ggml_tensor * x = add->src[0], * b = add->src[1];
    if (!ggml_are_same_shape(x, add)) {
        std::swap(x, b);
    }
    const ggml_tensor * m = mul->src[0] == sp ? mul->src[1] : mul->src[0];
    // a 2-d view of rows: x and mul contiguous within rows, rows of equal stride
    auto rows2d = [&](const ggml_tensor * t) {
        return t->type == GGML_TYPE_F32 && t->nb[0] == sizeof(float) && ggml_are_same_shape(t, add) &&
            t->nb[2] == t->ne[1]*t->nb[1] && t->nb[3] == t->ne[2]*t->nb[2];
    };
    if (add->type != GGML_TYPE_F32 || !rows2d(x) || !rows2d(mul) || !is_row_vec(b) || !is_row_vec(m) ||
            ne0*nrows > INT_MAX) {
        return false;
    }
    const int n = (int) (ne0*nrows);
    add_softplus_mul_f32<<<(n + 255)/256, 256, 0, ctx.stream()>>>((const float *) x->data, (const float *) b->data,
        (const float *) m->data, (float *) mul->data, (int) ne0, (int) nrows, x->nb[1]/sizeof(float), mul->nb[1]/sizeof(float));
    CUDA_CHECK(cudaGetLastError());
    return true;
}

#define CPY_BATCH_MAX 8

struct cpy_batch_args {
    const char * src[CPY_BATCH_MAX];
    char *       dst[CPY_BATCH_MAX];
    int64_t ne0, ne1, ne2;
    int64_t snb0, snb1, snb2, snb3;
};

static __global__ void k_cpy_batch_f32(const cpy_batch_args a, const int64_t n) {
    const int64_t idx = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }
    const int k = blockIdx.y;
    const int64_t i0 = idx % a.ne0;
    int64_t r = idx / a.ne0;
    const int64_t i1 = r % a.ne1; r /= a.ne1;
    const int64_t i2 = r % a.ne2;
    const int64_t i3 = r / a.ne2;
    const float v = *(const float *) (a.src[k] + i0*a.snb0 + i1*a.snb1 + i2*a.snb2 + i3*a.snb3);
    ((float *) a.dst[k])[idx] = v; // contiguous destination, in the source's element order (ggml_cpy)
}

int ggml_cuda_fuse_cpy_batch(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, const int i) {
    const ggml_tensor * c0 = cgraph->nodes[i];
    const ggml_tensor * s0 = c0->src[0];
    const ggml_tensor * d0 = c0->src[1];
    if (c0->op != GGML_OP_CPY || s0->type != GGML_TYPE_F32 || d0->type != GGML_TYPE_F32 || s0->view_src == nullptr ||
            d0->view_src == nullptr || (c0->flags & GGML_TENSOR_FLAG_OUTPUT) || s0->view_src == d0->view_src) {
        return 0;
    }
    cpy_batch_args a = {};
    int n = 0;
    int last = i;
    for (int j = i; j < cgraph->n_nodes && n < CPY_BATCH_MAX; ++j) {
        const ggml_tensor * c = cgraph->nodes[j];
        if (j > i && (c->op == GGML_OP_VIEW || c->op == GGML_OP_RESHAPE || c->op == GGML_OP_NONE)) {
            continue; // views between the copies
        }
        if (c->op != GGML_OP_CPY || (c->flags & GGML_TENSOR_FLAG_OUTPUT)) {
            break;
        }
        const ggml_tensor * s = c->src[0];
        const ggml_tensor * d = c->src[1];
        if (s->type != GGML_TYPE_F32 || d->type != GGML_TYPE_F32 || s->view_src != s0->view_src || d->view_src != d0->view_src ||
                !ggml_are_same_shape(s, s0) || !ggml_are_same_shape(d, d0) || !ggml_is_contiguous(d) ||
                ggml_nelements(d) != ggml_nelements(s) ||
                memcmp(s->nb, s0->nb, sizeof(s->nb)) != 0 || memcmp(d->nb, d0->nb, sizeof(d->nb)) != 0) {
            break;
        }
        a.src[n] = (const char *) s->data;
        a.dst[n] = (char *) d->data;
        ++n;
        last = j;
    }
    if (n < 2) {
        return 0;
    }
    a.ne0 = s0->ne[0]; a.ne1 = s0->ne[1]; a.ne2 = s0->ne[2];
    a.snb0 = s0->nb[0]; a.snb1 = s0->nb[1]; a.snb2 = s0->nb[2]; a.snb3 = s0->nb[3];
    const int64_t ne = ggml_nelements(s0);
    const dim3 grid((unsigned) ((ne + 255)/256), n);
    k_cpy_batch_f32<<<grid, 256, 0, ctx.stream()>>>(a, ne);
    CUDA_CHECK(cudaGetLastError());
    return last - i;
}
