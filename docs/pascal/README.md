# llama.cpp-Pascal

A llama.cpp fork tuned for the **Tesla P100 (sm_60)**. The work targets
Qwen3.8-27B on two P100 16 GB cards with `-sm tensor`, plus a 1-card
config. The P100 has fast FP16 (2:1) but no `dp4a` and no tensor cores, so
mainline's int8 decode kernels run ALU-bound on it. Most of the gains below
come from that.

Benchmarks, methodology and every A/B are in
[p100-inference-benchmarks](https://github.com/Tom1tk/p100-inference-benchmarks).
Design notes and per-lever results are in [fork-plan-2026-10-14.md](fork-plan-2026-10-14.md).

## Results (2 × P100, 200 W cap, Qwen3.8-27B UD-Q4_K_M, MTP drafter)

| | mainline | this fork |
|---|---|---|
| decode, 16k context | 28.7 t/s | **47.5 t/s** (+66%) |
| prefill, 16k | 318 t/s | **392 t/s** (+23%) |
| decode, 100k | 20.9 t/s | **33.1 t/s** (+58%) |
| prefill, 100k | 214 t/s | **288 t/s** (+34%) |

100k needle-in-a-haystack: 19/19. The Q5_K_M FP32 reference gets 18/19.

## Build

```
cmake -B build-cuda-p100 -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=60 -DCMAKE_BUILD_TYPE=Release
cmake --build build-cuda-p100 -j4 --target llama-server
```

## Serve (2 cards, 100k context)

```
GGML_CUDA_ALLREDUCE=none GGML_CUDA_P2P=1 \
llama-server -m Qwen3.8-27B-UD-Q4_K_M.gguf \
  --spec-type draft-mtp -md mtp-Qwen3.8-27B-Q4_0-head96k.gguf \
  -ngl 99 -fa 1 -t 8 -ctk f16 -ctv f16 -b 2048 -ub 2048 -c 100000 -np 1 -sm tensor
```

1 card (UD-IQ3_S, 16k): `-sm none`, no drafter, plus `GGML_PASCAL_DEQUANT_STAGED=1`.

The trimmed drafter is generated from the stock MTP GGUF:
`python3 scripts/pascal-trim-mtp-head.py mtp-Qwen3.8-27B-Q4_0.gguf mtp-Qwen3.8-27B-Q4_0-head96k.gguf 98304`.

## Levers

On by default when the device is sm_60. Set `NAME=0` to turn one off, or
`NAME=1` to force it on for another Pascal card (untested there).

| Env var | What it does | Bench |
|---|---|---|
| `GGML_CUDA_PASCAL_MMVH` | Half2 matvec for decode: weight bytes go straight into `__hfma2` as fp16 subnormals instead of emulated int8 dot products. Q4_K, Q5_K, Q6_K, IQ4_XS, IQ3_S, IQ3_XXS | H28, H29, H34 |
| `GGML_PASCAL_MMVH_LOWBIT` | Adds Q3_K, Q2_K, IQ4_NL and the IQ2 family to mmvh | H42, H44 |
| `GGML_PASCAL_MMVH_Q6K_WIDE` | Q6_K with 4 lanes per superblock instead of 8 | H41b |
| `GGML_CUDA_PASCAL_GEMM_ALGO` | cuBLAS `ALGO3` for FP16 GEMMs at n ≥ 512 (12–17 vs 5–7 TFLOPS) | H33, H35 |
| `GGML_PASCAL_FA_GEMM` | Prefill flash attention through cuBLAS GEMMs at q ≥ 512, kv ≥ 4096 (`_MIN_Q`, `_MIN_KV` to tune) | H46 |
| `GGML_PASCAL_DUPLEX_REDUCE` | The two directions of an AllReduce pair overlap on PCIe instead of serialising | H36 |
| `GGML_PASCAL_F16_REDUCE` | Prefill-sized AllReduce partials cross PCIe as F16. Meta backend, so on fork-wide | H31 |

Opt-in only:

| Env var / option | What it does | Bench |
|---|---|---|
| `GGML_PASCAL_DEQUANT_STAGED=1` | 256-thread staged dequant to F16, bit-identical. +3.9% 1-card prefill; noise on 2 cards | H39r |
| `-DGGML_CUDA_PASCAL_SM60_FP32=ON` | Build option: excludes sm_60 from the FP16 fast path. Slower; used only to build the FP32 quality reference | H17, H30 |

Measured and removed (tag `pre-tidy-2026-10-16` has the code): CUDA graphs on
Pascal (H27), internal AllReduce (H37), half2 tile Q·K (H38), parallel host
dispatch (H40), fp16 GEMM Q·K (H47), mmvh 4 rows/warp at 5–8 columns (H55),
GQA-6 tile FA packing (H54).

## Keeping up with mainline

The fork is rebased onto `ggml-org/llama.cpp` master, never merged, so the
Pascal commits stay a short readable stack on top. After each rebase:
`test-backend-ops -o MUL_MAT,FLASH_ATTN_EXT`, then a 3-rep 16k serve check
against the banked control. Output must be identical.
