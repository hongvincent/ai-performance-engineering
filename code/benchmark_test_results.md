# Benchmark Test Results Summary

**Generated:** 2025-11-05 20:14:38

## Overall Summary

- **Chapters tested:** 20/20
- **Chapters skipped:** 0 (CUDA unavailable)
- **Chapters with no benchmarks:** 0
- **Total benchmarks:** 257
- **Successful:** 194
- **Failed:** 63
- **Average speedup:** 183.44x
- **Best speedup:** 21822.68x
- **Worst speedup:** 1.00x

## Per-Chapter Summary

| Chapter | Status | Benchmarks | Successful | Failed | Avg Speedup | Max Speedup |
|---------|--------|------------|------------|--------|-------------|-------------|
| ch1 | PASS | 19 | 19 | 0 | 1.43x | 1.82x |
| ch10 | PASS | 16 | 10 | 6 | 122.74x | 837.93x |
| ch11 | PASS | 10 | 8 | 2 | 2.27x | 3.90x |
| ch12 | PASS | 22 | 17 | 5 | 51.98x | 378.71x |
| ch13 | PASS | 17 | 13 | 4 | 1457.99x | 10085.49x |
| ch14 | PASS | 4 | 0 | 4 | 1.00x | 1.00x |
| ch15 | PASS | 7 | 6 | 1 | 39.44x | 171.74x |
| ch16 | PASS | 8 | 5 | 3 | 3.94x | 6.13x |
| ch17 | PASS | 8 | 4 | 4 | 100.65x | 297.26x |
| ch18 | PASS | 11 | 9 | 2 | 4.81x | 11.32x |
| ch19 | PASS | 8 | 7 | 1 | 30.69x | 146.57x |
| ch2 | PASS | 7 | 4 | 3 | 29.27x | 77.69x |
| ch20 | PASS | 14 | 11 | 3 | 22.45x | 106.17x |
| ch3 | PASS | 11 | 8 | 3 | 2.42x | 3.41x |
| ch4 | PASS | 6 | 3 | 3 | 7294.05x | 21822.68x |
| ch5 | PASS | 7 | 7 | 0 | 76.65x | 299.38x |
| ch6 | PASS | 19 | 2 | 17 | 1.43x | 1.48x |
| ch7 | PASS | 29 | 28 | 1 | 18.71x | 124.51x |
| ch8 | PASS | 15 | 15 | 0 | 29.89x | 169.95x |
| ch9 | PASS | 19 | 18 | 1 | 5.53x | 45.61x |

## Detailed Results

### CH1

**nvlink**
- Baseline: `baseline_nvlink.py` (1.32 ms)
- `optimized_nvlink.py`: 0.80 ms (1.64x speedup)
- Best speedup: 1.64x

**double**
- Baseline: `baseline_double_buffering.py` (4.13 ms)
- `optimized_double_buffering.py`: 6.40 ms (0.65x speedup)

**continuous**
- Baseline: `baseline_continuous_batching.py` (0.30 ms)
- `optimized_continuous_batching.py`: 0.86 ms (0.35x speedup)

**cutlass**
- Baseline: `baseline_cutlass.py` (0.10 ms)
- `optimized_cutlass.py`: 0.14 ms (0.72x speedup)

**kv**
- Baseline: `baseline_kv_cache.py` (6.17 ms)
- `optimized_kv_cache.py`: 7.23 ms (0.85x speedup)

**speculative**
- Baseline: `baseline_speculative_decoding.py` (0.77 ms)
- `optimized_speculative_decoding.py`: 0.79 ms (0.97x speedup)

**moe**
- Baseline: `baseline_moe.py` (0.21 ms)
- `optimized_moe.py`: 0.62 ms (0.33x speedup)

**shared**
- Baseline: `baseline_shared_memory.py` (0.05 ms)
- `optimized_shared_memory.py`: 0.07 ms (0.76x speedup)

**coalescing**
- Baseline: `baseline_coalescing.py` (0.15 ms)
- `optimized_coalescing.py`: 0.42 ms (0.35x speedup)

**warp**
- Baseline: `baseline_warp_specialization.py` (0.05 ms)
- `optimized_warp_divergence.py`: 0.15 ms (0.34x speedup)
- `optimized_warp_specialization.py`: 0.05 ms (1.05x speedup)
- Best speedup: 1.05x

**ilp**
- Baseline: `baseline_ilp_basic.py` (0.14 ms)
- `optimized_ilp_basic.py`: 0.23 ms (0.62x speedup)

**warp**
- Baseline: `baseline_warp_divergence.py` (0.08 ms)
- `optimized_warp_divergence.py`: 0.11 ms (0.69x speedup)
- `optimized_warp_specialization.py`: 0.07 ms (1.12x speedup)
- Best speedup: 1.12x

**performance**
- Baseline: `baseline_performance.py` (0.81 ms)
- `optimized_performance_pinned.py`: 1.37 ms (0.59x speedup)
- `optimized_performance_batch.py`: 0.47 ms (1.71x speedup)
- `optimized_performance_graphs.py`: 0.52 ms (1.55x speedup)
- Best speedup: 1.71x

**disaggregated**
- Baseline: `baseline_disaggregated.py` (0.06 ms)
- `optimized_disaggregated.py`: 0.06 ms (0.89x speedup)

**nccl**
- Baseline: `baseline_nccl.py` (0.01 ms)
- `optimized_nccl.py`: 0.06 ms (0.24x speedup)

**guided**
- Baseline: `baseline_guided_decoding.py` (0.20 ms)
- `optimized_guided_decoding.py`: 0.77 ms (0.26x speedup)

**attention**
- Baseline: `baseline_attention.py` (0.70 ms)
- `optimized_attention.py`: 0.39 ms (1.82x speedup)
- Best speedup: 1.82x

**gemm**
- Baseline: `baseline_gemm.cu` (487.57 ms)
- `optimized_gemm_batched.cu`: 496.12 ms (0.98x speedup)
- `optimized_gemm_strided.cu`: 392.38 ms (1.24x speedup)
- Best speedup: 1.24x

**arithmetic**
- Baseline: `baseline_arithmetic_intensity.cu` (569.75 ms)
- `optimized_arithmetic_intensity_combined.cu`: 700.85 ms (0.81x speedup)


### CH10

**flash**
- Baseline: `baseline_flash_attention.py` (1.05 ms)
- `optimized_flash_attention.py`: 0.94 ms (1.12x speedup)
- Best speedup: 1.12x

**streams**
- Baseline: `baseline_streams.py` (4.74 ms)
- `optimized_streams.py`: 2.70 ms (1.76x speedup)
- Best speedup: 1.76x

**continuous**
- Baseline: `baseline_continuous_batching.py` (24.89 ms)
- `optimized_continuous_batching.py`: 0.03 ms (837.93x speedup)
- Best speedup: 837.93x

**cluster**
- Baseline: `baseline_cluster_group.py`
- Failed: Baseline execution failed: Benchmark failed: Benchmark execution failed: Failed to build baseline_cluster_group_sm121 (arch=sm_121).
stdout:

stderr:
make: *** No rule to make target 'baseline_cluster_group_sm121'.  Stop.


**roofline**
- Baseline: `baseline_roofline.py` (0.12 ms)
- `optimized_roofline.py`: 1.14 ms (0.11x speedup)

**warp**
- Baseline: `baseline_warp_divergence.py` (0.09 ms)
- `optimized_warp_divergence.py`: 0.03 ms (2.95x speedup)
- Best speedup: 2.95x

**triton**
- Baseline: `baseline_triton.py` (0.08 ms)
- `optimized_triton.py`: 0.09 ms (0.92x speedup)

**matmul**
- Baseline: `baseline_matmul.py` (49.60 ms)
- `optimized_matmul_tensor_cores.py`: 22.58 ms (2.20x speedup)
- Best speedup: 2.20x

**batch**
- Baseline: `baseline_batch.py` (0.11 ms)
- `optimized_batch.py`: 0.19 ms (0.59x speedup)

**nccl**
- Baseline: `baseline_nccl.py` (1.10 ms)
- `optimized_nccl.py`: 0.11 ms (10.37x speedup)
- Best speedup: 10.37x

**cooperative**
- Baseline: `baseline_cooperative_persistent.py`
- Failed: Baseline execution failed: Benchmark failed: Benchmark execution failed: Failed to build baseline_cooperative_persistent_sm121 (arch=sm_121).
stdout:

stderr:
make: *** No rule to make target 'baseline_cooperative_persistent_sm121'.  Stop.


**double**
- Baseline: `baseline_double_buffered_pipeline.py`
- Failed: Baseline execution failed: Benchmark failed: Benchmark execution failed: Failed to build baseline_double_buffered_pipeline_sm121 (arch=sm_121).
stdout:

stderr:
make: *** No rule to make target 'baseline_double_buffered_pipeline_sm121'.  Stop.


**attention**
- Baseline: `baseline_attention.py` (0.32 ms)
- `optimized_attention.py`: 0.11 ms (2.87x speedup)
- Best speedup: 2.87x

**cluster**
- Baseline: `baseline_cluster_group.cu`
- Failed: Baseline executable not found for baseline_cluster_group.cu

**double**
- Baseline: `baseline_double_buffered_pipeline.cu`
- Failed: Baseline executable not found for baseline_double_buffered_pipeline.cu

**cooperative**
- Baseline: `baseline_cooperative_persistent.cu`
- Failed: Baseline executable not found for baseline_cooperative_persistent.cu


### CH11

**coalescing**
- Baseline: `baseline_coalescing_streams.py` (0.03 ms)
- `optimized_coalescing_streams.py`: 0.11 ms (0.22x speedup)

**streams**
- Baseline: `baseline_streams.py` (3.36 ms)
- `optimized_streams.py`: 0.86 ms (3.90x speedup)
- Best speedup: 3.90x

**distributed**
- Baseline: `baseline_distributed_streams.py` (0.03 ms)
- `optimized_distributed_streams.py`: 0.07 ms (0.41x speedup)

**tiling**
- Baseline: `baseline_tiling_streams.py` (0.03 ms)
- `optimized_tiling_streams.py`: 0.87 ms (0.03x speedup)

**adaptive**
- Baseline: `baseline_adaptive_streams.py` (0.09 ms)
- `optimized_adaptive_streams.py`: 0.04 ms (2.22x speedup)
- Best speedup: 2.22x

**gemm**
- Baseline: `baseline_gemm_streams.py` (0.00 ms)
- Failed: None

**disaggregated**
- Baseline: `baseline_disaggregated_streams.py` (0.05 ms)
- `optimized_disaggregated_streams.py`: 0.04 ms (1.30x speedup)
- Best speedup: 1.30x

**tensor**
- Baseline: `baseline_tensor_cores_streams.py` (0.00 ms)
- `optimized_tensor_cores_streams.py`: 0.00 ms (2.62x speedup)
- Best speedup: 2.62x

**stream**
- Baseline: `baseline_stream_ordered_kv_cache.py`
- Failed: Baseline execution failed: Benchmark failed: Benchmark execution failed: name 'batch_size' is not defined

**streams**
- Baseline: `baseline_streams.cu` (319.27 ms)
- `optimized_streams_ordered.cu`: 242.47 ms (1.32x speedup)
- `optimized_streams_warp_specialized.cu`: 415.90 ms (0.77x speedup)
- Best speedup: 1.32x


### CH12

**kernel**
- Baseline: `baseline_kernel_launches.py` (28.65 ms)
- `optimized_kernel_fusion.py`: 0.08 ms (378.71x speedup)
- `optimized_kernel_launches_graphs.py`: 19.74 ms (1.45x speedup)
- Best speedup: 378.71x

**nvlink**
- Baseline: `baseline_nvlink.py` (89.76 ms)
- `optimized_nvlink.py`: 3.06 ms (29.36x speedup)
- Best speedup: 29.36x

**continuous**
- Baseline: `baseline_continuous_batching.py` (5.21 ms)
- `optimized_continuous_batching.py`: 0.02 ms (332.41x speedup)
- Best speedup: 332.41x

**cuda**
- Baseline: `baseline_cuda_graphs.py` (0.12 ms)
- `optimized_cuda_graphs.py`: 0.56 ms (0.22x speedup)

**distributed**
- Baseline: `baseline_distributed.py` (0.18 ms)
- `optimized_distributed.py`: 0.04 ms (4.69x speedup)
- Best speedup: 4.69x

**quantization**
- Baseline: `baseline_quantization.py` (0.15 ms)
- `optimized_quantization.py`: 0.05 ms (3.23x speedup)
- Best speedup: 3.23x

**roofline**
- Baseline: `baseline_roofline.py` (0.08 ms)
- `optimized_roofline.py`: 0.10 ms (0.80x speedup)

**graph**
- Baseline: `baseline_graph_bandwidth.py` (36.03 ms)
- `optimized_graph_bandwidth.py`: 35.98 ms (1.00x speedup)
- Best speedup: 1.00x

**work**
- Baseline: `baseline_work_queue.py`
- Failed: Baseline execution failed: Benchmark failed: Benchmark execution failed: Failed to load work_queue CUDA extension: /home/cfregly/ai-performance-engineering/code/ch12/cuda_extensions/build/work_queue_kernels.so: undefined symbol: _ZNK2at10TensorBase8data_ptrIKfEEPT_v

**kernel**
- Baseline: `baseline_kernel_fusion.py` (0.12 ms)
- `optimized_kernel_fusion.py`: 0.04 ms (2.93x speedup)
- `optimized_kernel_launches_graphs.py`: 19.10 ms (0.01x speedup)
- Best speedup: 2.93x

**nccl**
- Baseline: `baseline_nccl.py` (1.03 ms)
- `optimized_nccl.py`: 0.11 ms (9.32x speedup)
- Best speedup: 9.32x

**hbm**
- Baseline: `baseline_hbm.py` (0.11 ms)
- `optimized_hbm.py`: 0.18 ms (0.62x speedup)

**attention**
- Baseline: `baseline_attention.py` (0.19 ms)
- `optimized_attention.py`: 0.22 ms (0.86x speedup)

**bank**
- Baseline: `baseline_bank_conflicts.py` (0.11 ms)
- `optimized_bank_conflicts.py`: 0.14 ms (0.78x speedup)

**cuda**
- Baseline: `baseline_cuda_graphs_conditional_enhanced.cu` (1315.32 ms)
- `optimized_cuda_graphs_conditional.cu`: 290.85 ms (4.52x speedup)
- `optimized_cuda_graphs.cu`: 265.38 ms (4.96x speedup)
- `optimized_cuda_graphs_conditional_enhanced.cu`: 349.36 ms (3.76x speedup)
- Best speedup: 4.96x

**cuda**
- Baseline: `baseline_cuda_graphs_conditional.cu` (984.32 ms)
- `optimized_cuda_graphs_conditional.cu`: 800.21 ms (1.23x speedup)
- `optimized_cuda_graphs.cu`: 274.46 ms (3.59x speedup)
- `optimized_cuda_graphs_conditional_enhanced.cu`: 550.98 ms (1.79x speedup)
- Best speedup: 3.59x

**cuda**
- Baseline: `baseline_cuda_graphs.cu` (664.87 ms)
- `optimized_cuda_graphs_conditional.cu`: 575.22 ms (1.16x speedup)
- `optimized_cuda_graphs.cu`: 469.18 ms (1.42x speedup)
- `optimized_cuda_graphs_conditional_enhanced.cu`: 705.40 ms (0.94x speedup)
- Best speedup: 1.42x

**uneven**
- Baseline: `baseline_uneven_partition.cu`
- Failed: Baseline executable not found for baseline_uneven_partition.cu

**dynamic**
- Baseline: `baseline_dynamic_parallelism.cu`
- Failed: Baseline executable not found for baseline_dynamic_parallelism.cu

**graph**
- Baseline: `baseline_graph_bandwidth.cu`
- Failed: Baseline executable not found for baseline_graph_bandwidth.cu

**work**
- Baseline: `baseline_work_queue.cu` (502.69 ms)
- `optimized_work_queue.cu`: 445.02 ms (1.13x speedup)
- Best speedup: 1.13x

**kernel**
- Baseline: `baseline_kernel_fusion.cu`
- Failed: Baseline executable not found for baseline_kernel_fusion.cu


### CH13

**occupancy**
- Baseline: `baseline_occupancy.py` (12.51 ms)
- `optimized_occupancy.py`: 0.93 ms (13.50x speedup)
- Best speedup: 13.50x

**dataloader**
- Baseline: `baseline_dataloader_default.py` (3.61 ms)
- `optimized_dataloader_tuned.py`: 7.37 ms (0.49x speedup)

**continuous**
- Baseline: `baseline_continuous_batching.py` (117.77 ms)
- `optimized_continuous_batching.py`: 0.01 ms (10085.49x speedup)
- Best speedup: 10085.49x

**paged**
- Baseline: `baseline_paged_attention.py`
- Failed: Baseline execution failed: Benchmark failed: Benchmark execution failed: not enough values to unpack (expected 3, got 2)

**autograd**
- Baseline: `baseline_autograd_standard.py` (1.13 ms)
- Failed: None

**bandwidth**
- Baseline: `baseline_bandwidth_naive.py` (87.40 ms)
- `optimized_bandwidth_coalesced.py`: 1.00 ms (87.73x speedup)
- Best speedup: 87.73x

**precision**
- Baseline: `baseline_precision_bf16.py` (29.62 ms)
- `optimized_precision_mixed.py`: 2.90 ms (10.22x speedup)
- `optimized_precision_fp8.py`: 11.64 ms (2.54x speedup)
- Best speedup: 10.22x

**shared**
- Baseline: `baseline_shared_memory.py` (0.59 ms)
- `optimized_shared_memory.py`: 0.10 ms (5.67x speedup)
- Best speedup: 5.67x

**training**
- Baseline: `baseline_training_standard.py` (90.79 ms)
- `optimized_training_checkpoint.py`: 109.53 ms (0.83x speedup)

**arithmetic**
- Baseline: `baseline_arithmetic_intensity.py` (1.33 ms)
- `optimized_arithmetic_intensity.py`: 16.63 ms (0.08x speedup)

**warp**
- Baseline: `baseline_warp_specialization.py` (0.05 ms)
- `optimized_warp_specialization.py`: 0.05 ms (0.97x speedup)

**memory**
- Baseline: `baseline_memory_profiling.py` (2.22 ms)
- `optimized_memory_profiling.py`: 1.64 ms (1.36x speedup)
- Best speedup: 1.36x

**tiling**
- Baseline: `baseline_tiling.py` (0.06 ms)
- Failed: None

**precision**
- Baseline: `baseline_precision_fp32.py` (1.28 ms)
- `optimized_precision_mixed.py`: 2.30 ms (0.56x speedup)
- `optimized_precision_fp8.py`: 11.58 ms (0.11x speedup)

**attention**
- Baseline: `baseline_attention_standard.py` (0.61 ms)
- `optimized_attention_flex.py`: 0.31 ms (1.95x speedup)
- Best speedup: 1.95x

**kv**
- Baseline: `baseline_kv_cache_naive.py` (21.87 ms)
- `optimized_kv_cache.py`: 29.74 ms (0.74x speedup)

**matmul**
- Baseline: `baseline_matmul_pytorch.py` (0.36 ms)
- Failed: None


### CH14

**roofline**
- Baseline: `baseline_roofline_quantization.py`
- Failed: Baseline execution failed: Benchmark failed: Benchmark execution failed: Could not run 'quantized::linear_dynamic' with arguments from the 'CUDA' backend. This could be because the operator doesn't exist for this backend, or was omitted during the selective/custom build process (if using custom build). If you are a Facebook employee using PyTorch on mobile, please visit https://fburl.com/ptmfixes for possible resolutions. 'quantized::linear_dynamic' is only available for these backends: [CPU, Meta, BackendSelect, Python, FuncTorchDynamicLayerBackMode, Functionalize, Named, Conjugate, Negative, ZeroTensor, ADInplaceOrView, AutogradOther, AutogradCPU, AutogradCUDA, AutogradXLA, AutogradMPS, AutogradXPU, AutogradHPU, AutogradLazy, AutogradMTIA, AutogradMAIA, AutogradMeta, Tracer, AutocastCPU, AutocastMTIA, AutocastMAIA, AutocastXPU, AutocastMPS, AutocastCUDA, FuncTorchBatched, BatchedNestedTensor, FuncTorchVmapMode, Batched, VmapMode, FuncTorchGradWrapper, PythonTLSSnapshot, FuncTorchDynamicLayerFrontMode, PreDispatch, PythonDispatcher].

CPU: registered at /pytorch/aten/src/ATen/native/quantized/cpu/qlinear_dynamic.cpp:1027 [kernel]
Meta: registered at /pytorch/aten/src/ATen/core/MetaFallbackKernel.cpp:23 [backend fallback]
BackendSelect: fallthrough registered at /pytorch/aten/src/ATen/core/BackendSelectFallbackKernel.cpp:3 [backend fallback]
Python: registered at /pytorch/aten/src/ATen/core/PythonFallbackKernel.cpp:194 [backend fallback]
FuncTorchDynamicLayerBackMode: registered at /pytorch/aten/src/ATen/functorch/DynamicLayer.cpp:479 [backend fallback]
Functionalize: registered at /pytorch/aten/src/ATen/FunctionalizeFallbackKernel.cpp:387 [backend fallback]
Named: registered at /pytorch/aten/src/ATen/core/NamedRegistrations.cpp:7 [backend fallback]
Conjugate: registered at /pytorch/aten/src/ATen/ConjugateFallback.cpp:17 [backend fallback]
Negative: registered at /pytorch/aten/src/ATen/native/NegateFallback.cpp:18 [backend fallback]
ZeroTensor: registered at /pytorch/aten/src/ATen/ZeroTensorFallback.cpp:115 [backend fallback]
ADInplaceOrView: fallthrough registered at /pytorch/aten/src/ATen/core/VariableFallbackKernel.cpp:104 [backend fallback]
AutogradOther: registered at /pytorch/aten/src/ATen/core/VariableFallbackKernel.cpp:63 [backend fallback]
AutogradCPU: registered at /pytorch/aten/src/ATen/core/VariableFallbackKernel.cpp:67 [backend fallback]
AutogradCUDA: registered at /pytorch/aten/src/ATen/core/VariableFallbackKernel.cpp:75 [backend fallback]
AutogradXLA: registered at /pytorch/aten/src/ATen/core/VariableFallbackKernel.cpp:87 [backend fallback]
AutogradMPS: registered at /pytorch/aten/src/ATen/core/VariableFallbackKernel.cpp:95 [backend fallback]
AutogradXPU: registered at /pytorch/aten/src/ATen/core/VariableFallbackKernel.cpp:71 [backend fallback]
AutogradHPU: registered at /pytorch/aten/src/ATen/core/VariableFallbackKernel.cpp:108 [backend fallback]
AutogradLazy: registered at /pytorch/aten/src/ATen/core/VariableFallbackKernel.cpp:91 [backend fallback]
AutogradMTIA: registered at /pytorch/aten/src/ATen/core/VariableFallbackKernel.cpp:79 [backend fallback]
AutogradMAIA: registered at /pytorch/aten/src/ATen/core/VariableFallbackKernel.cpp:83 [backend fallback]
AutogradMeta: registered at /pytorch/aten/src/ATen/core/VariableFallbackKernel.cpp:99 [backend fallback]
Tracer: registered at /pytorch/torch/csrc/autograd/TraceTypeManual.cpp:294 [backend fallback]
AutocastCPU: fallthrough registered at /pytorch/aten/src/ATen/autocast_mode.cpp:324 [backend fallback]
AutocastMTIA: fallthrough registered at /pytorch/aten/src/ATen/autocast_mode.cpp:468 [backend fallback]
AutocastMAIA: fallthrough registered at /pytorch/aten/src/ATen/autocast_mode.cpp:506 [backend fallback]
AutocastXPU: fallthrough registered at /pytorch/aten/src/ATen/autocast_mode.cpp:544 [backend fallback]
AutocastMPS: fallthrough registered at /pytorch/aten/src/ATen/autocast_mode.cpp:209 [backend fallback]
AutocastCUDA: fallthrough registered at /pytorch/aten/src/ATen/autocast_mode.cpp:165 [backend fallback]
FuncTorchBatched: registered at /pytorch/aten/src/ATen/functorch/LegacyBatchingRegistrations.cpp:731 [backend fallback]
BatchedNestedTensor: registered at /pytorch/aten/src/ATen/functorch/LegacyBatchingRegistrations.cpp:758 [backend fallback]
FuncTorchVmapMode: fallthrough registered at /pytorch/aten/src/ATen/functorch/VmapModeRegistrations.cpp:27 [backend fallback]
Batched: registered at /pytorch/aten/src/ATen/LegacyBatchingRegistrations.cpp:1075 [backend fallback]
VmapMode: fallthrough registered at /pytorch/aten/src/ATen/VmapModeRegistrations.cpp:33 [backend fallback]
FuncTorchGradWrapper: registered at /pytorch/aten/src/ATen/functorch/TensorWrapper.cpp:210 [backend fallback]
PythonTLSSnapshot: registered at /pytorch/aten/src/ATen/core/PythonFallbackKernel.cpp:202 [backend fallback]
FuncTorchDynamicLayerFrontMode: registered at /pytorch/aten/src/ATen/functorch/DynamicLayer.cpp:475 [backend fallback]
PreDispatch: registered at /pytorch/aten/src/ATen/core/PythonFallbackKernel.cpp:206 [backend fallback]
PythonDispatcher: registered at /pytorch/aten/src/ATen/core/PythonFallbackKernel.cpp:198 [backend fallback]

Exception raised from reportError at /pytorch/aten/src/ATen/core/dispatch/OperatorEntry.cpp:650 (most recent call first):
frame #0: c10::Error::Error(c10::SourceLocation, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >) + 0xb0 (0xeeeb464ac700 in /home/cfregly/.local/lib/python3.11/site-packages/torch/lib/libc10.so)
frame #1: c10::impl::OperatorEntry::reportError(c10::DispatchKey) const + 0x448 (0xeeeb630d2ed8 in /home/cfregly/.local/lib/python3.11/site-packages/torch/lib/libtorch_cpu.so)
frame #2: <unknown function> + 0x1348c28 (0xeeeb62e58c28 in /home/cfregly/.local/lib/python3.11/site-packages/torch/lib/libtorch_cpu.so)
frame #3: <unknown function> + 0x54dfd24 (0xeeeb66fefd24 in /home/cfregly/.local/lib/python3.11/site-packages/torch/lib/libtorch_cpu.so)
frame #4: <unknown function> + 0x5c4c7fc (0xeeeb6775c7fc in /home/cfregly/.local/lib/python3.11/site-packages/torch/lib/libtorch_cpu.so)
frame #5: <unknown function> + 0xd4b67c (0xeeeb6c85b67c in /home/cfregly/.local/lib/python3.11/site-packages/torch/lib/libtorch_python.so)
frame #6: <unknown function> + 0xd4ba14 (0xeeeb6c85ba14 in /home/cfregly/.local/lib/python3.11/site-packages/torch/lib/libtorch_python.so)
frame #7: torch::jit::_get_operation_for_overload_or_packet(std::vector<std::shared_ptr<torch::jit::Operator>, std::allocator<std::shared_ptr<torch::jit::Operator> > > const&, c10::Symbol, pybind11::args const&, pybind11::kwargs const&, bool, std::optional<c10::DispatchKey>) + 0x3c (0xeeeb6c85baac in /home/cfregly/.local/lib/python3.11/site-packages/torch/lib/libtorch_python.so)
frame #8: <unknown function> + 0xc434e0 (0xeeeb6c7534e0 in /home/cfregly/.local/lib/python3.11/site-packages/torch/lib/libtorch_python.so)
frame #9: <unknown function> + 0x5d6d4c (0xeeeb6c0e6d4c in /home/cfregly/.local/lib/python3.11/site-packages/torch/lib/libtorch_python.so)
frame #10: python() [0x4d9630]
<omitting python frames>
frame #16: python() [0x616798]
frame #19: python() [0x514354]
frame #21: python() [0x514354]
frame #26: python() [0x616798]
frame #29: python() [0x514354]
frame #31: python() [0x514354]
frame #36: python() [0x616798]
frame #41: python() [0x514c68]
frame #42: python() [0x514548]
frame #43: python() [0x64013c]
frame #44: python() [0x5f6de4]
frame #45: <unknown function> + 0x8595c (0xeeeb7ae3595c in /lib/aarch64-linux-gnu/libc.so.6)
frame #46: <unknown function> + 0xebb0c (0xeeeb7ae9bb0c in /lib/aarch64-linux-gnu/libc.so.6)


**flex**
- Baseline: `baseline_flex_attention.py` (0.07 ms)
- Failed: None

**model**
- Baseline: `baseline_model_eager.py` (60.16 ms)
- Failed: None

**nccl**
- Baseline: `baseline_nccl_quantization.py`
- Failed: Baseline execution failed: Benchmark failed: Benchmark execution failed: Could not run 'quantized::linear_dynamic' with arguments from the 'CUDA' backend. This could be because the operator doesn't exist for this backend, or was omitted during the selective/custom build process (if using custom build). If you are a Facebook employee using PyTorch on mobile, please visit https://fburl.com/ptmfixes for possible resolutions. 'quantized::linear_dynamic' is only available for these backends: [CPU, Meta, BackendSelect, Python, FuncTorchDynamicLayerBackMode, Functionalize, Named, Conjugate, Negative, ZeroTensor, ADInplaceOrView, AutogradOther, AutogradCPU, AutogradCUDA, AutogradXLA, AutogradMPS, AutogradXPU, AutogradHPU, AutogradLazy, AutogradMTIA, AutogradMAIA, AutogradMeta, Tracer, AutocastCPU, AutocastMTIA, AutocastMAIA, AutocastXPU, AutocastMPS, AutocastCUDA, FuncTorchBatched, BatchedNestedTensor, FuncTorchVmapMode, Batched, VmapMode, FuncTorchGradWrapper, PythonTLSSnapshot, FuncTorchDynamicLayerFrontMode, PreDispatch, PythonDispatcher].

CPU: registered at /pytorch/aten/src/ATen/native/quantized/cpu/qlinear_dynamic.cpp:1027 [kernel]
Meta: registered at /pytorch/aten/src/ATen/core/MetaFallbackKernel.cpp:23 [backend fallback]
BackendSelect: fallthrough registered at /pytorch/aten/src/ATen/core/BackendSelectFallbackKernel.cpp:3 [backend fallback]
Python: registered at /pytorch/aten/src/ATen/core/PythonFallbackKernel.cpp:194 [backend fallback]
FuncTorchDynamicLayerBackMode: registered at /pytorch/aten/src/ATen/functorch/DynamicLayer.cpp:479 [backend fallback]
Functionalize: registered at /pytorch/aten/src/ATen/FunctionalizeFallbackKernel.cpp:387 [backend fallback]
Named: registered at /pytorch/aten/src/ATen/core/NamedRegistrations.cpp:7 [backend fallback]
Conjugate: registered at /pytorch/aten/src/ATen/ConjugateFallback.cpp:17 [backend fallback]
Negative: registered at /pytorch/aten/src/ATen/native/NegateFallback.cpp:18 [backend fallback]
ZeroTensor: registered at /pytorch/aten/src/ATen/ZeroTensorFallback.cpp:115 [backend fallback]
ADInplaceOrView: fallthrough registered at /pytorch/aten/src/ATen/core/VariableFallbackKernel.cpp:104 [backend fallback]
AutogradOther: registered at /pytorch/aten/src/ATen/core/VariableFallbackKernel.cpp:63 [backend fallback]
AutogradCPU: registered at /pytorch/aten/src/ATen/core/VariableFallbackKernel.cpp:67 [backend fallback]
AutogradCUDA: registered at /pytorch/aten/src/ATen/core/VariableFallbackKernel.cpp:75 [backend fallback]
AutogradXLA: registered at /pytorch/aten/src/ATen/core/VariableFallbackKernel.cpp:87 [backend fallback]
AutogradMPS: registered at /pytorch/aten/src/ATen/core/VariableFallbackKernel.cpp:95 [backend fallback]
AutogradXPU: registered at /pytorch/aten/src/ATen/core/VariableFallbackKernel.cpp:71 [backend fallback]
AutogradHPU: registered at /pytorch/aten/src/ATen/core/VariableFallbackKernel.cpp:108 [backend fallback]
AutogradLazy: registered at /pytorch/aten/src/ATen/core/VariableFallbackKernel.cpp:91 [backend fallback]
AutogradMTIA: registered at /pytorch/aten/src/ATen/core/VariableFallbackKernel.cpp:79 [backend fallback]
AutogradMAIA: registered at /pytorch/aten/src/ATen/core/VariableFallbackKernel.cpp:83 [backend fallback]
AutogradMeta: registered at /pytorch/aten/src/ATen/core/VariableFallbackKernel.cpp:99 [backend fallback]
Tracer: registered at /pytorch/torch/csrc/autograd/TraceTypeManual.cpp:294 [backend fallback]
AutocastCPU: fallthrough registered at /pytorch/aten/src/ATen/autocast_mode.cpp:324 [backend fallback]
AutocastMTIA: fallthrough registered at /pytorch/aten/src/ATen/autocast_mode.cpp:468 [backend fallback]
AutocastMAIA: fallthrough registered at /pytorch/aten/src/ATen/autocast_mode.cpp:506 [backend fallback]
AutocastXPU: fallthrough registered at /pytorch/aten/src/ATen/autocast_mode.cpp:544 [backend fallback]
AutocastMPS: fallthrough registered at /pytorch/aten/src/ATen/autocast_mode.cpp:209 [backend fallback]
AutocastCUDA: fallthrough registered at /pytorch/aten/src/ATen/autocast_mode.cpp:165 [backend fallback]
FuncTorchBatched: registered at /pytorch/aten/src/ATen/functorch/LegacyBatchingRegistrations.cpp:731 [backend fallback]
BatchedNestedTensor: registered at /pytorch/aten/src/ATen/functorch/LegacyBatchingRegistrations.cpp:758 [backend fallback]
FuncTorchVmapMode: fallthrough registered at /pytorch/aten/src/ATen/functorch/VmapModeRegistrations.cpp:27 [backend fallback]
Batched: registered at /pytorch/aten/src/ATen/LegacyBatchingRegistrations.cpp:1075 [backend fallback]
VmapMode: fallthrough registered at /pytorch/aten/src/ATen/VmapModeRegistrations.cpp:33 [backend fallback]
FuncTorchGradWrapper: registered at /pytorch/aten/src/ATen/functorch/TensorWrapper.cpp:210 [backend fallback]
PythonTLSSnapshot: registered at /pytorch/aten/src/ATen/core/PythonFallbackKernel.cpp:202 [backend fallback]
FuncTorchDynamicLayerFrontMode: registered at /pytorch/aten/src/ATen/functorch/DynamicLayer.cpp:475 [backend fallback]
PreDispatch: registered at /pytorch/aten/src/ATen/core/PythonFallbackKernel.cpp:206 [backend fallback]
PythonDispatcher: registered at /pytorch/aten/src/ATen/core/PythonFallbackKernel.cpp:198 [backend fallback]

Exception raised from reportError at /pytorch/aten/src/ATen/core/dispatch/OperatorEntry.cpp:650 (most recent call first):
frame #0: c10::Error::Error(c10::SourceLocation, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >) + 0xb0 (0xeeeb464ac700 in /home/cfregly/.local/lib/python3.11/site-packages/torch/lib/libc10.so)
frame #1: c10::impl::OperatorEntry::reportError(c10::DispatchKey) const + 0x448 (0xeeeb630d2ed8 in /home/cfregly/.local/lib/python3.11/site-packages/torch/lib/libtorch_cpu.so)
frame #2: <unknown function> + 0x1348c28 (0xeeeb62e58c28 in /home/cfregly/.local/lib/python3.11/site-packages/torch/lib/libtorch_cpu.so)
frame #3: <unknown function> + 0x54dfd24 (0xeeeb66fefd24 in /home/cfregly/.local/lib/python3.11/site-packages/torch/lib/libtorch_cpu.so)
frame #4: <unknown function> + 0x5c4c7fc (0xeeeb6775c7fc in /home/cfregly/.local/lib/python3.11/site-packages/torch/lib/libtorch_cpu.so)
frame #5: <unknown function> + 0xd4b67c (0xeeeb6c85b67c in /home/cfregly/.local/lib/python3.11/site-packages/torch/lib/libtorch_python.so)
frame #6: <unknown function> + 0xd4ba14 (0xeeeb6c85ba14 in /home/cfregly/.local/lib/python3.11/site-packages/torch/lib/libtorch_python.so)
frame #7: torch::jit::_get_operation_for_overload_or_packet(std::vector<std::shared_ptr<torch::jit::Operator>, std::allocator<std::shared_ptr<torch::jit::Operator> > > const&, c10::Symbol, pybind11::args const&, pybind11::kwargs const&, bool, std::optional<c10::DispatchKey>) + 0x3c (0xeeeb6c85baac in /home/cfregly/.local/lib/python3.11/site-packages/torch/lib/libtorch_python.so)
frame #8: <unknown function> + 0xc434e0 (0xeeeb6c7534e0 in /home/cfregly/.local/lib/python3.11/site-packages/torch/lib/libtorch_python.so)
frame #9: <unknown function> + 0x5d6d4c (0xeeeb6c0e6d4c in /home/cfregly/.local/lib/python3.11/site-packages/torch/lib/libtorch_python.so)
frame #10: python() [0x4d9630]
<omitting python frames>
frame #16: python() [0x616798]
frame #19: python() [0x514354]
frame #21: python() [0x514354]
frame #26: python() [0x616798]
frame #29: python() [0x514354]
frame #31: python() [0x514354]
frame #36: python() [0x616798]
frame #41: python() [0x514c68]
frame #42: python() [0x514548]
frame #43: python() [0x64013c]
frame #44: python() [0x5f6de4]
frame #45: <unknown function> + 0x8595c (0xeeeb7ae3595c in /lib/aarch64-linux-gnu/libc.so.6)
frame #46: <unknown function> + 0xebb0c (0xeeeb7ae9bb0c in /lib/aarch64-linux-gnu/libc.so.6)



### CH15

**nvlink**
- Baseline: `baseline_nvlink.py` (19.74 ms)
- `optimized_nvlink.py`: 5.47 ms (3.61x speedup)
- Best speedup: 3.61x

**flash**
- Baseline: `baseline_flash_attention.py` (1.08 ms)
- `optimized_flash_attention.py`: 1.02 ms (1.06x speedup)
- Best speedup: 1.06x

**continuous**
- Baseline: `baseline_continuous_batching.py` (3.25 ms)
- `optimized_continuous_batching.py`: 0.02 ms (171.74x speedup)
- Best speedup: 171.74x

**roofline**
- Baseline: `baseline_roofline.py` (0.14 ms)
- `optimized_roofline.py`: 0.43 ms (0.33x speedup)

**kv**
- Baseline: `baseline_kv_cache_management.py` (3.11 ms)
- Failed: None

**inference**
- Baseline: `baseline_inference_monolithic.py` (3.94 ms)
- `optimized_inference_disaggregated.py`: 3.91 ms (1.01x speedup)
- Best speedup: 1.01x

**nccl**
- Baseline: `baseline_nccl.py` (3.21 ms)
- `optimized_nccl.py`: 0.16 ms (19.77x speedup)
- Best speedup: 19.77x


### CH16

**occupancy**
- Baseline: `baseline_occupancy.py` (0.68 ms)
- `optimized_occupancy.py`: 0.11 ms (6.13x speedup)
- Best speedup: 6.13x

**moe**
- Baseline: `baseline_moe_dense.py` (259.91 ms)
- `optimized_moe_sparse.py`: 148.08 ms (1.76x speedup)
- Best speedup: 1.76x

**paged**
- Baseline: `baseline_paged_attention.py`
- Failed: Baseline execution failed: Benchmark failed: Benchmark execution failed: not enough values to unpack (expected 3, got 2)

**shared**
- Baseline: `baseline_shared_memory.py` (0.22 ms)
- `optimized_shared_memory.py`: 0.23 ms (0.97x speedup)

**coalescing**
- Baseline: `baseline_coalescing.py` (0.17 ms)
- `optimized_coalescing.py`: 0.46 ms (0.38x speedup)

**tiling**
- Baseline: `baseline_tiling.py` (0.09 ms)
- Failed: None

**disaggregated**
- Baseline: `baseline_disaggregated.py` (0.65 ms)
- `optimized_disaggregated.py`: 0.72 ms (0.90x speedup)

**regional**
- Baseline: `baseline_regional_compilation.py`
- Failed: Failed to load baseline


### CH17

**continuous**
- Baseline: `baseline_continuous_batching.py` (5.14 ms)
- `optimized_continuous_batching.py`: 0.02 ms (297.26x speedup)
- Best speedup: 297.26x

**paged**
- Baseline: `baseline_paged_attention.py`
- Failed: Baseline execution failed: Benchmark failed: Benchmark execution failed: not enough values to unpack (expected 3, got 2)

**inference**
- Baseline: `baseline_inference_full.py` (3.42 ms)
- `optimized_inference_early_exit.py`: 1.58 ms (2.17x speedup)
- Best speedup: 2.17x

**speculative**
- Baseline: `baseline_speculative_decoding.py`
- Failed: Baseline execution failed: Benchmark failed: Benchmark execution failed: TransformerDecoder.forward() missing 1 required positional argument: 'memory'

**moe**
- Baseline: `baseline_moe.py` (0.08 ms)
- Failed: None

**routing**
- Baseline: `baseline_routing_static.py` (3.27 ms)
- `optimized_routing_dynamic.py`: 1.30 ms (2.52x speedup)
- Best speedup: 2.52x

**kv**
- Baseline: `baseline_kv_cache_management.py` (3.05 ms)
- Failed: None

**attention**
- Baseline: `baseline_attention.py` (0.15 ms)
- `optimized_attention.py`: 0.17 ms (0.88x speedup)


### CH18

**streams**
- Baseline: `baseline_streams.py` (0.35 ms)
- `optimized_streams.py`: 0.37 ms (0.95x speedup)

**ai**
- Baseline: `baseline_ai_optimization.py` (0.14 ms)
- `optimized_ai_optimization.py`: 2.43 ms (0.06x speedup)

**distributed**
- Baseline: `baseline_distributed.py` (0.15 ms)
- `optimized_distributed.py`: 0.16 ms (0.93x speedup)

**quantization**
- Baseline: `baseline_quantization.py` (0.17 ms)
- `optimized_quantization.py`: 0.16 ms (1.03x speedup)
- Best speedup: 1.03x

**speculative**
- Baseline: `baseline_speculative_decoding.py`
- Failed: Baseline execution failed: Benchmark failed: Benchmark execution failed: TransformerDecoder.forward() missing 1 required positional argument: 'memory'

**shared**
- Baseline: `baseline_shared_memory.py` (0.15 ms)
- `optimized_shared_memory.py`: 0.18 ms (0.82x speedup)

**roofline**
- Baseline: `baseline_roofline.py` (0.20 ms)
- `optimized_roofline.py`: 0.38 ms (0.52x speedup)

**warp**
- Baseline: `baseline_warp_specialization.py` (0.17 ms)
- `optimized_warp_specialization.py`: 0.16 ms (1.09x speedup)
- Best speedup: 1.09x

**disaggregated**
- Baseline: `baseline_disaggregated.py`
- Failed: Baseline execution failed: Benchmark failed: Benchmark execution failed: TransformerDecoder.forward() missing 1 required positional argument: 'memory'

**nccl**
- Baseline: `baseline_nccl.py` (3.57 ms)
- `optimized_nccl.py`: 0.32 ms (11.32x speedup)
- Best speedup: 11.32x

**attention**
- Baseline: `baseline_attention.py` (8.77 ms)
- `optimized_attention_flash.py`: 1.51 ms (5.79x speedup)
- Best speedup: 5.79x


### CH19

**memory**
- Baseline: `baseline_memory_coalescing.py` (2.25 ms)
- `optimized_memory_double_buffering.py`: 2.13 ms (1.06x speedup)
- `optimized_memory_flash_attention.py`: 1.07 ms (2.10x speedup)
- `optimized_memory_moe.py`: 22.53 ms (0.10x speedup)
- `optimized_memory_coalescing.py`: 1.76 ms (1.28x speedup)
- Best speedup: 2.10x

**continuous**
- Baseline: `baseline_continuous_batching.py` (0.04 ms)
- `optimized_continuous_batching.py`: 0.50 ms (0.09x speedup)

**vectorization**
- Baseline: `baseline_vectorization_memory.py` (0.08 ms)
- `optimized_vectorization_memory.py`: 0.09 ms (0.93x speedup)

**disaggregated**
- Baseline: `baseline_disaggregated_memory.py` (0.06 ms)
- `optimized_disaggregated_memory.py`: 0.33 ms (0.17x speedup)

**precision**
- Baseline: `baseline_precision_bf16.py` (122.71 ms)
- `optimized_precision_fp8.py`: 183.04 ms (0.67x speedup)
- `optimized_precision_fp4.py`: Benchmark failed: TIMEOUT: Benchmark exceeded timeout of 15 seconds

**triton**
- Baseline: `baseline_triton_memory.py` (18.43 ms)
- `optimized_triton_memory.py`: 0.13 ms (146.57x speedup)
- Best speedup: 146.57x

**cutlass**
- Baseline: `baseline_cutlass_memory.py` (0.03 ms)
- Failed: None

**memory**
- Baseline: `baseline_memory_double_buffering.py` (3.09 ms)
- `optimized_memory_double_buffering.py`: 2.66 ms (1.16x speedup)
- `optimized_memory_flash_attention.py`: 1.21 ms (2.55x speedup)
- `optimized_memory_moe.py`: 9.84 ms (0.31x speedup)
- `optimized_memory_coalescing.py`: 1.53 ms (2.02x speedup)
- Best speedup: 2.55x


### CH2

**continuous**
- Baseline: `baseline_continuous_batching.py` (1.61 ms)
- `optimized_continuous_batching.py`: 0.02 ms (77.69x speedup)
- Best speedup: 77.69x

**cutlass**
- Baseline: `baseline_cutlass.py` (0.35 ms)
- Failed: None

**speculative**
- Baseline: `baseline_speculative_decoding.py`
- Failed: Baseline execution failed: Benchmark failed: Benchmark execution failed: TransformerDecoder.forward() missing 1 required positional argument: 'memory'

**moe**
- Baseline: `baseline_moe.py` (0.09 ms)
- Failed: None

**memory**
- Baseline: `baseline_memory_transfer.py` (2.76 ms)
- `optimized_memory_transfer.py`: 0.44 ms (6.32x speedup)
- Best speedup: 6.32x

**attention**
- Baseline: `baseline_attention.py` (0.13 ms)
- `optimized_attention.py`: 0.17 ms (0.77x speedup)

**memory**
- Baseline: `baseline_memory_transfer.cu` (6206.08 ms)
- `optimized_memory_transfer_nvlink.cu`: 21285.36 ms (0.29x speedup)
- `optimized_memory_transfer_zero_copy.cu`: 1636.93 ms (3.79x speedup)
- Best speedup: 3.79x


### CH20

**occupancy**
- Baseline: `baseline_occupancy.py` (4.87 ms)
- `optimized_occupancy.py`: 0.05 ms (106.17x speedup)
- Best speedup: 106.17x

**nvlink**
- Baseline: `baseline_nvlink.py` (3.02 ms)
- `optimized_nvlink.py`: 1.79 ms (1.69x speedup)
- Best speedup: 1.69x

**memory**
- Baseline: `baseline_memory_standard.py` (3.28 ms)
- `optimized_memory_hbm3e.py`: 2.48 ms (1.32x speedup)
- Best speedup: 1.32x

**integrated**
- Baseline: `baseline_integrated_kv_cache.py` (45.10 ms)
- `optimized_integrated_kv_cache.py`: 135.89 ms (0.33x speedup)

**precision**
- Baseline: `baseline_precision_bf16.py` (11.39 ms)
- `optimized_precision_fp8.py`: 11.55 ms (0.99x speedup)
- `optimized_precision_fp4.py`: 12.92 ms (0.88x speedup)

**moe**
- Baseline: `baseline_moe.py` (0.75 ms)
- `optimized_moe.py`: 23.41 ms (0.03x speedup)

**batching**
- Baseline: `baseline_batching_static.py` (0.42 ms)
- `optimized_batching_continuous.py`: 0.57 ms (0.73x speedup)

**autotuning**
- Baseline: `baseline_autotuning.py` (0.04 ms)
- Failed: None

**end**
- Baseline: `baseline_end_to_end_bandwidth.py` (0.99 ms)
- Failed: None

**multiple**
- Baseline: `baseline_multiple_unoptimized.py`
- Failed: Failed to load baseline

**pipeline**
- Baseline: `baseline_pipeline_sequential.py` (0.52 ms)
- `optimized_pipeline_overlap.py`: 0.29 ms (1.81x speedup)
- Best speedup: 1.81x

**inference**
- Baseline: `baseline_inference_monolithic.py` (17.90 ms)
- `optimized_inference_disaggregated.py`: 13.95 ms (1.28x speedup)
- Best speedup: 1.28x

**training**
- Baseline: `baseline_training_single.py` (0.84 ms)
- `optimized_training_distributed.py`: 1.92 ms (0.44x speedup)

**kv**
- Baseline: `baseline_kv_cache_naive.py` (49.61 ms)
- `optimized_kv_cache_paged.py`: 127.84 ms (0.39x speedup)


### CH3

**occupancy**
- Baseline: `baseline_occupancy.py` (0.55 ms)
- `optimized_occupancy.py`: 0.22 ms (2.54x speedup)
- Best speedup: 2.54x

**streams**
- Baseline: `baseline_streams.py` (1.11 ms)
- `optimized_streams.py`: 0.84 ms (1.32x speedup)
- Best speedup: 1.32x

**cutlass**
- Baseline: `baseline_cutlass.py` (0.35 ms)
- Failed: None

**docker**
- Baseline: `baseline_docker.py` (0.21 ms)
- `optimized_docker.py`: 0.06 ms (3.41x speedup)
- Best speedup: 3.41x

**moe**
- Baseline: `baseline_moe.py` (0.21 ms)
- `optimized_moe.py`: 7.24 ms (0.03x speedup)

**autotuning**
- Baseline: `baseline_autotuning.py` (0.12 ms)
- Failed: None

**triton**
- Baseline: `baseline_triton.py` (0.06 ms)
- `optimized_triton.py`: 0.07 ms (0.90x speedup)

**kubernetes**
- Baseline: `baseline_kubernetes.py` (0.12 ms)
- `optimized_kubernetes.py`: 0.14 ms (0.87x speedup)

**hbm**
- Baseline: `baseline_hbm.py` (0.06 ms)
- `optimized_hbm.py`: 0.23 ms (0.28x speedup)

**gemm**
- Baseline: `baseline_gemm.py` (0.34 ms)
- Failed: None

**numa**
- Baseline: `baseline_numa_unaware.py` (65.88 ms)
- `optimized_numa_aware.py`: 67.17 ms (0.98x speedup)


### CH4

**continuous**
- Baseline: `baseline_continuous_batching.py` (0.33 ms)
- `optimized_continuous_batching.py`: 0.01 ms (53.43x speedup)
- Best speedup: 53.43x

**no**
- Baseline: `baseline_no_overlap.py` (1.29 ms)
- Failed: None

**dataparallel**
- Baseline: `baseline_dataparallel.py` (0.27 ms)
- Failed: None

**kv**
- Baseline: `baseline_kv_cache_management.py` (2.23 ms)
- Failed: None

**disaggregated**
- Baseline: `baseline_disaggregated.py` (0.35 ms)
- `optimized_disaggregated.py`: 0.06 ms (6.03x speedup)
- Best speedup: 6.03x

**reinit**
- Baseline: `baseline_reinit_comm.py` (524.72 ms)
- `optimized_reinit_comm.py`: 0.02 ms (21822.68x speedup)
- Best speedup: 21822.68x


### CH5

**nvlink**
- Baseline: `baseline_nvlink.py` (3.94 ms)
- `optimized_nvlink.py`: 1.78 ms (2.22x speedup)
- Best speedup: 2.22x

**ai**
- Baseline: `baseline_ai_optimization.py` (0.01 ms)
- `optimized_ai_optimization.py`: 0.07 ms (0.17x speedup)

**distributed**
- Baseline: `baseline_distributed.py` (0.15 ms)
- `optimized_distributed.py`: 0.17 ms (0.87x speedup)

**roofline**
- Baseline: `baseline_roofline.py` (0.20 ms)
- `optimized_roofline.py`: 0.63 ms (0.31x speedup)

**storage**
- Baseline: `baseline_storage_cpu.py` (78.11 ms)
- `optimized_storage_gds.py`: 67.00 ms (1.17x speedup)
- Best speedup: 1.17x

**vectorization**
- Baseline: `baseline_vectorization.py` (3.72 ms)
- `optimized_vectorization.py`: 0.01 ms (299.38x speedup)
- Best speedup: 299.38x

**nccl**
- Baseline: `baseline_nccl.py` (0.81 ms)
- `optimized_nccl.py`: 0.21 ms (3.84x speedup)
- Best speedup: 3.84x


### CH6

**attention**
- Baseline: `baseline_attention_ilp.py`
- Failed: Failed to load baseline

**ai**
- Baseline: `baseline_ai_optimization.py`
- Failed: Failed to load baseline

**warp**
- Baseline: `baseline_warp_divergence_ilp.py`
- Failed: Failed to load baseline

**quantization**
- Baseline: `baseline_quantization_ilp.py`
- Failed: Failed to load baseline

**distributed**
- Baseline: `baseline_distributed_ilp.py`
- Failed: Failed to load baseline

**adaptive**
- Baseline: `baseline_adaptive.py`
- Failed: Failed to load baseline

**autotuning**
- Baseline: `baseline_autotuning.py`
- Failed: Failed to load baseline

**ilp**
- Baseline: `baseline_ilp.py`
- Failed: Baseline execution failed: Benchmark failed: TIMEOUT: Benchmark exceeded timeout of 15 seconds

**coalescing**
- Baseline: `baseline_coalescing.py`
- Failed: Baseline execution failed: Benchmark failed: TIMEOUT: Benchmark exceeded timeout of 15 seconds

**triton**
- Baseline: `baseline_triton.py`
- Failed: Failed to load baseline

**launch**
- Baseline: `baseline_launch_bounds.py`
- Failed: Baseline execution failed: Benchmark failed: TIMEOUT: Benchmark exceeded timeout of 15 seconds

**add**
- Baseline: `baseline_add.py`
- Failed: Failed to load baseline

**gemm**
- Baseline: `baseline_gemm_ilp.py`
- Failed: Failed to load baseline

**bank**
- Baseline: `baseline_bank_conflicts.py`
- Failed: Baseline execution failed: Benchmark failed: TIMEOUT: Benchmark exceeded timeout of 15 seconds

**coalescing**
- Baseline: `baseline_coalescing_uncoalesced.cu` (504.08 ms)
- `optimized_coalescing.cu`: 362.96 ms (1.39x speedup)
- Best speedup: 1.39x

**launch**
- Baseline: `baseline_launch_bounds.cu`
- Failed: Baseline executable not found for baseline_launch_bounds.cu

**add**
- Baseline: `baseline_add.cu` (296.64 ms)
- `optimized_add_parallel.cu`: 200.65 ms (1.48x speedup)
- Best speedup: 1.48x

**ilp**
- Baseline: `baseline_ilp.cu`
- Failed: Baseline execution failed or timed out

**bank**
- Baseline: `baseline_bank_conflicts.cu`
- Failed: Baseline execution failed or timed out


### CH7

**loop**
- Baseline: `baseline_loop_unrolling.py` (343.95 ms)
- `optimized_loop_unrolling.py`: 329.11 ms (1.05x speedup)
- Best speedup: 1.05x

**occupancy**
- Baseline: `baseline_occupancy.py` (0.08 ms)
- `optimized_occupancy.py`: 0.19 ms (0.44x speedup)

**nvlink**
- Baseline: `baseline_nvlink.py` (140.10 ms)
- `optimized_nvlink.py`: 4.58 ms (30.60x speedup)
- Best speedup: 30.60x

**double**
- Baseline: `baseline_double_buffering.py` (2.56 ms)
- `optimized_double_buffering.py`: 24.39 ms (0.11x speedup)

**lookup**
- Baseline: `baseline_lookup.py` (213.68 ms)
- `optimized_lookup.py`: 196.53 ms (1.09x speedup)
- Best speedup: 1.09x

**cutlass**
- Baseline: `baseline_cutlass.py` (0.36 ms)
- Failed: None

**scalar**
- Baseline: `baseline_scalar_copy.py` (203.76 ms)
- `optimized_scalar_copy.py`: 208.93 ms (0.98x speedup)

**hbm3epeak**
- Baseline: `baseline_hbm3epeak.py` (288.18 ms)
- `optimized_hbm3epeak.py`: 318.32 ms (0.91x speedup)

**memory**
- Baseline: `baseline_memory_access.py` (0.29 ms)
- `optimized_memory_access.py`: 0.43 ms (0.67x speedup)

**async**
- Baseline: `baseline_async_prefetch.py` (192.78 ms)
- `optimized_async_prefetch.py`: 1.55 ms (124.51x speedup)
- Best speedup: 124.51x

**hbm3ecopy**
- Baseline: `baseline_hbm3ecopy.py` (486.99 ms)
- `optimized_hbm3ecopy.py`: 1157.98 ms (0.42x speedup)

**autotuning**
- Baseline: `baseline_autotuning.py` (0.13 ms)
- `optimized_autotuning.py`: 0.14 ms (0.90x speedup)

**roofline**
- Baseline: `baseline_roofline.py` (0.08 ms)
- `optimized_roofline.py`: 0.18 ms (0.41x speedup)

**warp**
- Baseline: `baseline_warp_divergence.py` (0.12 ms)
- `optimized_warp_divergence.py`: 0.05 ms (2.43x speedup)
- Best speedup: 2.43x

**transpose**
- Baseline: `baseline_transpose.py` (197.09 ms)
- `optimized_transpose_padded.py`: 196.27 ms (1.00x speedup)
- Best speedup: 1.00x

**triton**
- Baseline: `baseline_triton.py` (0.22 ms)
- `optimized_triton.py`: 0.08 ms (2.80x speedup)
- Best speedup: 2.80x

**matmul**
- Baseline: `baseline_matmul.py` (187.58 ms)
- `optimized_matmul_tiled.py`: 175.18 ms (1.07x speedup)
- Best speedup: 1.07x

**tma**
- Baseline: `baseline_tma_copy.py` (184.65 ms)
- `optimized_tma_copy.py`: 187.20 ms (0.99x speedup)

**uncoalesced**
- Baseline: `baseline_uncoalesced_copy.py` (193.61 ms)
- `optimized_uncoalesced_copy.py`: 196.67 ms (0.98x speedup)

**loop**
- Baseline: `baseline_loop_unrolling.cu` (199.38 ms)
- `optimized_loop_unrolling.cu`: 198.34 ms (1.01x speedup)
- Best speedup: 1.01x

**hbm3e**
- Baseline: `baseline_hbm3e_copy.cu` (441.70 ms)
- `optimized_hbm3e_peak.cu`: 292.34 ms (1.51x speedup)
- `optimized_hbm3e_copy.cu`: 1137.07 ms (0.39x speedup)
- Best speedup: 1.51x

**async**
- Baseline: `baseline_async_prefetch.cu` (151.81 ms)
- `optimized_async_prefetch.cu`: 1.65 ms (91.74x speedup)
- Best speedup: 91.74x

**matmul**
- Baseline: `baseline_matmul.cu` (150.49 ms)
- `optimized_matmul_tiled.cu`: 164.03 ms (0.92x speedup)

**transpose**
- Baseline: `baseline_transpose.cu` (199.84 ms)
- `optimized_transpose_padded.cu`: 187.54 ms (1.07x speedup)
- Best speedup: 1.07x

**hbm3e**
- Baseline: `baseline_hbm3e_peak.cu` (292.58 ms)
- `optimized_hbm3e_peak.cu`: 307.32 ms (0.95x speedup)
- `optimized_hbm3e_copy.cu`: 1200.68 ms (0.24x speedup)

**tma**
- Baseline: `baseline_tma_copy.cu` (148.73 ms)
- `optimized_tma_copy.cu`: 150.47 ms (0.99x speedup)

**copy**
- Baseline: `baseline_copy_uncoalesced.cu` (154.32 ms)
- `optimized_copy_coalesced.cu`: 153.23 ms (1.01x speedup)
- `optimized_copy_vectorized.cu`: 157.92 ms (0.98x speedup)
- Best speedup: 1.01x

**lookup**
- Baseline: `baseline_lookup.cu` (156.19 ms)
- `optimized_lookup.cu`: 154.97 ms (1.01x speedup)
- Best speedup: 1.01x

**copy**
- Baseline: `baseline_copy_scalar.cu` (151.49 ms)
- `optimized_copy_coalesced.cu`: 151.91 ms (1.00x speedup)
- `optimized_copy_vectorized.cu`: 155.55 ms (0.97x speedup)


### CH8

**loop**
- Baseline: `baseline_loop_unrolling.py` (146.85 ms)
- `optimized_loop_unrolling.py`: 154.39 ms (0.95x speedup)

**occupancy**
- Baseline: `baseline_occupancy.py` (8.54 ms)
- `optimized_occupancy.py`: 0.05 ms (169.95x speedup)
- Best speedup: 169.95x

**nvlink**
- Baseline: `baseline_nvlink.py` (120.34 ms)
- `optimized_nvlink.py`: 2.31 ms (52.08x speedup)
- Best speedup: 52.08x

**double**
- Baseline: `baseline_double_buffering.py` (4.87 ms)
- `optimized_double_buffering.py`: 0.78 ms (6.24x speedup)
- Best speedup: 6.24x

**ai**
- Baseline: `baseline_ai_optimization.py` (0.12 ms)
- `optimized_ai_optimization.py`: 0.88 ms (0.13x speedup)

**distributed**
- Baseline: `baseline_distributed.py` (0.23 ms)
- `optimized_distributed.py`: 0.16 ms (1.43x speedup)
- Best speedup: 1.43x

**quantization**
- Baseline: `baseline_quantization.py` (0.25 ms)
- `optimized_quantization.py`: 0.21 ms (1.20x speedup)
- Best speedup: 1.20x

**roofline**
- Baseline: `baseline_roofline.py` (0.07 ms)
- `optimized_roofline.py`: 0.26 ms (0.26x speedup)

**coalescing**
- Baseline: `baseline_coalescing.py` (0.16 ms)
- `optimized_coalescing.py`: 0.04 ms (3.74x speedup)
- Best speedup: 3.74x

**threshold**
- Baseline: `baseline_threshold.py` (1386.51 ms)
- `optimized_threshold_predicated.py`: 2014.97 ms (0.69x speedup)

**tiling**
- Baseline: `baseline_tiling.py` (0.29 ms)
- `optimized_tiling.py`: 0.53 ms (0.55x speedup)

**nccl**
- Baseline: `baseline_nccl.py` (0.65 ms)
- `optimized_nccl.py`: 0.19 ms (3.38x speedup)
- Best speedup: 3.38x

**hbm**
- Baseline: `baseline_hbm.py` (0.19 ms)
- `optimized_hbm.py`: 0.23 ms (0.83x speedup)

**loop**
- Baseline: `baseline_loop_unrolling.cu` (198.97 ms)
- `optimized_loop_unrolling.cu`: 185.50 ms (1.07x speedup)
- Best speedup: 1.07x

**threshold**
- Baseline: `baseline_threshold.cu` (1390.93 ms)
- `optimized_threshold_predicated.cu`: 1976.52 ms (0.70x speedup)


### CH9

**nvlink**
- Baseline: `baseline_nvlink.py` (107.64 ms)
- `optimized_nvlink.py`: 2.36 ms (45.61x speedup)
- Best speedup: 45.61x

**double**
- Baseline: `baseline_double_buffering.py` (3.62 ms)
- `optimized_double_buffering.py`: 1.01 ms (3.60x speedup)
- Best speedup: 3.60x

**flash**
- Baseline: `baseline_flash_attention.py` (0.22 ms)
- `optimized_flash_attention.py`: 0.07 ms (3.24x speedup)
- Best speedup: 3.24x

**memory**
- Baseline: `baseline_memory_bound.py` (0.40 ms)
- `optimized_memory_bound.py`: 4.39 ms (0.09x speedup)

**distributed**
- Baseline: `baseline_distributed.py` (0.14 ms)
- `optimized_distributed.py`: 0.10 ms (1.41x speedup)
- Best speedup: 1.41x

**quantization**
- Baseline: `baseline_quantization.py` (0.05 ms)
- `optimized_quantization.py`: 0.14 ms (0.34x speedup)

**moe**
- Baseline: `baseline_moe.py` (0.12 ms)
- `optimized_moe.py`: 7.11 ms (0.02x speedup)

**autotuning**
- Baseline: `baseline_autotuning.py` (0.13 ms)
- Failed: None

**cutlass**
- Baseline: `baseline_cutlass_gemm.py` (127.83 ms)
- `optimized_cutlass_gemm.py`: 128.74 ms (0.99x speedup)

**warp**
- Baseline: `baseline_warp_specialization.py` (0.22 ms)
- `optimized_warp_specialization.py`: 0.12 ms (1.85x speedup)
- Best speedup: 1.85x

**micro**
- Baseline: `baseline_micro_tiling_matmul.py` (255.93 ms)
- `optimized_micro_tiling_matmul.py`: 183.13 ms (1.40x speedup)
- Best speedup: 1.40x

**compute**
- Baseline: `baseline_compute_bound.py` (4.39 ms)
- `optimized_compute_bound.py`: 4.41 ms (1.00x speedup)

**triton**
- Baseline: `baseline_triton.py` (0.08 ms)
- `optimized_triton.py`: 0.04 ms (1.78x speedup)
- Best speedup: 1.78x

**hbm**
- Baseline: `baseline_hbm.py` (0.14 ms)
- `optimized_hbm.py`: 0.22 ms (0.61x speedup)

**fused**
- Baseline: `baseline_fused_l2norm.py` (169.39 ms)
- `optimized_fused_l2norm.py`: 143.00 ms (1.18x speedup)
- Best speedup: 1.18x

**bank**
- Baseline: `baseline_bank_conflicts.py` (0.14 ms)
- `optimized_bank_conflicts.py`: 0.05 ms (2.66x speedup)
- Best speedup: 2.66x

**micro**
- Baseline: `baseline_micro_tiling_matmul.cu` (258.74 ms)
- `optimized_micro_tiling_matmul.cu`: 186.58 ms (1.39x speedup)
- Best speedup: 1.39x

**fused**
- Baseline: `baseline_fused_l2norm.cu` (181.02 ms)
- `optimized_fused_l2norm.cu`: 147.13 ms (1.23x speedup)
- Best speedup: 1.23x

**cutlass**
- Baseline: `baseline_cutlass_gemm.cu` (140.47 ms)
- `optimized_cutlass_gemm.cu`: 137.36 ms (1.02x speedup)
- Best speedup: 1.02x


