# Makefile for Chapter 1 - Performance Basics Examples
# Builds optimized CUDA examples demonstrating profiling improvements.

include ../common/cuda_arch.mk

NVCC_FLAGS = $(CUDA_NVCC_ARCH_FLAGS) --use_fast_math -lineinfo
NVCC_LIBS = -lcublas -lcublasLt
SUFFIX = $(TARGET_SUFFIX)

BASE_TARGETS = baseline_gemm optimized_gemm_batched optimized_gemm_strided baseline_arithmetic_intensity optimized_arithmetic_intensity_combined
TARGETS = $(addsuffix $(SUFFIX), $(BASE_TARGETS))

.PHONY: all clean test compare blackwell grace-blackwell

all: $(TARGETS)
	@echo "Building Chapter 1 examples with $(ARCH) (CUDA $(CUDA_VERSION))"
	@echo "✓ Targeting $(ARCH_NAME)"

baseline_gemm$(SUFFIX): baseline_gemm.cu
	$(NVCC) $(NVCC_FLAGS) $(NVCC_LIBS) -o $@ $<

optimized_gemm_batched$(SUFFIX): optimized_gemm_batched.cu
	$(NVCC) $(NVCC_FLAGS) $(NVCC_LIBS) -o $@ $<

optimized_gemm_strided$(SUFFIX): optimized_gemm_strided.cu
	$(NVCC) $(NVCC_FLAGS) $(NVCC_LIBS) -o $@ $<

baseline_arithmetic_intensity$(SUFFIX): baseline_arithmetic_intensity.cu
	$(NVCC) $(NVCC_FLAGS) $(NVCC_LIBS) -o $@ $<

optimized_arithmetic_intensity_combined$(SUFFIX): optimized_arithmetic_intensity_combined.cu
	$(NVCC) $(NVCC_FLAGS) $(NVCC_LIBS) -o $@ $<

test: all
	@echo "Testing batched GEMM optimization..."
	./batched_gemm_example$(SUFFIX)
	@echo ""
	@echo "Testing PyTorch optimizations..."
	cd .. && PYTHONPATH=.:$$PYTHONPATH python3 ch1/performance_basics_optimized.py

compare: clean
	@echo "========================================"
	@echo "Building Chapter 1 for both architectures..."
	@echo "========================================"
	@for arch in $(ARCH_LIST); do \
		echo ""; \
		echo ">>> Building $${arch}..."; \
		$(MAKE) ARCH=$${arch} all; \
	done
	@echo ""
	@echo "Run binaries:"
	@echo "  ./batched_gemm_example_sm100"
	@echo "  ./batched_gemm_example_sm121"

blackwell: ARCH=sm_100
blackwell: clean all

grace-blackwell: ARCH=sm_121
grace-blackwell: clean all

clean:
	rm -f batched_gemm_example_sm100 batched_gemm_example_sm121 *.o *.so *.a
	rm -f arithmetic_intensity_demo_sm100 arithmetic_intensity_demo_sm121
	rm -f nsys_profile_* ncu_profile_* hta_profile_* perf.data_* comprehensive_*
	@echo "Cleaned ch1 binaries and profiling artifacts"
