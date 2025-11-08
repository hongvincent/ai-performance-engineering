// baseline_cluster_group.cu -- Regular launch without clusters (baseline).

#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "cluster_group_common.cuh"

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t _status = (call);                                           \
        if (_status != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error %s (%d) at %s:%d\n",                    \
                    cudaGetErrorString(_status), _status, __FILE__, __LINE__);  \
            std::abort();                                                       \
        }                                                                       \
    } while (0)

__global__ void baseline_atomic_kernel(const float* __restrict__ in,
                                       float* __restrict__ final_sum,
                                       float* __restrict__ final_sq,
                                       int elems_per_block,
                                       int total_elements) {
    extern __shared__ float scratch[];
    float* sum_scratch = scratch;
    float* sq_scratch = scratch + blockDim.x;

    const int chunk_id = blockIdx.x / kClusterBlocks;
    const size_t chunk_base = static_cast<size_t>(chunk_id) * elems_per_block;
    if (chunk_base >= static_cast<size_t>(total_elements)) {
        return;
    }
    const int remaining = total_elements - static_cast<int>(chunk_base);
    const int valid = remaining < elems_per_block ? remaining : elems_per_block;

    float sum = 0.0f;
    float sq = 0.0f;
    for (int i = threadIdx.x; i < valid; i += blockDim.x) {
        const float value = in[chunk_base + i];
        sum += value;
        sq += value * value;
    }

    sum_scratch[threadIdx.x] = sum;
    sq_scratch[threadIdx.x] = sq;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sum_scratch[threadIdx.x] += sum_scratch[threadIdx.x + stride];
            sq_scratch[threadIdx.x] += sq_scratch[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        if ((blockIdx.x % kClusterBlocks) == 0) {
            atomicAdd(&final_sum[chunk_id], sum_scratch[0]);
        } else {
            atomicAdd(&final_sq[chunk_id], sq_scratch[0]);
        }
    }
}

int main() {
    CUDA_CHECK(cudaSetDevice(0));

    std::vector<float> h_input(kTotalElements);
    initialize_input(h_input);

    const int chunks = num_chunks();
    const size_t input_bytes = h_input.size() * sizeof(float);
    const size_t final_bytes = chunks * sizeof(float);

    float* d_input = nullptr;
    float* d_final_sum = nullptr;
    float* d_final_sq = nullptr;

    CUDA_CHECK(cudaMalloc(&d_input, input_bytes));
    CUDA_CHECK(cudaMalloc(&d_final_sum, final_bytes));
    CUDA_CHECK(cudaMalloc(&d_final_sq, final_bytes));

    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), input_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_final_sum, 0, final_bytes));
    CUDA_CHECK(cudaMemset(d_final_sq, 0, final_bytes));

    dim3 block(kThreadsPerBlock);
    dim3 grid(chunks * kClusterBlocks);
    const size_t shared_bytes = block.x * 2 * sizeof(float);

    // Warmup
    baseline_atomic_kernel<<<grid, block, shared_bytes>>>(
        d_input, d_final_sum, d_final_sq, kElementsPerBlock, kTotalElements);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemset(d_final_sum, 0, final_bytes));
    CUDA_CHECK(cudaMemset(d_final_sq, 0, final_bytes));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < kIterations; ++i) {
        baseline_atomic_kernel<<<grid, block, shared_bytes>>>(
            d_input, d_final_sum, d_final_sq, kElementsPerBlock, kTotalElements);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    printf("Baseline (atomics): %.2f ms\n", elapsed_ms / kIterations);

    // Produce single-run output for verification
    CUDA_CHECK(cudaMemset(d_final_sum, 0, final_bytes));
    CUDA_CHECK(cudaMemset(d_final_sq, 0, final_bytes));
    baseline_atomic_kernel<<<grid, block, shared_bytes>>>(
        d_input, d_final_sum, d_final_sq, kElementsPerBlock, kTotalElements);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> h_sum(chunks, 0.0f);
    std::vector<float> h_squares(chunks, 0.0f);
    CUDA_CHECK(cudaMemcpy(h_sum.data(), d_final_sum, final_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_squares.data(), d_final_sq, final_bytes, cudaMemcpyDeviceToHost));

    std::vector<float> ref_sum;
    std::vector<float> ref_squares;
    compute_reference(h_input, ref_sum, ref_squares, kElementsPerBlock);

    auto max_diff = [](const std::vector<float>& a, const std::vector<float>& b) {
        float diff = 0.0f;
        const std::size_t limit = std::min(a.size(), b.size());
        for (std::size_t i = 0; i < limit; ++i) {
            diff = std::max(diff, std::abs(a[i] - b[i]));
        }
        return diff;
    };

    printf("Verification (baseline): max |sum diff|=%.6f, |sq diff|=%.6f\n",
           max_diff(h_sum, ref_sum),
           max_diff(h_squares, ref_squares));

    if (std::getenv("AIPERF_DEBUG_CHUNK")) {
        const int inspect = std::min<int>(5, chunks);
        for (int i = 0; i < inspect; ++i) {
            printf("  chunk %d sum %.6f (ref %.6f) sq %.6f (ref %.6f)\n",
                   i,
                   h_sum[i],
                   ref_sum[i],
                   h_squares[i],
                   ref_squares[i]);
        }
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_final_sum));
    CUDA_CHECK(cudaFree(d_final_sq));

    return 0;
}
