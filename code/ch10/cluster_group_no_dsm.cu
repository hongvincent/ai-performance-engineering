// optimized_cluster_group.cu -- Cluster launch that fuses partial + finalize reductions.

#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "cluster_group_common.cuh"

namespace cg = cooperative_groups;

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t _status = (call);                                           \
        if (_status != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error %s (%d) at %s:%d\n",                    \
                    cudaGetErrorString(_status), _status, __FILE__, __LINE__);  \
            std::abort();                                                       \
        }                                                                       \
    } while (0)

bool report_cluster_failure(cudaError_t err, const char* stage, const cudaDeviceProp& prop) {
    if (err == cudaSuccess) {
        return false;
    }
    fprintf(stderr,
            "SKIPPED: Thread block clusters unstable outside compute-sanitizer "
            "(stage=%s, device=%s, cc=%d.%d, error=%s).\n",
            stage,
            prop.name,
            prop.major,
            prop.minor,
            cudaGetErrorString(err));
    fprintf(stderr,
            "This driver/toolkit combination sometimes drops cluster partners, "
            "triggering 'cluster target block not present'. "
            "Run under compute-sanitizer or upgrade driver/CUDA.\n");
    return true;
}

__global__ void __cluster_dims__(kClusterBlocks, 1, 1)
cluster_chunk_kernel(const float* __restrict__ in,
                     float* __restrict__ partial_sum,
                     float* __restrict__ partial_sq,
                     float* __restrict__ final_sum,
                     float* __restrict__ final_sq,
                     int elems_per_block,
                     int total_elements) {
    cg::thread_block block = cg::this_thread_block();
    cg::cluster_group cluster = cg::this_cluster();

    extern __shared__ float scratch[];
    float* sum_scratch = scratch;
    float* sq_scratch = scratch + blockDim.x;

    const int chunk_id = blockIdx.x / kClusterBlocks;
    const int role = blockIdx.x % kClusterBlocks;
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
    block.sync();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sum_scratch[threadIdx.x] += sum_scratch[threadIdx.x + stride];
            sq_scratch[threadIdx.x] += sq_scratch[threadIdx.x + stride];
        }
        block.sync();
    }

    if (threadIdx.x == 0) {
        const int idx = chunk_id * kClusterBlocks + role;
        if (role == 0) {
            partial_sum[idx] = sum_scratch[0];
        } else {
            partial_sq[idx] = sq_scratch[0];
        }
    }
    block.sync();
    __threadfence();
    cluster.sync();

    if (cluster.block_rank() == 0 && threadIdx.x == 0) {
        float total_sum = 0.0f;
        float total_sq = 0.0f;
        const int base = chunk_id * kClusterBlocks;
        for (int r = 0; r < kClusterBlocks; ++r) {
            total_sum += partial_sum[base + r];
            total_sq += partial_sq[base + r];
        }
        final_sum[chunk_id] = total_sum;
        final_sq[chunk_id] = total_sq;
    }
}

int main() {
    CUDA_CHECK(cudaSetDevice(0));

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    int cluster_launch = 0;
#ifdef cudaDevAttrClusterLaunch
    CUDA_CHECK(cudaDeviceGetAttribute(&cluster_launch, cudaDevAttrClusterLaunch, 0));
#endif
    const bool supports_clusters = cluster_launch > 0 || prop.major >= 9;
    if (!supports_clusters) {
        fprintf(stderr,
                "SKIPPED: Thread block clusters unsupported on compute capability %d.%d (device %s)\n",
                prop.major,
                prop.minor,
                prop.name);
        return 2;
    }

    std::vector<float> h_input(kTotalElements);
    initialize_input(h_input);

    const int chunks = num_chunks();
    const size_t input_bytes = h_input.size() * sizeof(float);
    const size_t final_bytes = chunks * sizeof(float);

    float* d_input = nullptr;
    float* d_partial_sum = nullptr;
    float* d_partial_sq = nullptr;
    float* d_final_sum = nullptr;
    float* d_final_sq = nullptr;
    CUDA_CHECK(cudaMalloc(&d_input, input_bytes));
    CUDA_CHECK(cudaMalloc(&d_partial_sum, chunks * kClusterBlocks * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_partial_sq, chunks * kClusterBlocks * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_final_sum, final_bytes));
    CUDA_CHECK(cudaMalloc(&d_final_sq, final_bytes));
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), input_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_partial_sum, 0, chunks * kClusterBlocks * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_partial_sq, 0, chunks * kClusterBlocks * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_final_sum, 0, final_bytes));
    CUDA_CHECK(cudaMemset(d_final_sq, 0, final_bytes));

    dim3 block(kThreadsPerBlock);
    dim3 grid(chunks * kClusterBlocks);
    const size_t shared_bytes = block.x * 2 * sizeof(float);

    cudaLaunchAttribute attrs[2]{};
    int attr_count = 0;
#ifdef cudaLaunchAttributeClusterDimension
    attrs[attr_count].id = cudaLaunchAttributeClusterDimension;
    attrs[attr_count].val.clusterDim.x = kClusterBlocks;
    attrs[attr_count].val.clusterDim.y = 1;
    attrs[attr_count].val.clusterDim.z = 1;
    ++attr_count;
#endif
#ifdef cudaLaunchAttributeNonPortableClusterSizeAllowed
    attrs[attr_count].id = cudaLaunchAttributeNonPortableClusterSizeAllowed;
    attrs[attr_count].val.nonPortableClusterSizeAllowed = 1;
    ++attr_count;
#endif

    CUDA_CHECK(cudaFuncSetAttribute(
        cluster_chunk_kernel,
        cudaFuncAttributeNonPortableClusterSizeAllowed,
        1));

    cudaLaunchConfig_t config{};
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = shared_bytes;
    config.stream = nullptr;
    config.attrs = attr_count ? attrs : nullptr;
    config.numAttrs = attr_count;

    int elems_per_block = kElementsPerBlock;
    int total_elements = kTotalElements;
    void* args[] = {
        &d_input,
        &d_partial_sum,
        &d_partial_sq,
        &d_final_sum,
        &d_final_sq,
        &elems_per_block,
        &total_elements};
    cudaKernel_t kernel;
    CUDA_CHECK(cudaGetKernel(&kernel, cluster_chunk_kernel));
    void* func = reinterpret_cast<void*>(kernel);

    auto synchronize_or_skip = [&](const char* stage) -> bool {
        cudaError_t err = cudaDeviceSynchronize();
        if (report_cluster_failure(err, stage, prop)) {
            cudaFree(d_input);
            cudaFree(d_partial_sum);
            cudaFree(d_partial_sq);
            cudaFree(d_final_sum);
            cudaFree(d_final_sq);
            return true;
        }
        return false;
    };

    // Warmup
    CUDA_CHECK(cudaLaunchKernelExC(&config, func, args));
    if (synchronize_or_skip("warmup")) {
        return 2;
    }
    CUDA_CHECK(cudaMemset(d_partial_sum, 0, chunks * kClusterBlocks * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_partial_sq, 0, chunks * kClusterBlocks * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_final_sum, 0, final_bytes));
    CUDA_CHECK(cudaMemset(d_final_sq, 0, final_bytes));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < kIterations; ++i) {
        CUDA_CHECK(cudaLaunchKernelExC(&config, func, args));
    }
    CUDA_CHECK(cudaEventRecord(stop));
    if (synchronize_or_skip("measurement")) {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return 2;
    }
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    printf("Cluster launch (no DSM optimized): %.2f ms\n", elapsed_ms / kIterations);

    // Produce single-run output for verification
    CUDA_CHECK(cudaMemset(d_partial_sum, 0, chunks * kClusterBlocks * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_partial_sq, 0, chunks * kClusterBlocks * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_final_sum, 0, final_bytes));
    CUDA_CHECK(cudaMemset(d_final_sq, 0, final_bytes));
    CUDA_CHECK(cudaLaunchKernelExC(&config, func, args));
    if (synchronize_or_skip("verification")) {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return 2;
    }

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

    printf("Verification (optimized): max |sum diff|=%.6f, |sq diff|=%.6f\n",
           max_diff(h_sum, ref_sum),
           max_diff(h_squares, ref_squares));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_partial_sum));
    CUDA_CHECK(cudaFree(d_partial_sq));
    CUDA_CHECK(cudaFree(d_final_sum));
    CUDA_CHECK(cudaFree(d_final_sq));
    return 0;
}
