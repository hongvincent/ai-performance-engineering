#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

constexpr int kThreadsPerBlock = 256;
constexpr int kElementsPerThread = 8;
constexpr int kElementsPerBlock = kThreadsPerBlock * kElementsPerThread;
constexpr int kClusterBlocks = 2;  // Producer (sum) + consumer (sum-of-squares)
constexpr int kTotalElements = 1 << 24;  // ~16 million elements (~64 MB)
constexpr int kIterations = 200;

inline void initialize_input(std::vector<float>& buffer) {
    for (std::size_t i = 0; i < buffer.size(); ++i) {
        float base = std::sin(0.0005f * static_cast<float>(i));
        buffer[i] = base + static_cast<float>(i % 17) * 0.03125f;
    }
}

inline void compute_reference(
    const std::vector<float>& input,
    std::vector<float>& sums,
    std::vector<float>& squares,
    int elems_per_block) {
    const std::size_t chunks = (input.size() + elems_per_block - 1) / elems_per_block;
    sums.assign(chunks, 0.0f);
    squares.assign(chunks, 0.0f);
    for (std::size_t chunk = 0; chunk < chunks; ++chunk) {
        const std::size_t base = chunk * elems_per_block;
        const std::size_t limit = std::min(base + static_cast<std::size_t>(elems_per_block), input.size());
        float sum = 0.0f;
        float sq_sum = 0.0f;
        for (std::size_t idx = base; idx < limit; ++idx) {
            const float val = input[idx];
            sum += val;
            sq_sum += val * val;
        }
        sums[chunk] = sum;
        squares[chunk] = sq_sum;
    }
}

inline int num_chunks(int elems_per_block = kElementsPerBlock) {
    return (kTotalElements + elems_per_block - 1) / elems_per_block;
}
