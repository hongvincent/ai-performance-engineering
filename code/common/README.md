# Common Infrastructure

Shared utilities for all chapter examples to ensure consistent build system, profiling workflow, and benchmarking methodology.

## Directory Structure

```
common/
├── headers/
│   ├── arch_detection.cuh      # GPU architecture detection & limits
│   └── tma_helpers.cuh         # Tensor Memory Accelerator utilities
└── python/
    ├── benchmark_harness.py    # Production-grade benchmarking harness
    ├── chapter_compare_template.py  # Standard template for chapter compare.py
    ├── compile_utils.py        # torch.compile and precision utilities
    └── env_defaults.py         # Environment configuration helpers
```

## Usage

### Build System

In your chapter Makefile:

```makefile
# Include common architecture flags
include ../common/cuda_arch.mk
```

This provides architecture detection and dual-architecture build support (sm_100 + sm_121).

### CUDA Headers

#### Architecture Detection
```cpp
#include "../../common/headers/arch_detection.cuh"

int main() {
    // Query GPU capabilities
    const auto& limits = cuda_arch::get_architecture_limits();
    
    // Check features
    if (limits.supports_clusters) {
        printf("Cluster size: %d\n", limits.max_cluster_size);
    }
    if (limits.has_grace_coherence) {
        printf("Grace-Blackwell coherence available\n");
    }
    
    // Select optimal tile size
    auto tile = cuda_arch::select_tensor_core_tile();
    printf("Tensor core tile: %dx%dx%d\n", tile.m, tile.n, tile.k);
    
    // Get TMA limits
    auto tma = cuda_arch::get_tma_limits();
    printf("TMA 2D box: %ux%u\n", tma.max_2d_box_width, tma.max_2d_box_height);
    
    return 0;
}
```

#### TMA (Tensor Memory Accelerator) Helpers
```cpp
#include "../../common/headers/arch_detection.cuh"
#include "../../common/headers/tma_helpers.cuh"

int main() {
    // Check TMA support
    if (!cuda_tma::device_supports_tma()) {
        printf("TMA not supported (requires SM 9.0+)\n");
        return 1;
    }
    
    // Create tensor map
    CUtensorMap desc;
    auto encode = cuda_tma::load_cuTensorMapEncodeTiled();
    bool ok = cuda_tma::make_2d_tensor_map(
        desc, encode, d_data, width, height, ld,
        box_width, box_height, CU_TENSOR_MAP_SWIZZLE_NONE);
    
    // Use in kernel with cp_async_bulk_tensor operations
    return 0;
}
```

### Python Utilities

#### Environment Configuration
```python
from common.python.env_defaults import apply_env_defaults, dump_environment_and_capabilities

# Apply default environment settings
apply_env_defaults()

# Print environment and hardware capabilities
dump_environment_and_capabilities()
```

#### Benchmarking
```python
from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode, BenchmarkConfig
from common.python.chapter_compare_template import discover_benchmarks, load_benchmark

# Discover and run benchmarks
harness = BenchmarkHarness()
benchmarks = discover_benchmarks(chapter_dir)
for baseline_path, optimized_paths, name in benchmarks:
    benchmark = load_benchmark(baseline_path, optimized_paths[0])
    results = harness.benchmark(benchmark)
```

#### Compilation Utilities
```python
from common.python.compile_utils import enable_tf32

# Enable TF32 precision
enable_tf32()
```

## Benefits

1. **Consistency**: All chapters use the same profiling methodology
2. **Maintainability**: Bug fixes and improvements propagate to all chapters
3. **Pedagogy**: Students see the same patterns across all examples
4. **Quality**: Professional-grade error checking and profiling built-in

## Adding New Utilities

To add new common utilities:

1. Add header files to `headers/`
2. Add Python modules to `python/`
3. Update this README with usage examples
4. Test with at least one chapter before rolling out
