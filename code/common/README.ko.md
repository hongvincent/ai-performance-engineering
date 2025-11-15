# 공통 인프라

모든 챕터 예제가 일관된 빌드 시스템, 프로파일링 워크플로우, 벤치마킹 방법론을 보장하기 위한 공유 유틸리티입니다.

## 디렉토리 구조

```
common/
├── headers/
│   ├── arch_detection.cuh      # GPU 아키텍처 감지 및 제한
│   └── tma_helpers.cuh         # Tensor Memory Accelerator 유틸리티
└── python/
    ├── benchmark_harness.py    # 프로덕션급 벤치마킹 하네스
    ├── chapter_compare_template.py  # 챕터 compare.py를 위한 표준 템플릿
    ├── compile_utils.py        # torch.compile 및 정밀도 유틸리티
    └── env_defaults.py         # 환경 구성 헬퍼
```

## 사용법

### 빌드 시스템

챕터 Makefile에서:

```makefile
# 공통 아키텍처 플래그 포함
include ../common/cuda_arch.mk
```

이는 아키텍처 감지 및 이중 아키텍처 빌드 지원(sm_100 + sm_121)을 제공합니다.

### CUDA 헤더

#### 아키텍처 감지
```cpp
#include "../../common/headers/arch_detection.cuh"

int main() {
    // GPU 기능 쿼리
    const auto& limits = cuda_arch::get_architecture_limits();

    // 기능 확인
    if (limits.supports_clusters) {
        printf("Cluster size: %d\n", limits.max_cluster_size);
    }
    if (limits.has_grace_coherence) {
        printf("Grace-Blackwell coherence available\n");
    }

    // 최적 타일 크기 선택
    auto tile = cuda_arch::select_tensor_core_tile();
    printf("Tensor core tile: %dx%dx%d\n", tile.m, tile.n, tile.k);

    // TMA 제한 가져오기
    auto tma = cuda_arch::get_tma_limits();
    printf("TMA 2D box: %ux%u\n", tma.max_2d_box_width, tma.max_2d_box_height);

    return 0;
}
```

#### TMA (Tensor Memory Accelerator) 헬퍼
```cpp
#include "../../common/headers/arch_detection.cuh"
#include "../../common/headers/tma_helpers.cuh"

int main() {
    // TMA 지원 확인
    if (!cuda_tma::device_supports_tma()) {
        printf("TMA not supported (requires SM 9.0+)\n");
        return 1;
    }

    // 텐서 맵 생성
    CUtensorMap desc;
    auto encode = cuda_tma::load_cuTensorMapEncodeTiled();
    bool ok = cuda_tma::make_2d_tensor_map(
        desc, encode, d_data, width, height, ld,
        box_width, box_height, CU_TENSOR_MAP_SWIZZLE_NONE);

    // cp_async_bulk_tensor 연산과 함께 커널에서 사용
    return 0;
}
```

### Python 유틸리티

#### 환경 구성
```python
from common.python.env_defaults import apply_env_defaults, dump_environment_and_capabilities

# 기본 환경 설정 적용
apply_env_defaults()

# 환경 및 하드웨어 기능 출력
dump_environment_and_capabilities()
```

#### 벤치마킹
```python
from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode, BenchmarkConfig
from common.python.chapter_compare_template import discover_benchmarks, load_benchmark

# 벤치마크 검색 및 실행
harness = BenchmarkHarness()
benchmarks = discover_benchmarks(chapter_dir)
for baseline_path, optimized_paths, name in benchmarks:
    benchmark = load_benchmark(baseline_path, optimized_paths[0])
    results = harness.benchmark(benchmark)
```

#### 컴파일 유틸리티
```python
from common.python.compile_utils import enable_tf32

# TF32 정밀도 활성화
enable_tf32()
```

## 이점

1. **일관성**: 모든 챕터가 동일한 프로파일링 방법론 사용
2. **유지보수성**: 버그 수정 및 개선 사항이 모든 챕터로 전파됨
3. **교육적**: 학생들이 모든 예제에서 동일한 패턴을 볼 수 있음
4. **품질**: 프로페셔널급 에러 검사 및 프로파일링이 내장됨

## 새로운 유틸리티 추가

새로운 공통 유틸리티를 추가하려면:

1. `headers/`에 헤더 파일 추가
2. `python/`에 Python 모듈 추가
3. 사용 예제로 이 README 업데이트
4. 배포하기 전에 최소 하나의 챕터에서 테스트
