# 환경 및 구성

이 문서는 이전에 최상위 `README.md`의 일부였던 환경, 도구, 시스템 설정 세부 사항을 통합합니다. 레포지토리의 NVIDIA Blackwell 중심 스택에 시스템을 맞추기 위한 참조 자료로 사용하세요.

## 타겟 아키텍처

이 레포지토리는 **NVIDIA Blackwell B200/B300 (SM100)**을 타겟팅합니다. 모든 스크립트, 커널, 구성은 다음을 가정합니다:

- CUDA 12.9 툴킷 및 툴체인
- PyTorch 2.9.0 (cu129 nightly 빌드)
- Triton 3.5.0

## 핵심 구성요소

- `arch_config.py`는 아키텍처 결정을 중앙 집중화하고 모든 것을 Blackwell로 정규화합니다.
- `build_all.sh`는 `sm_100`으로 CUDA 커널을 컴파일하고 챕터 전체에서 Python 구문을 검증합니다.
- 챕터 요구사항 파일은 `requirements_latest.txt`를 통해 조화됩니다.

## 툴체인 요구사항

| 구성요소 | 버전 / 채널 | 비고 |
|---------|------------|------|
| CUDA Toolkit | 12.9 (nvcc 12.9.x) | 모든 곳에서 `nvcc -arch=sm_100` |
| PyTorch | 2.9.0 (cu129 nightly) | `https://download.pytorch.org/whl/nightly/cu129`에서 설치 |
| Triton | 3.4.0 | 챕터 14 및 16의 Triton 커널에 필요 |
| Nsight Systems | 2024.6+ | 프로파일링 스크립트에서 사용 |
| Nsight Compute | 2024.3+ | 커널 수준 프로파일링 |

## 시스템 검증 및 실패 분석

`assert.sh` 스크립트는 심층 검증을 수행합니다:

```bash
./assert.sh
```

다음을 확인합니다:

- 시스템 의존성 (Python, CUDA, Nsight 도구, `numactl`, `perf`)
- GPU 가용성 및 상태
- PyTorch 및 CUDA 버전 호환성
- 예제 레지스트리 커버리지 (84개 예제)
- 하네스 전체의 빌드/스모크/프로파일링 실패
- 프로파일링 하네스 드라이 런

샘플 출력:

```
🚨 Recent Profile Session Analysis:
  Latest session: 20250928_182258
  📊 Results Summary:
    build: 83/84 successful (1 failed)
    smoke: 80/83 successful (3 failed)
    nsys: 1/80 successful (79 failed)
    ncu: 15/80 successful (65 failed)
    pytorch_full: 38/38 successful (0 failed)
```

## 환경 변수

```bash
# CUDA 최적화
export CUDA_LAUNCH_BLOCKING=0
export CUDA_CACHE_DISABLE=0

# NCCL 최적화
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0

# PyTorch 최적화
export TORCH_CUDNN_V8_API_ENABLED=1
export TORCH_CUDNN_V8_API_DISABLED=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,expandable_segments:True

# CUDA 경로
export CUDA_HOME=/usr/local/cuda-12.9
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

## 하드웨어 요구사항

- **GPU**: NVIDIA B200/B300 (Blackwell) 또는 호환 가능한 GPU
- **메모리**: 32GB+ 시스템 RAM 권장
- **스토리지**: 50GB+ 여유 공간
- **운영 체제**: Ubuntu 22.04+ (다른 Linux 배포판도 작동할 수 있음)

## 개발 환경

권장 개발자 도구:

```bash
# 개발 의존성 설치
pip3 install black flake8 mypy

# 코드 포맷팅
black code/

# 코드 린트
flake8 code/

# 타입 검사
mypy code/
```

## 고급 유틸리티

`archive/` 디렉토리에는 다음을 포함한 더 고급 오케스트레이션 기능이 있습니다:

- `update_blackwell_requirements.sh`: 챕터 요구사항을 최신 Blackwell 스택에 동기화
- `update_cuda_versions.sh`: Makefile 정규화
- `comprehensive_profiling.py`: 프로파일링 도구를 함께 사용하는 방법 시연
- `clean_profiles.sh`: 축적된 프로파일러 아티팩트 제거

프로파일링 제품군 및 자동화 워크플로우에 대한 자세한 내용은 `docs/tooling-and-profiling.md`를 참조하세요.
