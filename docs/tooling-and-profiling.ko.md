# 도구 및 프로파일링 가이드

이 가이드는 레포지토리의 프로파일링, 도구, 자동화 정보를 통합합니다. 이전에 루트 `README.md`에 있던 상세한 지침을 보존하면서 최상위 README는 높은 수준의 이야기에 집중하도록 유지합니다.

## 자동화된 프로파일링 하네스

모든 챕터 예제는 통합 하네스를 통해 프로파일링할 수 있습니다:

```bash
# 사용 가능한 모든 예제 나열
python scripts/profile_harness.py --list

# 여러 도구로 특정 예제 프로파일링
python scripts/master_profile.py ch10_warp_specialized_pipeline --profile nsys ncu

# 모든 예제 프로파일링
python scripts/profile_harness.py --profile all

# 단축 래퍼
./start.sh    # 모든 챕터에서 하네스 실행
./stop.sh     # 활성 실행 종료
./clean_profiles.sh  # 축적된 아티팩트 제거
```

## NVIDIA Nsight Systems (타임라인 분석)

```bash
nsys profile -t cuda,nvtx,osrt,triton -o timeline_profile python script.py
```

**주요 메트릭**

- GPU 활용률 타임라인
- 메모리 전송 패턴
- 커널 실행 오버헤드
- CUDA 스트림 오버랩
- 멀티 GPU 통신 패턴

## NVIDIA Nsight Compute (커널 분석)

```bash
ncu --metrics achieved_occupancy,warp_execution_efficiency -o kernel_profile python script.py
```

**주요 메트릭**

- 달성 점유율(Achieved occupancy)
- 워프 실행 효율성
- 메모리 처리량
- 연산 활용도
- 레지스터 사용량
- SM % 최대 활용도
- DRAM 처리량
- L2 히트율

## Holistic Tracing Analysis (HTA)

```bash
nsys profile -t cuda,nvtx,osrt,cudnn,cublas,nccl,triton -o hta_profile python script.py
```

**주요 메트릭**

- 멀티 GPU 통신 패턴
- NCCL 집합 연산 효율성
- GPU 간 로드 밸런싱
- 메모리 대역폭 분포
- GPU 간 동기화

## PyTorch Profiler (프레임워크 수준)

```python
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True,
    with_stack=True,
    with_flops=True,
    profile_memory=True,
) as prof:
    # 여기에 코드
```

**주요 메트릭**

- 연산자 실행 시간
- 메모리 할당 패턴
- FLOP 카운트
- 호출 스택 분석
- 모듈 수준 성능

## perf (시스템 수준 분석)

```bash
perf record -g -p $(pgrep python) -o perf.data
perf report -i perf.data
```

**주요 메트릭**

- CPU 활용률
- 캐시 미스율
- 시스템 콜 오버헤드
- 메모리 액세스 패턴

## 프로파일링 출력 분석

책 테이블 및 심층 분석을 위한 메트릭 추출:

```bash
# Nsight Compute 메트릭
python tools/extract_ncu_subset.py 'output/reports/*.csv'

# Nsight Systems 요약
python tools/extract_nsys_summary.py 'output/traces/*.nsys-rep'

# PyTorch 프로파일러 데이터
python tools/extract_pytorch_profile.py 'profiles/*/pytorch_*/*'

# 또는 자동화된 추출 스크립트 실행
./extract.sh
```

## 예제 프로파일링 워크플로우

### 1. 기본 성능 분석

```bash
# 성능 기초 실행
python3 code/ch1/performance_basics.py

# Nsight Systems로 프로파일링
nsys profile -t cuda,nvtx,osrt -o perf_basics python3 code/ch1/performance_basics.py

# 결과 분석
nsys stats perf_basics.nsys-rep
```

### 2. 하드웨어 최적화

```bash
# 하드웨어 기능 확인
python3 code/ch2/hardware_info.py

# NUMA 바인딩 테스트
python3 code/ch3/bind_numa_affinity.py

# 메모리 액세스 패턴 프로파일링
ncu --metrics memory_throughput -o memory_profile python3 code/ch7/memory_optimization.py
```

### 3. PyTorch 컴파일 분석

```bash
# torch.compile 성능 테스트
python3 code/ch14/torch_compiler_examples.py

# 컴파일 오버헤드 프로파일링
nsys profile -t cuda,nvtx,osrt -o compile_profile python3 code/ch14/torch_compiler_examples.py

# 커널 성능 분석
ncu --metrics achieved_occupancy -o kernel_profile python3 code/ch14/torch_compiler_examples.py
```

### 4. Triton 커널 개발

```bash
# Triton 커널 테스트
python3 code/ch14/triton_examples.py

# 커스텀 커널 프로파일링
ncu --metrics triton_kernel_efficiency -o triton_profile python3 code/ch14/triton_examples.py
```

## 도구 및 유틸리티

### 필수 도구 디렉토리 (`tools/`)

- `extract_ncu_subset.py`: 원고 테이블용 Nsight Compute CSV 메트릭 수집
- `extract_nsys_summary.py`: Nsight Systems 타임라인 요약 추출
- `extract_pytorch_profile.py`: PyTorch 프로파일러 출력 데이터 처리

### 아카이브 디렉토리 (`archive/`)

- `build_all.sh`: CUDA 샘플 빌드 및 Python 구문 검증
- `update_blackwell_requirements.sh`: 모든 requirements 파일 업데이트
- `update_cuda_versions.sh`: Blackwell을 위한 Makefile 정규화
- `comprehensive_profiling.py`: 모든 프로파일링 도구 데모
- `generate_example_inventory.py`: 챕터별 카탈로그 생성
- `run_all_examples.sh`: 모든 챕터 예제 실행
- `compare_nsight/`: Nsight Systems vs Compute 비교 도구
- `clean_profiles.sh`: 축적된 프로파일링 아티팩트 정리
- `assert.sh`: 추출 도구에 대한 유용한 정보

### 스크립트 디렉토리 (`scripts/`)

- `profile_harness.py`: 모든 예제를 위한 통합 프로파일링 하네스
- `master_profile.py`: 다중 도구 지원이 있는 마스터 프로파일링 스크립트
- `example_registry.py`: 모든 챕터 예제 및 메타데이터 레지스트리
- `ncu_profile.sh`, `nsys_profile.sh`, `perf_profile.sh`: Nsight 및 perf를 위한 CLI 단축키

## 문제 해결

### 일반적인 이슈

1. **설정 스크립트 실패**

   ```bash
   # 권한 확인
   sudo ./setup.sh

   # GPU 감지 확인
   nvidia-smi
   ```

2. **CUDA 버전 불일치**

   ```bash
   # CUDA 버전 확인
   nvcc --version

   # PyTorch CUDA 지원 확인
   python3 -c "import torch; print(torch.cuda.is_available())"
   ```

3. **메모리 이슈**

   ```bash
   # 사용 가능한 메모리 확인
   free -h

   # GPU 메모리 모니터링
   nvidia-smi
   ```

4. **프로파일링 도구 이슈**

   ```bash
   # Nsight 설치 확인
   nsys --version
   ncu --version

   # 프로파일링 권한 확인
   sudo sysctl kernel.perf_event_paranoid=1
   ```
