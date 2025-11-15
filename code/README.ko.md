# AI Systems Performance Engineering: Code

멀티 GPU Blackwell 시스템에서 PyTorch/CUDA/Triton 워크로드를 검증, 튜닝, 문서화하기 위한 프로덕션 플레이북입니다.

---

## 1. 개요
- **타겟 하드웨어**: NVIDIA Blackwell (B200/B300, sm100), Grace Blackwell (GB200/GB300, sm103), DGX Spark (GB10, sm121).
- **참조 스택**: CUDA 13+, PyTorch 2.10-dev+, Triton 3.5+, Python 3.10+.
- **포함 내용**: 20개 챕터의 베이스라인/최적화 벤치마크 쌍, 하네스 도구, 프로파일링 스크립트, 워크로드 구성, 프로덕션 준비 설정 자동화.
- **주요 목표**: 일관된 랩 머신 구축, 비교 가능한 아티팩트(JSON/MD/CSV) 캡처, 커널 진화에 따라 성능 입증(PoB) 보고서를 최신 상태로 유지.

### 챕터별 개념

| 챕터 | 다루는 개념 | 예제 수 |
|------|------------|---------|
| **ch1** | 성능 기초, 프로파일링, 메모리 관리, CUDA Graph, 배치 연산, roofline, 산술 강도 | 21 |
| **ch2** | GPU 아키텍처, NVLink, CPU-GPU 일관성, 계층 구조 검사 | 6 |
| **ch3** | 시스템 튜닝 (NUMA, governor, THP, IRQ, Docker/K8s) | 11 |
| **ch4** | 멀티 GPU 학습, NCCL, 텐서/파이프라인 병렬성, NVSHMEM | 7 |
| **ch5** | 스토리지/IO 최적화, GDS, DataLoader 튜닝, cuFile | 7 |
| **ch6** | CUDA 기초, 스레드 계층 구조, 점유율, 통합 메모리 | 14 |
| **ch7** | 메모리 액세스 패턴, 병합, 타일링, 공유 메모리 대역폭 | 20 |
| **ch8** | 점유율 튜닝, ILP, 루프 언롤링, 워프 분기, 리소스 제한 | 13 |
| **ch9** | 산술 강도, roofline, 커널 퓨전, CUTLASS | 17 |
| **ch10** | Tensor Core, TMA, 비동기 파이프라인, 워프 특화, 클러스터 | 17 |
| **ch11** | CUDA 스트림, 동시성, Hyper-Q, 오버랩 최적화 | 11 |
| **ch12** | CUDA Graph, 조건부 그래프, 동적 실행 | 14 |
| **ch13** | PyTorch 프로파일링, 메모리 분석, FSDP, 양자화, 할당자 | 23 |
| **ch14** | `torch.compile`, Triton 커널, FP8, 커스텀 코드 생성 | 5 |
| **ch15** | 분리형 추론, KV 캐시, 연속 배치 | 10 |
| **ch16** | 추론 최적화, 프로덕션 서빙, 추측적 디코딩 | 8 |
| **ch17** | 동적 라우팅, roofline 트레이드오프, 지연시간 vs 정확도 | 13 |
| **ch18** | 고급 어텐션 (Flash/Flex/MLA/Paged) | 18 |
| **ch19** | 저정밀도 학습 (FP4/FP6/FP8), Transformer Engine | 15 |
| **ch20** | 엔드투엔드 최적화 사례 연구 | 14 |

### 주제

1. **성능 기초 (ch1–ch3)** – 프로파일링, 하드웨어 인식, 시스템 튜닝.
2. **커널 최적화 (ch6–ch10, ch12)** – 점유율, ILP, 비동기 파이프라인, CUDA Graph.
3. **메모리 전략 (ch7, ch19)** – 병합, 타일링, 정밀도 감소.
4. **병렬성 및 분산 (ch4, ch11, ch13, ch15–ch17)** – 멀티 GPU, 스트림, 서빙.
5. **PyTorch/Triton 가속화 (ch13–ch14)** – 컴파일된 autograd, Triton 타일링.
6. **어텐션 및 고급 워크플로우 (ch18–ch20)** – 현대적 어텐션 디자인, 프로덕션 플레이북.

---

## 2. 시작하기

### 요구사항
- `setup.sh`를 위한 루트 권한 (NVIDIA 드라이버 580+, CUDA 13.0+, Nsight 도구 설치).
- 호스트에서 Python 3.10+ 사용 가능.
- 지원되는 Blackwell GPU 최소 1개.
- 패키지 설치 또는 transformer-engine 휠 다운로드를 위한 네트워크 액세스.

### 설정
```bash
git clone <repo-url>
cd ai-performance-engineering/code
sudo ./setup.sh
```
드라이버가 업그레이드된 경우, 재부팅 후 `sudo ./setup.sh`를 다시 실행하여 검증을 완료하세요.

### Transformer Engine 휠
CUDA/PyTorch 버전이 변경될 때 재빌드:
```bash
scripts/build_transformer_engine_wheel.sh v2.8.0+40c69e7
split -b 50M --numeric-suffixes=0 --suffix-length=2 \
  third_party/TransformerEngine/dist/transformer_engine-2.8.0+40c69e7-cp311-cp311-linux_aarch64.whl \
  vendor/wheels/transformer_engine-2.8.0+40c69e7-cp311-cp311-linux_aarch64.whl.part
split -b 50M --numeric-suffixes=0 --suffix-length=2 \
  third_party/TransformerEngine/dist/transformer_engine_cu12-2.8.0+40c69e7-cp311-cp311-linux_aarch64.whl \
  vendor/wheels/transformer_engine_cu12-2.8.0+40c69e7-cp311-cp311-linux_aarch64.whl.part
cp third_party/TransformerEngine/transformer_engine/pytorch/dist/transformer_engine_torch-2.8.0+40c69e7-cp311-cp311-linux_aarch64.whl \
   vendor/wheels/transformer_engine_torch-2.8.0+40c69e7-cp311-cp311-linux_aarch64.whl
```
`setup.sh`가 자동으로 휠을 재조립하고 설치합니다.

### 설치 확인
1. `nvidia-smi` – GPU 가시성 및 드라이버 ≥ 580 확인.
2. `python tools/cli/benchmark_cli.py verify` – import 및 구문 검사 (기본적으로 모든 챕터 실행; 범위 지정 실행을 위해 `--targets ch3 --targets ch4:resnet_50` 추가).
3. 준비가 되면 타겟 벤치마크 제품군 실행:
   `python tools/cli/benchmark_cli.py run --targets ch1 --artifacts-dir ./artifacts`.

### 레포지토리 레이아웃
```text
code/
├── setup.sh
├── ch1...ch20/          # README와 벤치마크가 포함된 챕터 워크스루
├── common/              # 하네스, 로깅, 워크로드 구성
├── tools/               # 검증, 분석, 벤치마킹 헬퍼
├── scripts/             # 프로파일링/프로빙 유틸리티
└── tests/               # 통합 테스트
```

### 워크플로우 체크리스트
1. 공유 워크로드 구성으로 베이스라인/최적화 쌍 정의.
2. 타이밍 전에 커널 워밍업 및 그래프 컴파일 캐시.
3. 프로덕션 수치를 위한 표준 실행 캡처 (실험 시에만 `--iterations/--warmup` 조정).
4. 타임스탬프가 지정된 폴더 아래 아티팩트 기록 (`benchmark_test_results.json`/`.md`, 로그).
5. 새 아티팩트 캡처 후 `tools/analysis/analyze_expectations.py` 실행.
6. 재현성을 위해 발견 사항 (속도 향상, 처리량, 적용된 최적화) 문서화.

---

## 3. 벤치마크 워크플로우

### 메인 CLI (Typer)
구조화된 실행을 위한 권장 진입점:
```bash
# 전체 제품군
python tools/cli/benchmark_cli.py run

# 단일 챕터
python tools/cli/benchmark_cli.py run --targets ch12 --artifacts-dir ./artifacts

# 커스텀 옵션
python tools/cli/benchmark_cli.py run --targets ch10 --timeout-multiplier 2.0 --reproducible --cold-start
python tools/cli/benchmark_cli.py run --targets ch18 --profile  # Nsight/torch 프로파일러 추적 수집
```

### 레거시 러너 / 타겟 예제
`tools/testing/run_all_benchmarks.py`는 모든 `baseline_*.py` 파일을 검색하고, 일치하는 최적화 구현과 쌍을 이루며, PoB 친화적 요약을 생성합니다.
```bash
PYTHONPATH=. python tools/testing/run_all_benchmarks.py --targets ch10:cluster_group_no_dsmem
PYTHONPATH=. python tools/testing/run_all_benchmarks.py --targets all --timeout-multiplier 2.0
```
아티팩트는 `artifacts/<timestamp>/<example>/results/benchmark_test_results.{json,md}`에 매니페스트, 구성 스냅샷, 환경 로그와 함께 저장됩니다.

### 옵션 및 제어
- `--timeout-multiplier`, `--suite-timeout` – 실행 시간 연장 또는 제한.
- `--reproducible` – 결정론적 시드/알고리즘 강제; 느린 폴백 커널과 결정론적 지원이 없는 연산은 실패할 수 있음.
- `--cold-start` – 벤치마크 간 추가 정리 (가비지 컬렉션 및 CUDA 컨텍스트 재설정 포함).
- `--profile/--no-profile` – 필요할 때 Nsight/Torch 추적 옵트인.
- `--iterations`, `--warmup` – 빠른 로컬 실험이 필요할 때 기본 20/5 샘플링 재정의.

### 보고 및 성능 입증
1. 원하는 벤치마크를 실행하고 아티팩트 저장.
2. `python tools/analysis/analyze_expectations.py --artifacts-dir artifacts --output-csv reports/proof_of_benefit.csv` 실행 (또는 다른 기대치 보고서 경로로 `--output-csv` 지정).
3. 생성된 CSV에서 실패/회귀로 플래그가 지정된 행을 검토하여 저장된 기대치 아래로 떨어진 벤치마크 식별.

### 통합 테스트
```bash
pytest tests/integration
```
벤치마크 검색, 하네스 실행 경로, 프로파일링 토글, 실패 처리를 엔드투엔드로 검증합니다.

### 다중 메트릭 비교
```python
from common.python.benchmark_comparison import compare_and_display_all_metrics
summary = compare_and_display_all_metrics(
    baseline_result=baseline,
    optimized_result=optimized,
    name="My Benchmark",
    chapter="ch7",
    include_raw_metrics=False,
)
```
백분율 개선은 델타 계산을 사용합니다 (낮을수록 좋음 및 높을수록 좋음이 자동으로 처리됨). 타이밍, 메모리, 프로파일러 메트릭을 테이블 또는 요약으로 렌더링할 수 있습니다.

### 최대 성능 검증
`setup.sh` (및 캐시된 결과가 없을 때 CLI)는 `tools/benchmarking/benchmark_peak.py`를 실행하여 측정된 TFLOPS, 대역폭, NVLink, torch.compile 베이스라인을 캡처합니다. 수동으로 재실행하려면:
```bash
python tools/benchmarking/benchmark_peak.py
```
결과는 `performance_targets.py`에 전달되어 챕터가 정적 기대치 대신 측정된 상한선과 비교할 수 있도록 합니다.

---

## 4. 도구 및 진단

- **벤치마크 하네스 가이드**: 아키텍처, 매니페스트 캡처, 프로파일링 후크, 로깅 세부사항은 `docs/benchmark_harness_guide.md` 참조.
- **유틸리티 프로브**: `tools/tcgen05_probe.cu`는 tcgen05 텐서 코어를 위한 최소 CUTLASS 4.2 드라이버입니다. NVCC 또는 벤더 제공 CUTLASS CMake 래퍼로 직접 빌드하여 툴체인 지원 확인.
- **스크립트 폴더**: 재사용 가능한 Nsight 워크플로우, 성능 스위프, GPU 복원력 헬퍼.
- **공통 헬퍼**: `common/python`은 모든 챕터가 공유하는 워크로드 구성, CUDA 바이너리 래퍼, NVCC 아키텍처 감지, 로깅, 실행 매니페스트를 호스팅합니다.

---

## 5. 유지 관리 및 문제 해결

### 정리
생성된 바이너리, 아티팩트, 캐시 제거:
```bash
python cleanup.py
```

### GPU 재설정
```bash
sudo ./reset-gpu.sh
```
실행 중인 프로세스를 중지하고, 영속성 모드를 토글하며, NVIDIA 커널 모듈을 다시 로드하고, PCIe/NVML 재설정을 수행합니다. 루트 권한이 필요합니다.

### 다음 단계
- 커널이 의미 있게 변경될 때마다 새 아티팩트 캡처.
- 재현성을 위해 벤치마크 차이와 관련 아티팩트 번들을 포함한 이슈 또는 PR 제출.

---

## 플랫폼 주의사항 및 특수사항

### GB10 / SM121 tcgen05 지원 격차
- GB10 (SM 12.1)의 CUDA 13.0 + 드라이버 580.95는 CUTLASS tcgen05 로우어링에 필요한 멀티캐스트/TMA 기능이 부족합니다.
- `ptxas`가 `Feature '.multicast::cluster::all' not supported on .target 'sm_121'`로 실패하므로 cubin/SASS가 생성되지 않습니다.
- 해결 방법: SM100+/SM103 tcgen05를 기본적으로 노출하는 하드웨어 (B200/B300)에서 실행하거나 SM121에서 기능을 활성화하는 CUDA/펌웨어 드롭을 기다립니다. 그때까지 tcgen05 예제는 건너뜁니다.

### DSMEM 없는 스레드 블록 클러스터 (ch10)
- GB10/SM121은 스레드 블록 클러스터를 노출하지만 분산 공유 메모리(DSMEM)는 노출하지 않으므로 DSMEM이 활성화된 클러스터 커널을 실행할 수 없습니다.
