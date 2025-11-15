# 벤치마크 하네스: 파워와 아키텍처

## 개요

`BenchmarkHarness`는 20개 챕터에 걸쳐 264개 이상의 베이스라인/최적화 예제 쌍의 **자동 검색, 실행, 프로파일링, 비교**를 제공하는 프로덕션급 벤치마킹 프레임워크입니다. 포괄적인 타임아웃 보호, 프로파일링 통합, 재현성 보장을 통해 실제 GPU 워크로드를 안정적으로 처리하도록 설계되었습니다.

---

## 핵심 아키텍처

### 간단한 프로토콜, 강력한 실행

하네스는 벤치마크가 구현해야 하는 **최소한의 프로토콜**을 사용합니다:

```python
class Benchmark(Protocol):
    def setup(self) -> None:          # 모델, 데이터 등 초기화
    def benchmark_fn(self) -> None:    # 측정할 코드
    def teardown(self) -> None:        # 정리
    def get_config(self) -> Optional[BenchmarkConfig]:  # 선택적 재정의
    def validate_result(self) -> Optional[str]:  # 선택적 검증
```

**강력함**: 벤치마크는 로직에만 집중합니다. 하네스가 처리하는 것:
- ✅ 정확한 타이밍 (CUDA Event, Triton do_bench, PyTorch Timer)
- ✅ 통계 분석 (평균, 중앙값, 표준편차, 백분위수)
- ✅ 메모리 추적
- ✅ 프로파일링 통합 (nsys, ncu, PyTorch profiler)
- ✅ 타임아웃 보호
- ✅ 에러 처리 및 복구
- ✅ 재현성 (시드, 결정론적 모드)
- ✅ GPU 상태 관리

---

## 주요 기능

### 1. **자동 벤치마크 검색**

하네스는 간단한 명명 규칙을 사용하여 벤치마크를 자동으로 검색합니다:

```
ch*/baseline_*.py  →  ch*/optimized_*.py
```

**예제**: `baseline_moe.py`는 `optimized_moe.py` 또는 `optimized_moe_sparse.py`와 쌍을 이룹니다

**검색 로직**:
- 모든 챕터 디렉토리 스캔 (`ch1` ~ `ch20`)
- `baseline_*.py` 파일 찾기
- `optimized_{name}*.py` 파일과 매칭
- 예제 이름 추출 (예: `baseline_moe_dense.py` → `moe`)
- 튜플 반환: `(baseline_path, [optimized_paths], example_name)`

**결과**: 단일 명령으로 **20개 챕터**에 걸쳐 **264개 벤치마크** 실행:
```bash
python tools/cli/benchmark_cli.py run
```

---

### 2. **다중 실행 모드**

#### 서브프로세스 모드 (기본값 - 프로덕션급)

**왜 서브프로세스?**
- ✅ **진정한 격리**: 각 벤치마크가 별도의 프로세스에서 실행
- ✅ **신뢰할 수 있는 타임아웃**: 중단된 프로세스를 종료 가능 (CUDA 커널은 Python에서 중단 불가)
- ✅ **깨끗한 GPU 상태**: 벤치마크 간 오염 없음
- ✅ **충돌 보호**: 하나의 벤치마크 충돌이 전체 제품군을 종료하지 않음

**작동 방식**:
1. 벤치마크 클래스와 구성을 JSON으로 직렬화
2. `isolated_runner.py`를 통해 격리된 Python 서브프로세스 생성
3. 서브프로세스가 모듈 import, 벤치마크 인스턴스화, 하네스 실행
4. JSON을 통해 결과 반환 (Pydantic 모델)
5. 부모 프로세스가 타임아웃 및 정리 처리

**타임아웃 처리**: 신뢰할 수 있는 종료를 위해 프로세스 그룹 종료(`os.killpg`)와 함께 `subprocess.communicate(timeout=...)` 사용.

#### 스레딩 모드 (폴백)

**사용 시기**:
- 모듈 파일 경로를 확인할 수 없을 때
- 서브프로세스 격리를 사용할 수 없을 때
- 빠른 프로토타이핑

**제한사항**:
- ⚠️ 중단된 CUDA 커널을 강제 중지할 수 없음
- ⚠️ 격리 수준 낮음 (메모리 공간 공유)
- ⚠️ 타임아웃 적용이 덜 신뢰적

---

### 3. **포괄적인 타임아웃 보호**

하네스는 중단을 방지하기 위해 **단계별 타임아웃**을 제공합니다:

| 단계 | 기본 타임아웃 | 목적 |
|------|--------------|------|
| **Setup** | 30s | 모델 로딩, CUDA 확장 컴파일, torch.compile() |
| **Warmup** | 15s | GPU 워밍업, cuDNN 자동 튜닝 |
| **Measurement** | 15s | 실제 벤치마크 반복 |
| **Profiling** | 180s | nsys/ncu 프로파일링 (느릴 수 있음) |

**타임아웃 배수**: 느린 시스템을 위해 `timeout_multiplier=2.0`로 모든 타임아웃 조정.

**타임아웃 동작**:
- 구조화된 타임아웃 결과 생성 (단순 실패가 아님)
- 상세한 진단 로깅 (타임아웃된 단계, 경과 시간, 제안사항)
- GPU 리소스 자동 정리
- 나머지 벤치마크 계속 진행

**타임아웃 결과 예제**:
```json
{
  "timeout_stage": "measurement",
  "timeout_duration_seconds": 45.2,
  "timeout_limit_seconds": 15,
  "errors": ["TIMEOUT: Benchmark measurement stage exceeded timeout..."],
  "watchdog": {
    "setup": {"status": "completed", "duration": 2.1},
    "warmup": {"status": "completed", "duration": 1.5},
    "measurement": {"status": "timeout", "duration": 45.2}
  }
}
```

---

### 4. **통합 프로파일링**

하네스는 **세 가지 프로파일링 도구**를 원활하게 통합합니다:

#### Nsight Systems (nsys)
- **목적**: 타임라인 프로파일링 (CPU/GPU 활동, 메모리 전송, 커널 실행)
- **출력**: `nsys-ui`에서 볼 수 있는 `.nsys-rep` 파일
- **활성화 시**: `enable_nsys=True` (기본값: True)
- **타임아웃**: 벤치마크당 120초

#### Nsight Compute (ncu)
- **목적**: 커널 수준 메트릭 (SM 활용도, 메모리 처리량, 점유율)
- **출력**: `ncu-ui`에서 볼 수 있는 `.ncu-rep` 파일
- **활성화 시**: `enable_ncu=True` (기본값: True)
- **타임아웃**: 벤치마크당 180초

#### PyTorch Profiler
- **목적**: Python 수준 프로파일링 (연산자 수준 타이밍, 메모리 사용량)
- **출력**: Chrome trace JSON (`chrome://tracing`에서 볼 수 있음)
- **활성화 시**: `enable_profiling=True` (기본값: True)

**자동 NVTX**: 프로파일링이 활성화되면 더 나은 추적 시각화를 위해 NVTX 마커가 자동으로 추가됩니다.

**프로파일링 오케스트레이션**:
- 타이밍과 **함께** 프로파일링 실행 (별도 실행 아님)
- 메트릭 자동 캡처 (SM 처리량, 메모리 대역폭 등)
- 구조화된 출력 디렉토리에 아티팩트 저장
- 주요 메트릭을 `ProfilerMetrics` Pydantic 모델로 추출

**결과**: 모든 벤치마크 실행이 생성하는 것:
- 타이밍 통계 (평균, 중앙값, p99 등)
- 메모리 통계 (피크, 할당됨, 예약됨)
- 프로파일러 아티팩트 (`.nsys-rep`, `.ncu-rep`, `.json`)
- 프로파일러 메트릭 (SM 활용도, 대역폭, 점유율)

---

### 5. **재현성 보장**

하네스는 구성 시 **비트 단위 재현성**을 보장합니다:

#### 시드 관리
```python
config = BenchmarkConfig(
    seed=42,                    # 모든 랜덤 시드 설정
    deterministic=True          # 결정론적 알고리즘 활성화
)
```

**시드가 설정되는 것**:
- Python `random` 모듈
- NumPy 랜덤 상태
- PyTorch 랜덤 상태
- CUDA 랜덤 상태 (모든 GPU)

**결정론적 모드**:
- `torch.use_deterministic_algorithms(True)` - 느리지만 재현 가능한 알고리즘 사용
- `torch.backends.cudnn.deterministic = True` - cuDNN 자동 튜닝 비활성화
- **트레이드오프**: 느린 커널로 폴백 (보통 5-20% 성능 저하), 결정론적 지원이 없는 연산은 런타임에 오류 발생

#### 실행 매니페스트

모든 벤치마크 실행은 **완전한 매니페스트**를 캡처합니다:

```python
harness.benchmark_with_manifest(benchmark, run_id="my_run")
# BenchmarkRun을 반환:
# - Manifest: 하드웨어, 소프트웨어, git 상태, 환경
# - Result: 타이밍, 메모리, 프로파일링 데이터
# - Metadata: Run ID, 타임스탬프, 구성
```

**매니페스트 캡처**:
- 하드웨어: GPU 모델, 연산 능력, 메모리, 클럭
- 소프트웨어: PyTorch 버전, CUDA 버전, 드라이버 버전, Triton 버전
- Git 상태: 커밋 해시, 브랜치, dirty 플래그
- 환경: Python 버전, OS, 환경 변수
- 시드: 사용된 모든 랜덤 시드 값
- 구성: 모든 벤치마크 구성 매개변수

**사용 사례**: 실행 간 매니페스트를 비교하여 "성능이 왜 변경되었는가?" 디버깅.

---

### 6. **메모리 추적**

자동 GPU 메모리 추적:

```python
config = BenchmarkConfig(enable_memory_tracking=True)
```

**추적 대상**:
- **피크 메모리**: 벤치마크 중 할당된 최대 메모리
- **할당된 메모리**: 현재 할당된 메모리
- **예약된 메모리**: PyTorch 할당자가 예약한 메모리

**메모리 컨텍스트 매니저**:
```python
with self._memory_tracking(config) as mem_result:
    # 벤치마크 실행
    times_ms = self._benchmark_without_profiling(fn, config)
# mem_result에 이제 MemoryStats 포함
```

**결과**: 모든 벤치마크가 메모리 사용량을 보고하여 다음을 가능하게 함:
- 메모리 최적화 분석
- OOM 디버깅
- 메모리 효율성 비교 (베이스라인 vs 최적화)

---

### 7. **통계 분석**

하네스는 **포괄적인 통계**를 계산합니다:

**타이밍 통계**:
- 평균, 중앙값, 표준편차
- 최소값, 최대값
- 백분위수: p25, p50 (중앙값), p75, p90, p95, p99
- `config.percentiles`를 통한 커스텀 백분위수

**추론 타이밍** (LLM 워크로드용):
- **TTFT** (Time To First Token): 평균, p50, p90, p95, p99
- **TPOT** (Time Per Output Token): 평균, p50, p90, p95, p99
- 요청 및 토큰 수

**원시 데이터 보존**: 모든 원시 타이밍 측정값이 커스텀 분석을 위해 `raw_times_ms`에 저장됨.

**결과 예제**:
```json
{
  "timing": {
    "mean_ms": 12.5,
    "median_ms": 12.3,
    "std_ms": 0.8,
    "p99_ms": 14.2,
    "iterations": 100,
    "raw_times_ms": [12.1, 12.3, 12.5, ...]
  }
}
```

---

### 8. **다중 벤치마킹 모드**

하네스는 **세 가지 타이밍 모드**를 지원합니다:

#### CUSTOM 모드 (기본값)
- GPU 타이밍을 위해 **CUDA Event** 사용 (가장 정확함)
- CPU 타이밍을 위해 `time.perf_counter()` 사용
- **장점**: 최소 오버헤드, 정확한 GPU 타이밍
- **최적**: 대부분의 GPU 벤치마크

#### TRITON 모드
- 내부적으로 `triton.testing.do_bench()` 사용
- **장점**: Triton 커널에 최적화됨
- **최적**: 커스텀 Triton 커널 벤치마크

#### PYTORCH 모드
- `torch.utils.benchmark.Timer` 사용
- **장점**: `min_run_time_ms`를 기반으로 자동 반복 횟수
- **최적**: PyTorch 연산자 벤치마크

**모드 선택**:
```python
harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM)
```

---

### 9. **자동 비교**

내장 비교 함수:

```python
from common.python.benchmark_harness import compare_benchmarks

result = compare_benchmarks(
    baseline=baseline_benchmark,
    optimized=optimized_benchmark,
    name="MoE Optimization",
    regression_threshold_pct=5.0
)
```

**반환**:
- 속도 향상 비율 (baseline / optimized)
- 회귀 감지 (최적화가 >5% 느린 경우)
- 두 벤치마크의 전체 통계
- 추가 분석을 위한 완전한 `BenchmarkResult` 객체

**사용 사례**: CI/CD에서 자동 성능 회귀 감지.

---

### 10. **GPU 상태 관리**

하네스는 벤치마크 간 **깨끗한 GPU 상태**를 보장합니다:

**자동 정리**:
- 벤치마크 간 CUDA 상태 재설정 (계단식 실패 방지)
- GPU 캐시 지우기 (`torch.cuda.empty_cache()`)
- 피크 메모리 통계 재설정
- 콜드 스타트 측정을 위한 `reset_gpu_state()`를 통한 GPU 재설정 처리

**콜드 스타트 모드**:
```python
config = BenchmarkConfig(enable_cleanup=True)  # 정리 강제
# 또는 CLI에서 cold_start 플래그 사용
```

**결과**: 각 벤치마크가 깨끗한 GPU 상태로 시작하여 공정한 비교 보장.

---

### 11. **에러 처리 및 복구**

**포괄적인 에러 처리**:

1. **타임아웃 에러**: 진단이 포함된 구조화된 타임아웃 결과
2. **실행 에러**: `errors` 목록에 캡처됨, 벤치마크 계속 진행
3. **검증 에러**: `validate_result()`를 통한 커스텀 검증
4. **프로파일링 에러**: 프로파일러를 사용할 수 없는 경우 우아한 폴백
5. **Import 에러**: 서브프로세스를 사용할 수 없는 경우 스레딩 모드로 폴백

**에러 보고**:
- `BenchmarkResult.errors` 목록에 에러 저장
- 컨텍스트가 포함된 상세 에러 메시지
- 디버깅을 위한 스택 추적
- 부분 결과 보존 (예: 타이밍이 실패해도 메모리 통계)

**복구**: 하나의 벤치마크 실패가 제품군을 중지하지 않음. 일부가 실패해도 모든 264개 벤치마크 실행 가능.

---

### 12. **구성 유연성**

**단일 진실 소스**: `BenchmarkDefaults`가 모든 기본값 제공.

**구성 계층**:
1. **BenchmarkDefaults** (하드코딩된 기본값)
2. **BenchmarkConfig** (인스턴스 수준 재정의)
3. **Benchmark.get_config()** (벤치마크별 재정의)
4. **CLI 플래그** (런타임 재정의)

**예제**:
```python
# 전역 기본값 (benchmark_defaults.py)
iterations = 100
warmup = 10

# 인스턴스 구성
config = BenchmarkConfig(iterations=50)  # 재정의

# 벤치마크별
class MyBenchmark(BaseBenchmark):
    def get_config(self):
        return BenchmarkConfig(iterations=25)  # 추가 재정의
```

**결과**: 코드 변경 없이 유연한 구성.

---

## 파워: 엔드투엔드 자동화

### 264개 벤치마크 모두 실행

**단일 명령**:
```bash
python tools/cli/benchmark_cli.py run
```

**실행 내용**:
1. ✅ 모든 `baseline_*.py` / `optimized_*.py` 쌍 검색
2. ✅ 각 벤치마크 모듈 동적 로딩
3. ✅ 베이스라인 → 최적화 비교 실행
4. ✅ 타이밍, 메모리, 프로파일링 데이터 수집
5. ✅ 속도 향상 및 통계 계산
6. ✅ JSON + Markdown 보고서 생성
7. ✅ 실패를 우아하게 처리 (나머지 벤치마크 계속)
8. ✅ 포괄적인 요약 생성

**출력**:
- `benchmark_test_results.json` (기계 판독 가능)
- `benchmark_test_results.md` (사람 판독 가능)
- `artifacts/<run_id>/`의 프로파일링 아티팩트
- 속도 향상 통계가 포함된 챕터별 요약

---

### 출력 예제

```markdown
## 전체 요약
- **테스트된 챕터:** 20/20
- **총 벤치마크:** 264
- **성공:** 258
- **실패:** 6
- **평균 속도 향상:** 3.2x
- **최고 속도 향상:** 89x (ch20: 엔드투엔드 최적화)
- **최저 속도 향상:** 0.95x (ch14: 메모리 제한 모델에서 torch.compile)

## 챕터별 요약
| 챕터 | 상태 | 벤치마크 | 성공 | 평균 속도향상 |
|------|------|----------|------|--------------|
| ch1  | PASS | 21       | 21   | 2.5x         |
| ch13 | PASS | 23       | 23   | 4.1x         |
| ch19 | PASS | 15       | 15   | 6.8x         |
...
```

---

## 고급 기능

### 1. **추론 타이밍 지원**

LLM 추론 벤치마크의 경우, 하네스는 **TTFT** 및 **TPOT**를 캡처합니다:

```python
def benchmark_fn(self):
    # 추론 타이밍이 포함된 dict 반환
    return {
        "ttft_times_ms": [50.2, 48.1, 52.3],  # 요청당 하나
        "tpot_times_ms": [12.1, 11.9, 12.5, ...]  # 토큰당 하나
    }
```

**결과**: 자동 추론 타이밍 통계 (TTFT 및 TPOT 모두에 대한 평균, p50, p90, p99).

### 2. **하드웨어 제한 감지**

하네스는 알려진 하드웨어 제한이 있는 벤치마크를 감지하고 건너뜁니다:

- Triton SM 12.1 지원 이슈
- 디바이스 측 assert 계단식
- 아키텍처별 기능 (예: non-Blackwell GPU의 TMA)

**결과**: 벤치마크가 실패하는 대신 명확한 메시지와 함께 우아하게 건너뜁니다.

### 3. **워크로드 스케일링**

`BaseBenchmark`는 GPU 메모리를 기반으로 자동 워크로드 스케일링을 제공합니다:

```python
size = self._scale_workload_by_memory(base_size=4096)
# >=16GB: 4096 (100%)
# >=8GB:  2048 (50%)
# >=4GB:  1024 (25%)
# <4GB:   409 (10%)
```

**결과**: 벤치마크가 사용 가능한 하드웨어에 자동으로 적응.

### 4. **NVTX 범위 관리**

프로파일링을 위한 자동 NVTX 마커:

```python
with self._nvtx_range("my_operation"):
    # nsys 추적에서 자동으로 마크됨
    result = model(input)
```

**결과**: Nsight Systems에서 더 나은 추적 시각화.

---

## 실제 사용 예제

### 예제 1: 단일 챕터 실행

```bash
python tools/cli/benchmark_cli.py run --targets ch13
```

**결과**: Chapter 13 (PyTorch Profiling)의 23개 벤치마크 모두 실행, 베이스라인 vs 최적화 비교, 보고서 생성.

### 예제 2: 재현 가능한 실행

```bash
python tools/cli/benchmark_cli.py run --reproducible
```

**결과**: 모든 시드가 42로 설정됨, 결정론적 알고리즘 활성화; 느린 커널과 결정론적 경로가 없는 경우 연산 오류 가능성을 감수하고 일치하는 출력 예상.

### 예제 3: 느린 시스템을 위한 확장 타임아웃

```bash
python tools/cli/benchmark_cli.py run --timeout-multiplier 2.0
```

**결과**: 모든 타임아웃 두 배 (30s → 60s setup, 15s → 30s measurement 등).

### 예제 4: 콜드 스타트 측정

```bash
python tools/cli/benchmark_cli.py run --cold-start
```

**결과**: 벤치마크 간 GPU 상태 재설정, 추가 정리, 콜드 스타트 성능 측정.

### 예제 5: 프로파일링 비활성화 (더 빠른 실행)

```bash
python tools/cli/benchmark_cli.py run --no-profile
```

**결과**: 타이밍 전용 실행 (nsys/ncu/PyTorch profiler 없음), 더 빠른 실행, 여전히 타이밍 및 메모리 통계 수집.

---

## 통합 포인트

### 1. **검색 시스템**

`discover_benchmarks()`는 벤치마크 쌍을 자동으로 찾습니다:
- 챕터 디렉토리 스캔
- 베이스라인/최적화 파일 매칭
- 구조화된 튜플 반환

### 2. **비교 템플릿**

`chapter_compare_template.py`가 제공하는 것:
- `load_benchmark()`: 동적 모듈 로딩
- `compare_baseline_optimized()`: 표준 비교 워크플로우
- 하네스와의 통합

### 3. **매니페스트 시스템**

`run_manifest.py`가 캡처하는 것:
- 완전한 환경 상태
- 하드웨어/소프트웨어 버전
- Git 상태
- 구성

### 4. **아티팩트 관리**

`artifact_manager.py`가 구성하는 것:
- 프로파일링 출력
- 로그
- 보고서
- 타임스탬프가 지정된 실행 디렉토리

---

## 이 하네스가 강력한 이유

### 1. **제로 구성 벤치마킹**

벤치마크 클래스를 작성하면 하네스가 나머지를 처리:
- ✅ 정확한 타이밍
- ✅ 통계 분석
- ✅ 프로파일링 통합
- ✅ 에러 처리
- ✅ GPU 상태 관리

### 2. **프로덕션급 신뢰성**

- ✅ 서브프로세스 격리가 계단식 실패 방지
- ✅ 포괄적인 타임아웃 보호
- ✅ 우아한 에러 복구
- ✅ 벤치마크 간 깨끗한 GPU 상태

### 3. **포괄적인 데이터 수집**

모든 벤치마크가 생성하는 것:
- 타이밍 통계 (평균, 중앙값, p99 등)
- 메모리 통계 (피크, 할당됨)
- 프로파일러 아티팩트 (nsys, ncu, PyTorch 추적)
- 프로파일러 메트릭 (SM 활용도, 대역폭, 점유율)
- 완전한 환경 매니페스트

### 4. **자동 비교**

- ✅ 베이스라인/최적화 쌍 자동 검색
- ✅ 속도 향상 자동 계산
- ✅ 회귀 자동 감지
- ✅ 보고서 자동 생성

### 5. **확장성**

- ✅ 하나의 명령으로 264개 벤치마크 실행
- ✅ 실패를 우아하게 처리 (나머지 계속)
- ✅ 제품군 수준 타임아웃 (기본값 4시간)
- ✅ 병렬 실행 지원 (서브프로세스 통해)

### 6. **재현성**

- ✅ 시드 관리 (모든 RNG)
- ✅ 결정론적 모드 지원
- ✅ 완전한 환경 캡처
- ✅ Git 상태 추적

### 7. **개발자 경험**

- ✅ 간단한 프로토콜 (3개 메서드만 구현)
- ✅ 진단이 포함된 풍부한 에러 메시지
- ✅ 구조화된 출력 (JSON + Markdown)
- ✅ 기존 도구와 통합 (nsys-ui, ncu-ui, Chrome tracing)

---

## 요약

`BenchmarkHarness`는 다음을 수행하는 **프로덕션급 벤치마킹 프레임워크**입니다:

1. **자동 검색** 20개 챕터에 걸쳐 264개 벤치마크
2. **안정적 실행** 서브프로세스 격리 및 타임아웃 보호
3. **포괄적 프로파일링** nsys, ncu, PyTorch profiler 통합
4. **통계 분석** 평균, 중앙값, 백분위수, 커스텀 메트릭
5. **자동 비교** 베이스라인 vs 최적화 구현
6. **포괄적 보고** JSON 및 Markdown 출력
7. **우아한 에러 처리** 복구 및 부분 결과
8. **재현성 보장** 시드 관리 및 매니페스트 캡처

**결과**: 단일 명령이 전체 벤치마크 제품군을 실행하여 포괄적인 성능 분석, 프로파일링 데이터, 비교 보고서를 생성 - 대규모 체계적인 성능 엔지니어링을 가능하게 합니다.
