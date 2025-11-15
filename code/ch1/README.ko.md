# Chapter 1: 성능 기초

## 개요

이 챕터는 모든 성능 최적화 작업의 기초를 확립합니다. 코드를 프로파일링하고, 병목 지점을 식별하며, 5-10배 속도 향상을 제공하는 기본 최적화를 적용하는 방법을 배웁니다. 이러한 기법은 보편적으로 적용 가능하며 이후 챕터의 고급 최적화를 위한 기반이 됩니다.

## 학습 목표

이 챕터를 완료하면 다음을 수행할 수 있습니다:

- [OK] Python/PyTorch 코드를 프로파일링하여 성능 병목 지점 식별
- [OK] goodput(유용한 연산 vs 전체 시간)을 측정하여 효율성 정량화
- [OK] 메모리 관리 최적화 적용 (고정 메모리, 사전 할당 버퍼)
- [OK] 배치 연산을 사용하여 GPU 활용도 향상
- [OK] CUDA Graph를 활용하여 커널 실행 오버헤드 감소
- [OK] 기본 최적화를 언제, 어떻게 적용하는지 이해

## 사전 요구사항

**없음** - 이것은 기초 챕터입니다. 여기서 시작하세요!

**하드웨어**: NVIDIA GPU

**소프트웨어**: PyTorch [file]+, CUDA [file]+

## 예제

### 1. 베이스라인 구현

**목적**: 베이스라인 성능 측정 방법론 확립.

**시연 내용**:
- 기본 학습 루프 구조
- 순진한 구현의 일반적인 비효율성
- 벤치마크 프로토콜 통합

**실행 방법**:
```bash
python3 [baseline script]
```

**예상 출력**:
```
Baseline: Performance Basics
======================================================================
Average time: [file] ms
Median: [file] ms
Std: [file] ms
```

일반적인 베이스라인: **40-60% goodput** (텐서 생성 및 전송으로 인한 상당한 오버헤드)

---

### 2. 최적화 구현

**목적**: 개별 기본 최적화를 적용하여 영향 측정.

이 챕터는 별도의 벤치마크 파일을 통해 최적화를 시연합니다:

#### 2a. 고정 메모리(Pinned Memory) 최적화

**문제**: 고정되지 않은 메모리는 H2D 전송을 위해 CPU 스테이징 버퍼가 필요합니다.

**해결책**: CPU-GPU 전송을 빠르게 하기 위해 고정 메모리 사용:
```python
host_data = [file](32, 256, pin_memory=True)
[file]_(host_data, non_blocking=True)  # 논블로킹 복사
```

**영향**: 2-6배 빠른 H2D 전송 (시스템에 따라 다름)

**실행 방법**:
```bash
python3 [pinned memory script]
```

#### 2b. 더 큰 배치 크기 최적화

**문제**: 작은 배치(32)는 GPU를 충분히 활용하지 못함 (낮은 MFLOP).

**해결책**: 연산을 포화시키기 위해 배치 크기 증가:
```python
batch_size = 256  # vs 베이스라인 32
```

**영향**: 87 MFLOP → 1000+ MFLOP (10배 이상 GEMM 효율성)

**실행 방법**:
```bash
python3 [batch size script]
```

#### 2c. CUDA Graph 최적화

**문제**: 각 커널 실행에 약 5-20μs 오버헤드가 있습니다. 작은 커널은 연산보다 실행에 더 많은 시간을 소비합니다!

**해결책**: 정적 연산 그래프를 캡처하고 재생:
```python
graph = [file].CUDAGraph()
with [file].graph(graph):
    output = model(input)
[file]()  # 재실행보다 훨씬 빠름
```

**영향**: 실행 오버헤드 약 50-70% 감소

**포함된 주요 최적화**:
- 사전 할당된 디바이스 버퍼 (텐서 생성 오버헤드 제거)
- 더 빠른 전송을 위한 고정 메모리

**실행 방법**:
```bash
python3 [CUDA graphs script]
```

**모든 최적화 비교**:
```bash
python3 [compare script]  # 베이스라인 vs 모든 최적화 변형 비교
```

**예상 전체 속도 향상**: **2-5배** (워크로드 및 하드웨어에 따라 다름)

---

### 3. CUDA GEMM 예제 - 배치 GEMM 최적화

**목적**: CUDA 수준에서 배치 연산의 중요성 시연.

**프로파일링에서 관찰된 문제**: 학습 루프가 40개의 별도 GEMM을 순차적으로 실행:
- 각 실행: 약 10μs 오버헤드
- 총 오버헤드: 배치당 400μs
- 커널 퓨전 기회 불량

**해결책**: cuBLAS 배치 GEMM API 사용:
```cpp
cublasSgemmBatched(handle, ..., batch_count);
```

**실행 방법**:
```bash
cd ch1
make
# 컴파일된 바이너리 실행 (아키텍처 접미사 자동 추가)
```

**예상 출력**:
```
Individual GEMMs: XXX ms
Batched GEMM:     YYY ms
Speedup:          [file]
```

**일반적인 속도 향상**: **20-40배** (작은 행렬에서 더 극적)

**핵심 인사이트**: 이것이 PyTorch가 내부적으로 연산을 자동으로 배치하는 이유입니다!

---

### 4. Roofline 성능 모델

**목적**: roofline 분석을 구현하여 커널을 연산 제한 또는 메모리 제한으로 분류.

**시연 내용**:
- 다양한 연산에 대한 산술 강도(FLOP/Byte) 계산
- NVIDIA GPU용 roofline 모델에 커널 플로팅
- 최적화가 연산 또는 메모리 대역폭을 타겟팅해야 하는지 식별
- 벡터 연산(메모리 제한) vs 행렬 연산(연산 제한) 비교

**핵심 개념**:
- **Roofline 모델**: 연산 피크 또는 메모리 대역폭에 의해 정의된 성능 상한선
- **Ridge 포인트**: 연산 및 대역폭 상한선이 교차하는 산술 강도
- **예제 사양**: 최신 NVIDIA GPU는 높은 TFLOP 및 메모리 대역폭 달성
- **최적화 전략**: 메모리 제한 커널은 더 나은 데이터 재사용 필요; 연산 제한 커널은 더 나은 명령어 혼합 필요

**실행 방법**:
```bash
python3 [roofline analysis script]
```

**예상 출력**:
```
Vector Add:
  AI: [file] FLOP/Byte
  Achieved: [file] TFLOPS
  Memory-bound (AI << 250)

Matrix Multiply:
  AI: [file] FLOP/Byte
  Achieved: [file] TFLOPS
  Compute-bound (AI > 250)

Roofline plot saved to [file]
```

---

### 챕터 프로파일링

챕터 프로파일링은 비교 스크립트에 의해 처리됩니다. 프로젝트 루트에서 실행:

```bash
python3 -c "from [file] import profile; profile()"
```

또는 통합 진입점을 사용하여 벤치마크 실행:
```bash
python [benchmark script] --chapter 1
```

**핵심 인사이트**: ridge 포인트 아래의 연산은 연산이 아닌 메모리 대역폭에 의해 제한됩니다!

---

### 5. 산술 강도 최적화 - 커널 최적화 전략

**목적**: 커널 최적화 기법과 산술 강도에 대한 영향 표시.

**구현**: 점진적 최적화를 시연하는 베이스라인 및 최적화 CUDA 커널

**시연 내용**:
- **베이스라인**: 간단한 요소별 커널
- **루프 언롤링**: 분기 오버헤드 감소, ILP 노출
- **벡터화 로드**: `float4`를 사용하여 한 번에 128비트 로드
- **FLOP 증가**: 유용한 작업을 추가하여 AI 개선
- **커널 퓨전**: 여러 패스를 결합하여 메모리 트래픽 제거

**성능 진행**:
```
Baseline:     125 GB/s,  AI = [file] FLOP/Byte
Unrolled:     145 GB/s,  AI = [file] FLOP/Byte (더 나은 활용)
Vectorized:   245 GB/s,  AI = [file] FLOP/Byte (병합 + 대역폭)
More FLOPs:   280 GB/s,  AI = [file] FLOP/Byte (6배 더 나은 AI!)
Fused:        420 GB/s,  AI = [file] FLOP/Byte (12배 더 나은 AI!)
```

**실행 방법**:
```bash
make
# 컴파일된 바이너리 실행 (적절한 아키텍처 접미사 추가)
# 예제: ```

**예상 출력**:
```
Arithmetic Intensity Optimization Demo (N = 10M elements)

Baseline kernel:
  Time: [file] ms, Bandwidth: [file] GB/s, AI: [file] FLOP/Byte

Unrolled kernel:
  Time: [file] ms, Bandwidth: [file] GB/s, AI: [file] FLOP/Byte

Vectorized kernel:
  Time: [file] ms, Bandwidth: [file] GB/s, AI: [file] FLOP/Byte

Optimized kernel (more FLOPs):
  Time: [file] ms, Bandwidth: [file] GB/s, AI: [file] FLOP/Byte

Fused kernel:
  Time: [file] ms, Bandwidth: [file] GB/s, AI: [file] FLOP/Byte
  Overall speedup: [file]
```

**핵심 인사이트**: 산술 강도(바이트당 더 많은 FLOP) 증가는 메모리 병목을 줄이고 성능을 향상시킵니다!

---

## 성능 분석

### 자신의 코드 프로파일링

공통 프로파일링 인프라 사용:

```bash
# Python 예제 프로파일링
../.[executable]/profiling/[file] [baseline script]
../.[executable]/profiling/[file] [optimized script]

# Nsight Systems에서 타임라인 보기
nsys-ui [profile output file]
```

**찾아야 할 것**:
- ERROR: GPU 커널 간 긴 CPU 간격 → 비동기 연산 추가
- ERROR: 많은 작은 커널 실행 → 배치 또는 퓨전 연산
- ERROR: `aten::empty_strided`가 상당한 시간 소요 → 버퍼 사전 할당
- [OK] GPU 활용률 > 80% → 좋습니다!

### 예상 성능 개선

| 최적화 | 베이스라인 → 최적화 | 속도향상 |
|---------|-------------------|---------|
| 사전 할당 버퍼 | 210ms 오버헤드 → 0ms | ~2배 |
| 고정 메모리 | 시스템 의존적 | 2-6배 |
| CUDA Graph | 5-20μs/실행 → <1μs | [file]-2배 |
| 더 큰 배치 | 87 MFLOP → 1000+ | 10배+ |
| **결합** | **전체 엔드투엔드** | **5-10배** |

*결과는 하드웨어에 따라 다를 수 있습니다.*

---

## 베이스라인/최적화 예제 쌍

모든 예제는 베이스라인/최적화 패턴을 따르며 벤치마킹 프레임워크와 통합됩니다:

### 사용 가능한 쌍

1. **Coalescing** - 베이스라인 및 최적화 구현
   - 병합된 vs 병합되지 않은 메모리 액세스 패턴 시연
   - 적절한 메모리 액세스로 인한 대역폭 개선 표시

2. **Double Buffering** - 베이스라인 및 최적화 구현
   - CUDA 스트림을 사용하여 메모리 전송과 연산 오버랩
   - 비동기 연산을 통한 지연 시간 숨김 시연

**비교 실행:**
```bash
python3 [compare script]  # 모든 베이스라인/최적화 쌍 비교
```

---

## 모든 예제 실행 방법

```bash
cd ch1

# 의존성 설치
pip install -r [file]

# Python 예제 실행
python3 [baseline script]                        # 베이스라인
python3 [pinned memory script]                   # 고정 메모리 최적화
python3 [batch size script]                      # 더 큰 배치 크기
python3 [CUDA graphs script]                     # CUDA Graph 최적화
python3 [roofline script]                        # Roofline 모델

# 베이스라인/최적화 비교 실행
python3 [compare script]                         # 모든 쌍 비교

# CUDA 예제 빌드 및 실행
make
# 컴파일된 바이너리 실행 (아키텍처 접미사 자동 추가)

# 예제 프로파일링 (선택 사항)
../.[executable]/profiling/[file] [baseline script]
../.[executable]/profiling/[file] [optimized script]
../.[executable]/profiling/[file] [CUDA binary] ch1_ai
```

---

## 핵심 요점

1. **항상 먼저 프로파일링**: 맹목적으로 최적화하지 마세요. 프로파일러를 사용하여 실제 병목 지점을 식별하세요.

2. **메모리 관리가 중요합니다**: 버퍼를 사전 할당하고 고정 메모리를 사용하면 최소한의 코드 변경으로 2-6배 속도 향상을 얻을 수 있습니다.

3. **배치 연산**: GPU는 병렬 처리에서 번창합니다. 연산을 배치하면 오버헤드가 줄어들고 효율성이 극적으로 향상됩니다(많은 경우 10배 이상).

4. **실행 오버헤드는 실제입니다**: 작은 연산의 경우 커널 실행 오버헤드가 지배적입니다. CUDA Graph와 배치가 이를 완화합니다.

5. **복합 개선**: 개별 최적화가 곱해집니다. 2배 + 2배 + [file] → 6배 결합 속도 향상.

6. **쉬운 성과**: 이러한 최적화는 최소한의 코드 변경이 필요하지만 주요 개선을 제공합니다. 항상 먼저 적용하세요!

---

## 일반적인 함정

### 함정 1: 과도한 배치

**문제**: 배치 크기가 너무 큼 → OOM(메모리 부족) 오류.

**해결책**: 배치 크기 스윕을 사용하여 최적점 찾기. 일반적 범위: 최신 NVIDIA GPU의 경우 64-512.

### 함정 2: 동적 형상의 CUDA Graph

**문제**: CUDA Graph는 정적 형상이 필요합니다. 동적 모델은 실패하거나 속도 향상이 없습니다.

**해결책**: 모델의 정적 부분에만 그래프를 사용하세요. 추론에서 prefill/decode가 좋은 후보입니다.

### 함정 3: 동기화 없이 측정

**문제**: CUDA 연산은 비동기입니다. `[file].synchronize()` 없이 `[file]()`는 대기열 시간을 측정하지 실행 시간을 측정하지 않습니다!

**해결책**: 타이밍 전에 항상 동기화:
```python
[file].synchronize()
start = [file]()
model(input)
[file].synchronize()  # 중요!
elapsed = [file]() - start
```

### 함정 4: 콜드 스타트 측정

**문제**: 처음 몇 번의 반복에는 GPU 워밍업, 드라이버 오버헤드, cuDNN 자동 튜닝이 포함됩니다.

**해결책**: 벤치마킹 전에 항상 워밍업(10-20회 반복)하세요.

---

## 다음 단계

**더 준비되셨나요?** → [Chapter 2: GPU Hardware Architecture](.[executable]/[file])

다음을 배웁니다:
- NVIDIA GPU 하드웨어 아키텍처
- 메모리 계층 구조
- NVLink 및 인터커넥트
- 하드웨어 아키텍처가 최적화 전략에 어떻게 정보를 제공하는지

**프로파일링을 더 깊이 탐구하고 싶으신가요?** → [Chapter 13: PyTorch Profiling](.[executable]/[file])

---

## 추가 리소스

- **공식 문서**: [PyTorch Performance Tuning Guide](https://[file]/tutorials/recipes/recipes/[file])
- **cuBLAS 문서**: [CUDA Toolkit Docs - cuBLAS](https://[file].com/cuda/cublas/)
- **CUDA Graph**: [CUDA Programming Guide - Graphs](https://[file].com/cuda/cuda-c-programming-guide/[file]#cuda-graphs)

---

**챕터 상태**: [OK] 완료
