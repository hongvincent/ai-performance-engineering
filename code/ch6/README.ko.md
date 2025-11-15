# Chapter 6: CUDA 기초 - 첫 번째 커널

## 개요

이 챕터는 CUDA 프로그래밍을 기초부터 소개합니다. 첫 번째 CUDA 커널을 작성하고, 스레드 계층 구조를 이해하며, 모든 GPU 프로그래밍의 기반이 되는 기본 병렬화 패턴을 배웁니다.

## 학습 목표

이 챕터를 완료하면 다음을 수행할 수 있습니다:

- [OK] 기본 CUDA 커널 작성 및 실행
- [OK] CUDA 스레드 계층 구조 이해 (스레드, 블록, 그리드)
- [OK] 임의의 문제 크기에 대한 그리드 및 블록 차원 계산
- [OK] 스레드 인덱싱을 사용하여 연산을 데이터에 매핑
- [OK] 기본 병렬화 패턴 적용
- [OK] 점유율 및 리소스 제한 이해

## 사전 요구사항

**이전 챕터**:
- [Chapter 1: Performance Basics](.[executable]/README.md) - 프로파일링 기초
- [Chapter 2: NVIDIA GPU Hardware](.[executable]/README.md) - GPU 아키텍처 기초

**필수**: 기본 C/C++ 지식, CUDA 지원 GPU

## CUDA 스레드 계층 구조

예제를 시작하기 전에 실행 모델을 이해하세요:

```
Grid (전체 커널 실행)
├── Block 0
│   ├── Warp 0 (threads 0-31)
│   ├── Warp 1 (threads 32-63)
│   └── ...
├── Block 1
│   ├── Warp 0
│   └── ...
└── ...

주요 제약 (NVIDIA GPU):
- 블록당 최대 스레드: 1024
- 워프 크기: 32 (스레드가 lock-step으로 실행)
- SM당 최대 블록: 32
- SM당 최대 워프: 64
```

**중요한 개념**: 워프 내의 스레드는 동시에 실행됩니다 (SIMT - Single Instruction, Multiple Threads).

---

## 예제

###  Hello World

**목적**: 가장 간단한 CUDA 커널 - GPU에서 출력.

**코드**:
```cpp
__global__ void hello() {
    printf("Hello from thread %d in block %d\n",
           threadIdx.x, blockIdx.x);
}

int main() {
    hello<<<2, 4>>>();  // 2개 블록, 블록당 4개 스레드
    cudaDeviceSynchronize();
    return 0;
}
```

**핵심 개념**:
- `__global__`: 커널 함수 (GPU에서 실행, CPU에서 호출)
- `<<<blocks, threads>>>`: 실행 구성
- `threadIdx.x`: 블록 내 스레드 인덱스
- `blockIdx.x`: 그리드 내 블록 인덱스

**실행 방법**:
```bash
make my_first_kernel
```

**예상 출력**:
```
Hello from thread 0 in block 0
Hello from thread 1 in block 0
...
Hello from thread 3 in block 1
```

---

###  요소별 연산

**목적**: 책의 메모리 관리 및 커널 실행 구성을 포함한 완전한 CUDA 워크플로우 시연.

**시연 내용**:
- 고정 메모리 할당 (`cudaMallocHost`)
- 디바이스 메모리 할당 및 H2D/D2H 전송
- 스레드 인덱싱 계산
- 그리드/블록 차원 계산
- 제자리 요소별 연산

**코드 워크스루**:

```cpp
__global__ void myKernel(float* input, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        input[idx] *= 2.0f;  // 각 요소를 2배로 스케일
    }
}

int main() {
    const int N = 1'000'000;

    // 고정 호스트 메모리 할당 (더 빠른 H2D/D2H 전송)
    float* h_input = nullptr;
    cudaMallocHost(&h_input, N * sizeof(float));
    for (int i = 0; i < N; ++i) {
        h_input[i] = 1.0f;
    }

    // 디바이스 메모리 할당
    float* d_input = nullptr;
    cudaMalloc(&d_input, N * sizeof(float));

    // 디바이스로 복사
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);

    // 그리드 차원 계산
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // 커널 실행
    myKernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, N);
    cudaDeviceSynchronize();

    // 결과 복사
    cudaMemcpy(h_input, d_input, N * sizeof(float), cudaMemcpyDeviceToHost);

    // 정리
    cudaFree(d_input);
    cudaFreeHost(h_input);
    return 0;
}
```

**핵심 개념**:
- **그리드 계산**: `(N + threadsPerBlock - 1) / threadsPerBlock`는 전체 커버리지 보장
- **고정 메모리**: `cudaMallocHost`는 `malloc`보다 더 빠른 전송 가능
- **동기화**: `cudaDeviceSynchronize()`는 커널 완료 대기
- **경계 검사**: `if (idx < N)`는 그리드 크기 > 문제 크기인 경우 처리

**실행 방법**:
```bash
make simple_kernel
```

**예상 출력**:
```
Simple kernel succeeded: 1000000 elements scaled by 2.0f
  Configuration: 3907 blocks × 256 threads = 1000192 total threads
```

**성능 특성**:
- 1M 요소 @ 블록당 256 스레드 = 3,907 블록
- 각 블록이 약 256개 요소 처리
- NVIDIA GPU에서 < 1ms 실행 (메모리 대역폭 제한)

---

### 3. [baseline] → [optimized] - 베이스라인 vs 최적화

**목적**: 기본 병렬화 패턴 시연: 순차 → 병렬.

#### 베이스라인

**문제**: 단일 스레드가 모든 작업 수행 (GPU 활용 부족).

```cpp
__global__ void addSequential(const float* A, const float* B, float* C, int n) {
    // 블록 0의 스레드 0만 작업!
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        for (int i = 0; i < n; i++) {
            C[i] = A[i] + B[i];
        }
    }
}

// 실행: addSequential<<<1, 1>>>(...)
```

**성능**: 병렬 버전보다 약 2,000배 느림!

#### 최적화

**해결책**: 각 스레드가 하나의 요소 처리.

```cpp
__global__ void addParallel(const float* A, const float* B, float* C, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        C[idx] = A[idx] + B[idx];
    }
}

// 실행:
int threads = 256;
int blocks = (n + threads - 1) / threads;
addParallel<<<blocks, threads>>>(A, B, C, n);
```

**핵심 패턴**: `idx = blockIdx.x * blockDim.x + threadIdx.x`
CUDA에서 가장 일반적인 인덱싱 패턴입니다!

**실행 방법**:
```bash
make add_sequential add_parallel
```

**예상 속도 향상**: **약 2000배** (1,000,000 요소)

**왜 이렇게 극적인 속도 향상?**
- 순차: 1 스레드 × 1,000,000 반복 = 1,000,000 연산 순차적으로
- 병렬: 3,907 스레드 × 각각 256 연산 = 병렬로 실행!

---

###  2D 스레드 인덱싱

**목적**: 2D 문제(이미지, 행렬)로 확장.

**2D 그리드 패턴**:
```cpp
__global__ void process2D(float* matrix, int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width && row < height) {
        int idx = row * width + col;  // 행 우선 인덱싱
        matrix[idx] = /* 연산 */;
    }
}

// 실행:
dim3 threads(16, 16);  // 16×16 = 블록당 256 스레드
dim3 blocks((width + 15) / 16, (height + 15) / 16);
process2D<<<blocks, threads>>>(matrix, width, height);
```

**사용 사례**: 이미지 처리, 행렬 연산, 컨볼루션.

**실행 방법**:
```bash
make 2d_kernel
```

---

###  점유율 계산

**목적**: 점유율 및 리소스 제한 이해.

**점유율이란?**
```
Occupancy = Active_Warps / Max_Warps_Per_SM
```

높은 점유율 → 더 나은 지연 시간 숨김 → 더 높은 처리량 (보통).

**코드**:
```cpp
int blockSize = 256;
int minGridSize, gridSize;

// 최적 블록 크기 계산
cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, myKernel);

// 선택한 블록 크기에 대한 점유율 계산
int numBlocks;
cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks, myKernel, blockSize, 0);

float occupancy = (numBlocks * blockSize / 32.0f) / 64.0f;  // NVIDIA GPU에서 SM당 64 워프
printf("Occupancy: %.1f%%\n", occupancy * 100);
```

**실행 방법**:
```bash
make occupancy_api
```

**예상 출력**:
```
Block size: 256
Occupancy: 100% (full SM utilization) [OK]
```

**점유율 가이드라인**:
- **> 50%**: 일반적으로 좋음
- **> 75%**: 대부분의 커널에 우수
- **100%**: 완벽하지만 항상 필요한 것은 아님

**트레이드오프**: 때로는 스레드당 더 많은 레지스터로 낮은 점유율이 더 빠릅니다!

---

### 6. `[CUDA file]` (구현은 소스 파일 참조) - 리소스 사용 제어

**목적**: `__launch_bounds__`를 사용하여 레지스터 및 공유 메모리 사용 최적화.

**코드**:
```cpp
__global__ void __launch_bounds__(256, 4)  // 256 스레드, SM당 최소 4 블록
myKernel(float* data) {
    // 컴파일러 보장:
    // - 커널이 SM당 4 블록에 맞도록 ≤ 레지스터 사용
    // - 블록당 256 스레드에 최적화
}
```

**사용 시기**:
- 커널이 높은 레지스터 압력 → 레지스터 제한
- 보장된 점유율 원함 → SM당 최소 블록 지정
- 성능 튜닝 → 다양한 경계로 실험

**실행 방법**:
```bash
make launch_bounds_example
```

---

###  관리 메모리(Managed Memory)

**목적**: CUDA 통합 메모리로 메모리 관리 단순화.

**전통적인 CUDA** (수동 관리):
```cpp
float *h_data, *d_data;
h_data = (float*)malloc(size);
cudaMalloc(&d_data, size);
cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);
kernel<<<...>>>(d_data);
cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost);
```

**통합 메모리** (자동):
```cpp
float *data;
cudaMallocManaged(&data, size);  // CPU와 GPU에서 액세스 가능!
kernel<<<...>>>(data);  // 자동 마이그레이션
// CPU에서 데이터 직접 사용, 복사 불필요!
```

**이점**:
- [OK] 더 간단한 코드 (명시적 전송 없음)
- [OK] 자동 페이지 마이그레이션
- [OK] 오버서브스크립션 (GPU보다 많은 메모리 사용)

**단점**:
- ERROR: 명시적 전송보다 느림 (페이지 폴트 오버헤드)
- ERROR: 배치에 대한 제어 감소
- ERROR: 고성능 코드에 이상적이지 않음

**사용 시기**: 프로토타이핑, 불규칙한 액세스 패턴, 오버서브스크립션 시나리오.

**실행 방법**:
```bash
make unified_memory
```

---

## 그리드 및 블록 차원 계산

### 1D 문제

```cpp
int threads = 256;  // 일반적: 128, 256, 또는 512
int blocks = (n + threads - 1) / threads;  // 올림 나누기
kernel<<<blocks, threads>>>(data, n);
```

**왜 256 스레드?**
- 워프 크기(32)의 배수 [OK]
- 대부분의 GPU에서 좋은 점유율 [OK]
- 병렬성과 리소스 사용의 균형 [OK]

### 2D 문제

```cpp
dim3 threads(16, 16);  // 총 256 스레드
dim3 blocks(
    (width + threads.x - 1) / threads.x,
    (height + threads.y - 1) / threads.y
);
kernel<<<blocks, threads>>>(data, width, height);
```

**일반적인 2D 크기**: (16,16), (32,8), (8,32) 액세스 패턴에 따라.

### 3D 문제 (드물게)

```cpp
dim3 threads(8, 8, 8);  // 총 512 스레드
dim3 blocks(
    (depth + 7) / 8,
    (height + 7) / 8,
    (width + 7) / 8
);
```

**사용 사례**: 3D 컨볼루션, 볼류메트릭 데이터, 시뮬레이션.

---

## 성능 분석

### 공통 인프라 사용

```bash
# CUDA 커널 프로파일링
../.[executable]/profiling/profile_cuda.sh [executable] baseline
../.[executable]/profiling/profile_cuda.sh [executable] baseline

# Nsight Compute에서 비교
ncu-ui ../.[executable]/ch6/add_parallel_baseline_metrics_*.ncu-rep
```

### 주시할 주요 메트릭

| 메트릭 | 타겟 | 확인 방법 |
|--------|------|----------|
| 점유율(Occupancy) | > 50% | Nsight Compute |
| 워프 실행 효율성 | > 90% | 분기 없음 |
| 메모리 처리량 | 피크에 가까움 | 병합된 액세스 (Ch7) |
| 연산 처리량 | 다양함 | 알고리즘에 따라 |

---

## 베이스라인/최적화 예제 쌍

모든 CUDA 예제는 `baseline_*.cu` / `optimized_*.cu` 패턴을 따릅니다:

### 사용 가능한 쌍

1. **Add Operation** ([source file] / [source file])
   - 순차 vs 병렬 벡터 덧셈
   - 기본 병렬화 패턴 시연

2. **Coalescing** ([source file] / [source file])
   - 병합되지 않은 vs 병합된 메모리 액세스
   - 적절한 액세스 패턴으로 인한 대역폭 개선 표시

3. **Bank Conflicts** ([source file] / [source file])
   - 공유 메모리 뱅크 충돌 및 패딩 솔루션
   - 패딩으로 뱅크 충돌 제거 시연

4. **Instruction-Level Parallelism** ([source file] / [source file])
   - 순차 vs 독립 연산 및 루프 언롤링
   - 명령어 지연 시간 숨김을 위한 ILP 이점 표시

**비교 실행:**
```bash
python3 [script]  # 모든 베이스라인/최적화 쌍 비교 (Python 래퍼 통해)
```

---

## 모든 예제 실행 방법

```bash
cd ch6

# 모든 예제 빌드
make

# 복잡도 순서로 실행

# 학습을 위한 프로파일링
../.[executable]/profiling/profile_cuda.sh [executable] baseline
```

---

## 핵심 요점

1. **스레드 인덱싱이 기본입니다**: `idx = blockIdx.x * blockDim.x + threadIdx.x`는 CUDA에서 가장 중요한 패턴입니다.

2. **병렬화가 대규모 속도 향상 제공**: 순진한 병렬 코드도 순차보다 100-1000배 빠를 수 있습니다.

3. **블록 크기가 중요합니다**: 블록당 256 스레드가 좋은 기본값입니다. 특정 커널에 맞게 튜닝하세요.

4. **경계 검사가 필수입니다**: 범위 초과 액세스를 피하기 위해 항상 `if (idx < n)` 확인하세요.

5. **점유율이 전부는 아닙니다**: 100% 점유율이 최고 성능을 보장하지 않습니다. 메모리 액세스 패턴(Ch7)이 종종 더 중요합니다.

6. **간단하게 시작, 나중에 최적화**: 먼저 올바른 병렬 코드를 얻은 다음 최적화하세요(챕터 7-10).

---

## 일반적인 함정

### 함정 1: 경계 검사 잊기

**문제**: `n`이 블록 크기로 나누어지지 않을 때 배열 범위 초과.

**해결책**: 항상 경계 검사 추가:
```cpp
int idx = blockIdx.x * blockDim.x + threadIdx.x;
if (idx < n) {  // 중요!
    data[idx] = ...;
}
```

### 함정 2: 잘못된 그리드 크기 계산

**문제**: 정수 나누기가 절삭 → 일부 요소가 처리되지 않음.

**잘못된 예**:
```cpp
int blocks = n / threads;  // 절삭!
```

**올바른 예**:
```cpp
int blocks = (n + threads - 1) / threads;  // 올림 나누기
```

### 함정 3: 커널 후 동기화하지 않음

**문제**: 커널이 완료되기 전에 결과에 액세스.

**해결책**: `cudaDeviceSynchronize()` 호출 또는 비동기 확인:
```cpp
kernel<<<...>>>();
cudaDeviceSynchronize();  // 완료 대기
// 이제 결과에 안전하게 액세스
```

### 함정 4: 너무 적은 스레드 사용

**문제**: 블록 크기 32 → 낮은 점유율 → 성능 저하.

**해결책**: 좋은 점유율을 위해 블록당 128-512 스레드 사용.

### 함정 5: 고성능 코드에 통합 메모리 사용

**문제**: 성능 중요 코드에 `cudaMallocManaged` 사용.

**현실**: 명시적 `cudaMemcpy`가 더 빠릅니다. 프로토타이핑에만 통합 메모리 사용.

---

## 다음 단계

**CUDA 마스터리 계속** → [Chapter 7: Memory Access Patterns](.[executable]/README.md)

다음을 배웁니다:
- 10배 대역폭 개선을 위한 메모리 병합
- 데이터 재사용을 위한 공유 메모리
- 뱅크 충돌 및 회피 방법
- 벡터화된 메모리 액세스

**PyTorch 세계로 돌아가기** → [Chapter 13: PyTorch Profiling](.[executable]/README.md)

---

## 추가 리소스

- **CUDA C Programming Guide**: [Official NVIDIA Docs](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- **CUDA Best Practices**: [Performance Guidelines](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- **Nsight Compute**: [Profiler User Guide](https://docs.nvidia.com/nsight-compute/)
- **Common Headers**: 에러 검사 매크로는 `../.[executable]/headers/cuda_helpers.cuh` 참조

---

**챕터 상태**: [OK] 완료
