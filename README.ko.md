# AI Performance Engineering

_**업데이트:** 이 자료에 대한 실습 중심 강좌에 관심이 있으신가요?_

_관심이 있으시다면, 이 [**양식**](https://docs.google.com/forms/d/e/1FAIpQLSf4TMDLsPcfuoLhaDktXu-hhKIGntQm550BY-ov6bRT_VMJhQ/viewform?usp=sharing&ouid=111382272947765737941)을 작성하여 의견을 표현하고 알림을 받으세요._

## 이 레포지토리 소개

GPU 최적화, 분산 학습, 추론 스케일링, 그리고 현대 AI 워크로드를 위한 전체 스택 성능 튜닝을 다루는 O'Reilly 책을 위한 AI 시스템 성능 엔지니어링 코드, 도구, 리소스입니다.

[![O'Reilly Book](img/ai_sys_perf_engg_cover_cheetah_sm.png)](https://www.amazon.com/Systems-Performance-Engineering-Optimizing-Algorithms/dp/B0F47689K8/)

> **O'Reilly 도서 – 2025년 가을**
> [Amazon에서 구매 가능](https://www.amazon.com/Systems-Performance-Engineering-Optimizing-Algorithms/dp/B0F47689K8/)

### AI 시스템 성능 엔지니어링 도서
현대 AI 시스템은 단순한 FLOP 성능 이상을 요구합니다—하드웨어, 소프트웨어, 알고리즘 전반에 걸친 goodput 중심, 프로파일 우선 엔지니어링이 필요합니다. 이 실습 가이드는 GPU, 인터커넥트, 런타임 스택을 효율적이고 신뢰할 수 있는 학습 및 추론 파이프라인으로 전환하는 방법을 보여줍니다.

Nsight와 PyTorch 프로파일러를 사용하여 실제 병목 지점을 진단하고, 대역폭과 메모리를 최적화하며, 컴파일러 스택(PyTorch + OpenAI Triton)을 사용하여 고효율 커널을 작성하는 방법을 배우게 됩니다. 서빙 측면에서는 vLLM/SGLang, TensorRT-LLM, NVIDIA Dynamo를 사용한 고처리량 추론을 마스터하고—분리형 prefill/decode 및 paged KV 캐시 포함—예산을 초과하지 않으면서 랙 전체로 확장하는 방법을 다룹니다.

사례 연구와 프로파일링 데이터를 활용한 실습 중심, 실증적 방법론을 사용하여, 이 책은 대규모 학습/추론을 구축하거나 운영하는 AI/ML 엔지니어, 시스템 엔지니어, 연구자, 플랫폼 팀에게 유용합니다. 이 책은 최신 NVIDIA GPU를 위한 수천 줄의 PyTorch 및 CUDA C++ 코드 예제를 포함합니다.

* 단순한 활용도가 아닌 goodput을 위한 프로파일링—Nsight Systems/Compute 및 PyTorch 프로파일러를 사용하여 실제 정체 지점을 찾습니다.

* 메모리 및 대역폭 활용—레이아웃, 캐싱, 데이터 이동을 최적화하여 GPU에 지속적으로 데이터를 공급합니다.

* 컴파일러를 활용한 튜닝—PyTorch 컴파일러 스택과 Triton을 활용하여 C++ 보일러플레이트 없이 고효율 커널을 생성합니다.

* 합리적인 학습 확장—병렬 처리 전략(DP, FSDP, TP, PP, CP, MoE)을 적용하고 연산/통신을 오버랩하여 버블을 최소화합니다.

* 조 단위 파라미터 모델을 효율적으로 서빙—vLLM, SGLang, TensorRT-LLM, NVIDIA Dynamo를 사용하여 분리형 prefill/decode 및 KV 캐시 이동을 구현합니다.

* 토큰당 비용 절감—최고 속도뿐만 아니라 와트당 성능과 달러당 처리량을 고려한 엔지니어링을 수행합니다.

* AI 지원 최적화 도입—시스템이 수동 조정을 넘어서 성장함에 따라 AI가 커널을 합성하고 튜닝하도록 합니다.

* 자신감 있게 배포—175개 이상의 항목 체크리스트를 적용하여 팀 전체에서 성과를 재현하고 성능 저하를 방지합니다.

### 저자 소개

Chris Fregly는 Netflix, Databricks, Amazon Web Services(AWS)에서 혁신을 주도해 온 성능 엔지니어이자 AI 제품 리더입니다. 그는 AI/ML 제품을 구축하고, 시장 진출 이니셔티브를 확장하며, 대규모 생성형 AI 및 분석 워크로드의 비용을 절감하는 성능 중심 엔지니어링 팀을 이끌었습니다.

Chris는 두 권의 다른 O'Reilly 책의 저자이기도 합니다: Data Science on AWS 및 Generative AI on AWS. 그는 또한 O'Reilly 과정 "High-Performance AI in Production with NVIDIA GPUs"와 Andrew Ng과 함께한 DeepLearning.ai 과정 "Generative AI with Large-Language Models"의 창시자입니다.

그의 작업은 커널 수준 튜닝, 컴파일러 기반 가속화, 분산 학습, 고처리량 추론에 걸쳐 있습니다. Chris는 [AI Performance Engineering](https://www.meetup.com/ai-performance-engineering)이라는 월간 모임을 주최합니다.

### 175개 이상의 항목 성능 체크리스트

이 책은 전체 라이프사이클을 다루는 현장에서 검증된 최적화를 담은 **175개 이상의 항목 성능 체크리스트**를 제공합니다. 이를 즉시 적용할 수 있습니다:

- ✅ 성능 튜닝 마인드셋 및 비용 최적화
- ✅ 재현성 및 문서화 모범 사례
- ✅ 시스템 아키텍처 및 하드웨어 계획
- ✅ 운영 체제 및 드라이버 최적화
- ✅ GPU 프로그래밍 및 CUDA 튜닝
- ✅ 분산 학습 및 네트워크 최적화
- ✅ 효율적인 추론 및 서빙
- ✅ 전력 및 열 관리
- ✅ 최신 프로파일링 도구 및 기법
- ✅ 아키텍처별 최적화

### 링크

- **도서**: [Amazon의 AI Systems Performance Engineering](https://www.amazon.com/Systems-Performance-Engineering-Optimizing-Algorithms/dp/B0F47689K8/)
- **모임**: [AI Performance Engineering](https://www.meetup.com/ai-performance-engineering)
- **YouTube**: [AI Performance Engineering 채널](https://www.youtube.com/@AIPerformanceEngineering)

> *AI 성능 엔지니어링 커뮤니티를 위해 샌프란시스코에서 제작*

### 주요 초점 영역

- **GPU 아키텍처, PyTorch, CUDA, OpenAI Triton 프로그래밍**
- **분산 학습 및 추론**
- **메모리 최적화 및 프로파일링**
- **PyTorch 성능 튜닝**
- **멀티 노드 스케일링 전략**

## 도서 챕터

### Chapter 1: 소개 및 AI 시스템 개요

- AI 시스템 성능 엔지니어
- 벤치마킹 및 프로파일링
- 분산 학습 및 추론 확장
- 효율적인 리소스 관리
- 팀 간 협업
- 투명성 및 재현성

### Chapter 2: AI 시스템 하드웨어 개요

- CPU 및 GPU "슈퍼칩"
- NVIDIA Grace CPU 및 Blackwell GPU
- NVIDIA GPU Tensor Core 및 Transformer Engine
- Streaming Multiprocessor, Thread, Warp
- 초대규모 네트워킹
- NVLink 및 NVSwitch
- 멀티 GPU 프로그래밍

### Chapter 3: OS, Docker, Kubernetes 튜닝

- 운영 체제 구성
- GPU 드라이버 및 소프트웨어 스택
- NUMA 인식 및 CPU 고정
- 컨테이너 런타임 최적화
- 토폴로지 인식 오케스트레이션을 위한 Kubernetes
- 메모리 격리 및 리소스 관리

### Chapter 4: 분산 네트워킹 통신 튜닝

- 통신 및 연산 오버랩
- 분산 멀티 GPU 통신을 위한 NCCL
- NCCL의 토폴로지 인식
- 분산 데이터 병렬 전략
- NVIDIA Inference Transfer Library (NIXL)
- 인네트워크 SHARP 집계

### Chapter 5: GPU 기반 스토리지 I/O 최적화

- 빠른 스토리지 및 데이터 지역성
- NVIDIA GPUDirect Storage
- 분산 병렬 파일 시스템
- NVIDIA DALI를 사용한 멀티모달 데이터 처리
- 고품질 LLM 데이터셋 생성

### Chapter 6: GPU 아키텍처, CUDA 프로그래밍, 점유율 극대화

- GPU 아키텍처 이해
- Thread, Warp, Block, Grid
- CUDA 프로그래밍 복습
- GPU 메모리 계층 구조 이해
- 높은 점유율 및 GPU 활용도 유지
- Roofline 모델 분석

### Chapter 7: GPU 메모리 액세스 패턴 프로파일링 및 튜닝

- 병합된(Coalesced) vs. 비병합된(Uncoalesced) 전역 메모리 액세스
- 벡터화된 메모리 액세스
- 공유 메모리를 사용한 타일링 및 데이터 재사용
- Warp Shuffle 내장 함수
- 비동기 메모리 프리페칭

### Chapter 8: 점유율 튜닝, Warp 효율성, 명령어 수준 병렬성

- GPU 병목 지점 프로파일링 및 진단
- Nsight Systems 및 Compute 분석
- 점유율 튜닝
- Warp 실행 효율성 향상
- 명령어 수준 병렬성 노출

### Chapter 9: CUDA 커널 효율성 및 산술 강도 증가

- 다단계 마이크로 타일링
- 커널 퓨전
- 혼합 정밀도 및 Tensor Core
- 최적 성능을 위한 CUTLASS 사용
- 인라인 PTX 및 SASS 튜닝

### Chapter 10: 커널 내 파이프라이닝 및 협력적 Thread Block 클러스터

- 커널 내 파이프라이닝 기법
- Warp 특화 생산자-소비자 모델
- 영구 커널 및 메가커널
- Thread Block 클러스터 및 분산 공유 메모리
- Cooperative Group

### Chapter 11: 커널 간 파이프라이닝 및 CUDA Stream

- Stream을 사용한 연산과 데이터 전송 오버랩
- Stream 순서 메모리 할당자
- 이벤트를 사용한 세밀한 동기화
- CUDA Graph를 사용한 제로 오버헤드 실행

### Chapter 12: 동적 및 디바이스 측 커널 오케스트레이션

- Atomic 작업 큐를 사용한 동적 스케줄링
- CUDA Graph를 사용한 반복 커널 실행 배치
- 동적 병렬성
- NVSHMEM을 사용한 여러 GPU 간 오케스트레이션

### Chapter 13: PyTorch 프로파일링, 튜닝, 스케일링

- NVTX 마커 및 프로파일링 도구
- PyTorch 컴파일러 (torch.compile)
- PyTorch에서 메모리 프로파일링 및 튜닝
- PyTorch Distributed를 사용한 스케일링
- HTA를 사용한 멀티 GPU 프로파일링

### Chapter 14: PyTorch 컴파일러, XLA, OpenAI Triton 백엔드

- PyTorch 컴파일러 심층 분석
- OpenAI Triton을 사용한 커스텀 커널 작성
- PyTorch XLA 백엔드
- 고급 Triton 커널 구현

### Chapter 15: 멀티 노드 추론 병렬성 및 라우팅

- 분리형 Prefill 및 Decode 아키텍처
- MoE 모델을 위한 병렬 처리 전략
- 추측적 및 병렬 디코딩 기법
- 동적 라우팅 전략

### Chapter 16: 대규모 추론 프로파일링, 디버깅, 튜닝

- 성능 프로파일링 및 튜닝 워크플로우
- 동적 요청 배치 및 스케줄링
- 시스템 수준 최적화
- 실시간 추론을 위한 양자화 접근법
- 애플리케이션 수준 최적화

### Chapter 17: 분리형 Prefill 및 Decode 스케일링

- Prefill-Decode 분리의 이점
- Prefill Worker 설계
- Decode Worker 설계
- 분리형 라우팅 및 스케줄링 정책
- 확장성 고려사항

### Chapter 18: 고급 Prefill-Decode 및 KV 캐시 튜닝

- 최적화된 Decode 커널 (FlashMLA, ThunderMLA, FlexDecoding)
- KV 캐시 활용 및 관리 튜닝
- 이기종 하드웨어 및 병렬 처리 전략
- SLO 인식 요청 관리

### Chapter 19: 동적 및 적응형 추론 엔진 최적화

- 적응형 병렬 처리 전략
- 동적 정밀도 변경
- 커널 자동 튜닝
- 런타임 튜닝을 위한 강화 학습 에이전트
- 적응형 배치 및 스케줄링

### Chapter 20: AI 지원 성능 최적화

- AlphaTensor AI 발견 알고리즘
- 자동화된 GPU 커널 최적화
- 자기 개선 AI 에이전트
- 수백만 GPU 클러스터로의 확장

## 커뮤니티 리소스

20개 이상의 도시에서 100,000명 이상의 회원과 함께하는 월간 모임:

- [YouTube 채널](https://www.youtube.com/@AIPerformanceEngineering)
- [Meetup 그룹](https://www.meetup.com/ai-performance-engineering)

최근 세션:

- [Dynamic Adaptive RL Inference CUDA Kernel Tuning](resources/Dynamic_Adaptive_RL_Inference_CUDA_Kernel_Tuning.pdf)
- [High Performance Agentic AI Inference Systems](resources/High_Performance_Agentic_AI_Inference_Systems.pdf)
- [PyTorch Model Optimization](resources/PyTorch_Model_Optimization.pdf)

### 월간 모임 요약
- **2025년 10월 20일** - [YouTube](https://youtu.be/d3ZLodGTlAo): AI 기반 GPU 커널 최적화 + nbdistributed를 사용한 분산 PyTorch
- **2025년 9월 15일** – [YouTube](https://www.youtube.com/watch?v=eLnHXL1xXfM): 동적 적응형 RL 추론 커널 튜닝 심층 분석
- **2025년 8월 18일** – [YouTube](https://www.youtube.com/watch?v=SBPlOUww57I): 멀티 GPU 오케스트레이션 전략 및 Nsight 프로파일링 사례 연구
- **2025년 7월 21일** – [YouTube](https://youtu.be/jaiMotxv8ck): FlashMLA, ThunderMLA, FlexDecoding 커널 워크스루 및 라이브 Nsight Compute 데모
- **2025년 6월 16일** – 슬라이드: [High Performance Agentic AI Inference Systems](resources/High_Performance_Agentic_AI_Inference_Systems.pdf) 분리형 추론 라우팅 다룸
- **2025년 5월 19일** – [YouTube](https://youtu.be/F8jJwI9xHTE) 및 [PyTorch Data Loader Optimization](resources/PyTorch_Model_Optimization_Data_Loader.pdf): Torch.compile 파이프라인, 데이터 로더 처리량 튜닝, 크로스 아키텍처 CUDA/ROCm 커널
- **2025년 4월 21일** – [YouTube](https://youtu.be/XoZcY_fDUKA) 및 [AI Performance Engineering Meetup Slides](resources/AI_Performance_Engineering_Meetup_Apr_21_2025.pdf): 엔드투엔드 GPU 성능 플레이북 및 [PyTorch Model Optimization](resources/PyTorch_Model_Optimization.pdf) 워크샵

## 기여하기

기여를 환영합니다! 코드, 문서, 성능 개선에 대한 가이드라인은 `CONTRIBUTING.md`를 참조하세요.

## 라이선스

MIT License – 자세한 내용은 `LICENSE`를 참조하세요.
