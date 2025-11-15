# AI Performance Engineering - 한국어 번역 프로젝트

## 프로젝트 개요

본 문서는 AI Performance Engineering 레포지토리의 한국어 번역 전략, 가이드라인, 진행 상황을 관리하기 위한 문서입니다. 이 프로젝트의 목표는 한국어 사용자들이 GPU 최적화, 분산 학습, 추론 스케일링 등 AI 시스템 성능 엔지니어링을 쉽게 학습할 수 있도록 포괄적인 한국어 문서를 제공하는 것입니다.

## 번역 철학 및 원칙

### 1. 기술 용어 처리
- **유지**: CUDA, GPU, tensor, kernel, pipeline 등 널리 알려진 기술 용어는 원문 유지
- **병기**: 처음 등장 시 한국어(영문) 형태로 병기 (예: 커널(kernel))
- **번역**: 일반적인 개념은 자연스러운 한국어로 번역 (예: performance → 성능)

### 2. 코드 및 명령어
- 모든 코드 블록, 파일명, 명령어는 **원문 그대로 유지**
- 주석은 한국어로 번역 가능
- 변수명, 함수명 등은 변경하지 않음

### 3. 링크 및 참조
- 외부 링크는 원문 유지
- 내부 문서 참조는 한국어 번역본으로 업데이트
- 영문 원본 참조 링크도 함께 제공

### 4. 번역 품질
- 기술적 정확성 최우선
- 자연스러운 한국어 표현 사용
- 문맥을 고려한 번역 (직역 지양)
- 일관된 용어 사용

## 파일 명명 규칙

한국어 번역 파일은 다음 규칙을 따릅니다:

```
원본 파일          → 한국어 번역 파일
README.md          → README.ko.md
CONTRIBUTING.md    → CONTRIBUTING.ko.md
docs/environment.md → docs/environment.ko.md
code/ch1/README.md → code/ch1/README.ko.md
```

## 번역 우선순위

### Phase 1: 핵심 문서 (High Priority)
1. ✅ `README.md` → `README.ko.md` - 프로젝트 소개 및 개요
2. ✅ `CONTRIBUTING.md` → `CONTRIBUTING.ko.md` - 기여 가이드라인
3. ✅ `code/README.md` → `code/README.ko.md` - 코드 시작 가이드
4. ✅ `docs/environment.md` → `docs/environment.ko.md` - 환경 설정

### Phase 2: 기술 가이드 (Medium Priority)
5. `docs/tooling-and-profiling.md` → `docs/tooling-and-profiling.ko.md`
6. `code/common/README.md` → `code/common/README.ko.md`
7. `code/docs/benchmark_harness_guide.md` → `code/docs/benchmark_harness_guide.ko.md`

### Phase 3: 챕터 문서 (Medium Priority)
8. `code/ch1/README.md` → `code/ch1/README.ko.md` (Performance Basics)
9. `code/ch2/README.md` → `code/ch2/README.ko.md` (GPU Hardware)
10. `code/ch6/README.md` → `code/ch6/README.ko.md` (CUDA Fundamentals)
... (20개 챕터 순차 진행)

### Phase 4: 레퍼런스 문서 (Lower Priority)
- `docs/appendix.md` → `docs/appendix.ko.md` (200+ 항목 체크리스트)
- 기타 보조 문서들

## 주요 기술 용어 매핑

| 영문 | 한국어 | 비고 |
|------|--------|------|
| GPU | GPU | 그대로 사용 |
| Kernel | 커널 | 병기 권장 |
| Thread | 스레드 | 병기 권장 |
| Warp | 워프 | 병기 권장 |
| Performance | 성능 | |
| Optimization | 최적화 | |
| Profiling | 프로파일링 | |
| Benchmark | 벤치마크 | |
| Throughput | 처리량/스루풋 | 문맥에 따라 |
| Latency | 지연시간/레이턴시 | 문맥에 따라 |
| Memory Coalescing | 메모리 병합 | 병기 권장 |
| Occupancy | 점유율 | 병기 권장 |
| Tensor Core | 텐서 코어 | 병기 권장 |
| Flash Attention | Flash Attention | 고유명사로 취급 |
| Distributed Training | 분산 학습 | |
| Inference | 추론 | |
| Quantization | 양자화 | |
| Mixed Precision | 혼합 정밀도 | |
| Pipeline | 파이프라인 | 병기 권장 |
| Batch | 배치 | |
| Stream | 스트림 | 병기 권장 |
| Asynchronous | 비동기 | |
| Synchronization | 동기화 | |
| Shared Memory | 공유 메모리 | |
| Global Memory | 전역 메모리 | |
| Register | 레지스터 | |

## 번역 워크플로우

1. **준비 단계**
   - 원본 파일 읽기 및 구조 파악
   - 기술 용어 식별 및 매핑 확인
   - 참조 링크 및 의존성 확인

2. **번역 단계**
   - 섹션별 번역 진행
   - 코드 블록 및 명령어 원문 유지
   - 용어 일관성 유지
   - 내부 링크 업데이트

3. **검토 단계**
   - 기술적 정확성 검증
   - 자연스러운 한국어 표현 확인
   - 용어 일관성 검증
   - 링크 동작 확인

4. **배포 단계**
   - 한국어 파일 생성 (.ko.md)
   - Git commit 및 push
   - 진행 상황 업데이트

## 품질 보증 체크리스트

- [ ] 모든 코드 블록이 원문 그대로 유지되었는가?
- [ ] 기술 용어가 일관되게 번역/유지되었는가?
- [ ] 내부 문서 링크가 올바르게 업데이트되었는가?
- [ ] 마크다운 포맷팅이 올바른가?
- [ ] 명령어 예시가 그대로 유지되었는가?
- [ ] 테이블 구조가 올바르게 유지되었는가?
- [ ] 번역이 기술적으로 정확한가?
- [ ] 자연스러운 한국어 표현인가?

## 진행 상황 추적

### Phase 1 진행률: 4/4 ✅ 완료
- [x] README.ko.md
- [x] CONTRIBUTING.ko.md
- [x] code/README.ko.md
- [x] docs/environment.ko.md

### Phase 2 진행률: 3/3 ✅ 완료
- [x] docs/tooling-and-profiling.ko.md
- [x] code/common/README.ko.md
- [x] code/docs/benchmark_harness_guide.ko.md

### Phase 3 진행률: 0/20
- [ ] code/ch1/README.ko.md through code/ch20/README.ko.md

### Phase 4 진행률: 0/2
- [ ] docs/appendix.ko.md
- [ ] 기타 문서들

## 커뮤니티 참여

한국어 번역 프로젝트에 기여하고 싶으신 분들은:
1. 이 문서의 가이드라인을 숙지해 주세요
2. `agents.md`에서 워크플로우를 확인해 주세요
3. Issue를 생성하여 번역할 문서를 선택해 주세요
4. PR을 제출할 때 체크리스트를 완료해 주세요

## 문의 및 피드백

- 번역 관련 질문: GitHub Issue 생성
- 용어 제안: GitHub Discussion 활용
- 오류 보고: Pull Request 또는 Issue

---

**마지막 업데이트**: 2025-11-15
**관리자**: AI Performance Engineering Korean Translation Team
