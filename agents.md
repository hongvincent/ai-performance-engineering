# 한국어 번역 프로젝트 - Agent 워크플로우 및 협업 가이드

## 개요

본 문서는 AI Performance Engineering 레포지토리의 한국어 번역 작업을 체계적으로 수행하기 위한 Agent 기반 워크플로우와 협업 방법론을 정의합니다.

## Agent 역할 정의

### 1. Translation Agent (번역 에이전트)
**책임 범위:**
- 원본 영문 문서를 한국어로 번역
- 기술 용어 일관성 유지
- 마크다운 포맷팅 보존
- 코드 블록 및 명령어 원문 유지

**입력:**
- 원본 .md 파일
- 기술 용어 매핑 테이블 (claude.md 참조)
- 번역 가이드라인

**출력:**
- .ko.md 한국어 번역 파일
- 번역 과정에서 발견한 이슈 로그

**워크플로우:**
1. 원본 파일 읽기
2. 섹션별 구조 분석
3. 기술 용어 및 코드 블록 식별
4. 단락별 번역 수행
5. 내부 링크 업데이트
6. 최종 검토 및 파일 생성

---

### 2. Review Agent (검토 에이전트)
**책임 범위:**
- 번역 품질 검증
- 기술적 정확성 확인
- 용어 일관성 검증
- 마크다운 문법 검사

**체크리스트:**
- [ ] 모든 코드 블록이 변경되지 않았는가?
- [ ] 파일 경로 및 명령어가 원문 그대로인가?
- [ ] 기술 용어가 가이드라인에 따라 처리되었는가?
- [ ] 내부 링크가 올바르게 업데이트되었는가?
- [ ] 테이블 구조가 유지되었는가?
- [ ] 번호 매기기 및 들여쓰기가 올바른가?
- [ ] 외부 링크가 동작하는가?
- [ ] 한국어 표현이 자연스러운가?

**도구:**
- Markdown linter
- 기술 용어 사전
- 링크 검증 도구

---

### 3. Coordination Agent (조정 에이전트)
**책임 범위:**
- 번역 작업 우선순위 관리
- 진행 상황 추적
- 의존성 관리 (예: ch1이 완료되어야 ch2 시작 가능)
- 중복 작업 방지

**태스크 관리:**
```
Phase 1: 핵심 문서 (필수 선행)
├─ README.md → README.ko.md
├─ CONTRIBUTING.md → CONTRIBUTING.ko.md
├─ code/README.md → code/README.ko.md
└─ docs/environment.md → docs/environment.ko.md

Phase 2: 기술 가이드 (Phase 1 완료 후)
├─ docs/tooling-and-profiling.md
├─ code/common/README.md
└─ code/docs/benchmark_harness_guide.md

Phase 3: 챕터 문서 (병렬 수행 가능)
├─ code/ch1/README.md (우선)
├─ code/ch6/README.md (CUDA 기초)
├─ code/ch10/README.md (파이프라이닝)
└─ ... (나머지 챕터들)

Phase 4: 레퍼런스 문서
└─ docs/appendix.md (200+ 체크리스트)
```

---

### 4. Integration Agent (통합 에이전트)
**책임 범위:**
- Git 커밋 및 푸시
- 번역 파일 통합
- 충돌 해결
- 브랜치 관리

**Git 워크플로우:**
```bash
# 현재 브랜치: claude/add-korean-translation-01R9FwimruNxLjiZgX8iXesY

# 1. 번역 파일 생성 후
git add README.ko.md
git commit -m "Add Korean translation for README.md

- Translate main project overview
- Preserve all code blocks and commands
- Update internal links to .ko.md files
- Maintain markdown formatting"

# 2. Phase별 푸시
git push -u origin claude/add-korean-translation-01R9FwimruNxLjiZgX8iXesY
```

**커밋 메시지 규칙:**
- 제목: 간결한 변경 사항 요약 (영문)
- 본문: 상세 번역 내용 (불릿 포인트)
- 참조: 관련 이슈 번호

---

## 번역 워크플로우 (단계별)

### Step 1: 사전 준비
```
Coordination Agent:
  → 다음 번역 대상 파일 선택 (우선순위 기반)
  → 의존성 확인 (예: 상위 README가 번역되었는지)
  → Translation Agent에게 작업 할당
```

### Step 2: 번역 수행
```
Translation Agent:
  1. Read 도구로 원본 파일 읽기
  2. 구조 분석:
     - 헤더 구조
     - 코드 블록 위치
     - 테이블 구조
     - 링크 목록
  3. 섹션별 번역:
     - 제목 번역
     - 본문 번역 (용어 매핑 적용)
     - 코드/명령어 보존
     - 링크 업데이트
  4. Write 도구로 .ko.md 파일 생성
```

### Step 3: 품질 검토
```
Review Agent:
  1. Read 도구로 원본 및 번역본 비교
  2. 체크리스트 검증:
     ✓ 코드 블록 무결성
     ✓ 용어 일관성
     ✓ 링크 유효성
     ✓ 마크다운 문법
  3. 이슈 발견 시 → Translation Agent에게 피드백
  4. 승인 시 → Integration Agent에게 전달
```

### Step 4: 통합 및 배포
```
Integration Agent:
  1. git add .ko.md 파일
  2. git commit (규칙에 따른 메시지)
  3. git push -u origin 브랜치
  4. Coordination Agent에게 완료 보고
```

### Step 5: 진행 상황 업데이트
```
Coordination Agent:
  1. claude.md의 진행 상황 체크리스트 업데이트
  2. 다음 작업 우선순위 결정
  3. 완료율 계산 및 보고
```

---

## 병렬 작업 전략

### 가능한 병렬 작업
- **챕터 문서**: ch1 ~ ch20은 상호 의존성이 낮아 병렬 번역 가능
- **독립 가이드**: tooling-and-profiling.md와 benchmark_harness_guide.md는 병렬 가능

### 순차 필수 작업
- **Phase 1 → Phase 2**: 핵심 문서가 먼저 완료되어야 함
- **상위 README → 하위 README**: code/README.md 완료 후 챕터별 README 시작

---

## 이슈 처리 프로토콜

### 용어 불일치 발견
```
Translation Agent → Coordination Agent:
  "ch3/README.md에서 'memory bandwidth'가
   ch1에서는 '메모리 대역폭', ch2에서는 '메모리 대역'으로 번역됨"

Coordination Agent:
  → claude.md 용어 매핑 테이블 업데이트
  → 관련 파일 재검토 요청
```

### 기술 용어 애매모호
```
Translation Agent → Review Agent:
  "occupancy를 '점유율'로 번역했으나,
   문맥상 '사용률'이 더 자연스러울 수 있음"

Review Agent:
  → 원문 문맥 분석
  → 커뮤니티 피드백 요청 (필요 시)
  → 최종 결정 후 claude.md 업데이트
```

### 링크 깨짐
```
Translation Agent:
  README.ko.md 생성 시 내부 링크가 영문 파일을 가리킴

Solution:
  1. 해당 참조 파일이 번역되었는지 확인
  2. 번역 완료: .ko.md로 링크 업데이트
  3. 번역 미완료:
     - 원문 링크 유지
     - TODO 코멘트 추가: <!-- TODO: Update to .ko.md when available -->
```

---

## 품질 메트릭

### 번역 품질 지표
- **완전성**: 모든 섹션이 번역되었는가?
- **정확성**: 기술적 내용이 정확한가?
- **일관성**: 용어 사용이 일관적인가?
- **가독성**: 한국어 표현이 자연스러운가?

### 진행 지표
- **Phase 완료율**: Phase 1 (0/4), Phase 2 (0/3), ...
- **전체 완료율**: 28개 문서 중 완료된 수
- **주간 번역량**: 주당 완료된 문서 수

---

## 협업 도구

### 커뮤니케이션
- **GitHub Issues**: 번역 관련 질문 및 논의
- **GitHub Discussions**: 용어 제안 및 피드백
- **Pull Request**: 번역 파일 제출 및 리뷰

### 문서 관리
- **claude.md**: 전략 및 가이드라인 중앙 저장소
- **agents.md** (본 문서): 워크플로우 및 협업 방법론
- **Progress Tracking**: claude.md의 체크리스트 활용

---

## 자동화 가능 영역

### 검증 스크립트
```bash
# 번역 파일 링크 검증
./scripts/validate_korean_links.sh

# 코드 블록 무결성 검사
./scripts/check_code_blocks.sh README.md README.ko.md

# 용어 일관성 검사
./scripts/check_terminology.py --term "occupancy" --expected "점유율"
```

### CI/CD 통합
```yaml
# .github/workflows/korean-translation-check.yml
name: Korean Translation Quality Check
on: [pull_request]
jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - name: Check code blocks unchanged
      - name: Validate internal links
      - name: Check terminology consistency
```

---

## 다음 단계

### Phase 1 시작 (현재)
1. **Translation Agent**: README.md → README.ko.md 번역
2. **Review Agent**: 번역 품질 검증
3. **Integration Agent**: Git 커밋 및 푸시
4. **Coordination Agent**: 진행 상황 업데이트

### Phase 1 완료 후
1. Phase 2 기술 가이드 번역 시작
2. 챕터 문서 병렬 번역 준비
3. 커뮤니티 피드백 수집

---

**마지막 업데이트**: 2025-11-15
**관리 Agent**: Coordination Agent
**현재 Phase**: Phase 1 (0/4)
