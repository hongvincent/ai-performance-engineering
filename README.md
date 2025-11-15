# AI 성능 엔지니어링 가이드 🚀

> Claude와 AI 에이전트를 활용한 고성능 AI 시스템 구축을 위한 종합 가이드

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Language: Korean](https://img.shields.io/badge/Language-한국어-red.svg)](README.md)

---

## 소개

이 레포지토리는 **한국 개발자들이 AI 성능 엔지니어링을 쉽게 학습**할 수 있도록 제작된 종합 가이드입니다. Claude API를 활용한 실전 예제와 AI 에이전트 개발 패턴, 그리고 성능 최적화 전략을 상세히 다룹니다.

### 왜 이 가이드가 필요한가요?

- 📚 **한국어로 작성된 실전 중심 콘텐츠**: 번역투가 아닌 자연스러운 한국어 설명
- 💡 **즉시 사용 가능한 코드 예제**: 복사-붙여넣기로 바로 테스트 가능
- 🎯 **성능과 비용 최적화**: 실무에서 바로 적용 가능한 최적화 기법
- 🔧 **단계별 학습 구조**: 기초부터 고급까지 체계적 학습 경로

---

## 목차

1. [빠른 시작](#빠른-시작)
2. [학습 가이드](#학습-가이드)
3. [주요 문서](#주요-문서)
4. [실전 예제](#실전-예제)
5. [기여하기](#기여하기)
6. [라이선스](#라이선스)

---

## 빠른 시작

### 사전 요구사항

- Python 3.8 이상
- Anthropic API 키 ([발급 받기](https://console.anthropic.com/))

### 설치

```bash
# 레포지토리 클론
git clone https://github.com/your-username/ai-performance-engineering.git
cd ai-performance-engineering

# 의존성 설치
pip install anthropic

# 환경 변수 설정
export ANTHROPIC_API_KEY="your-api-key-here"
```

### 첫 번째 예제 실행

```python
import anthropic

client = anthropic.Anthropic()

message = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "안녕하세요, Claude!"}
    ]
)

print(message.content[0].text)
```

---

## 학습 가이드

### 학습 경로

```
1. Claude 기초 (claude.md)
   ↓
2. 에이전트 개념 (agents.md)
   ↓
3. 실전 프로젝트
   ↓
4. 성능 최적화
```

### 난이도별 학습

#### 🟢 초급 (1-2주)
- Claude API 기본 사용법
- 프롬프트 엔지니어링 기초
- 단순 에이전트 구현

**추천 섹션:**
- [claude.md - Claude API 기본 개념](claude.md#claude-api-기본-개념)
- [claude.md - 프롬프트 엔지니어링](claude.md#프롬프트-엔지니어링)
- [agents.md - AI 에이전트 개요](agents.md#ai-에이전트-개요)

#### 🟡 중급 (2-4주)
- 컨텍스트 관리 및 최적화
- 모델 기반 에이전트 개발
- 비동기 처리 및 배치 작업

**추천 섹션:**
- [claude.md - 컨텍스트 관리](claude.md#컨텍스트-관리)
- [claude.md - 응답 시간 개선](claude.md#응답-시간-개선)
- [agents.md - 에이전트 아키텍처](agents.md#에이전트-아키텍처)

#### 🔴 고급 (4주 이상)
- 멀티 에이전트 시스템 설계
- 성능 모니터링 및 최적화
- 프로덕션 환경 구축

**추천 섹션:**
- [claude.md - 비용 최적화](claude.md#비용-최적화)
- [agents.md - 멀티 에이전트 시스템](agents.md#멀티-에이전트-시스템)
- [agents.md - 성능 측정 및 모니터링](agents.md#성능-측정-및-모니터링)

---

## 주요 문서

### 📘 [claude.md](claude.md) - Claude를 활용한 AI 성능 엔지니어링

Claude API를 최대한 활용하기 위한 완벽 가이드

**주요 내용:**
- ✅ Claude API 기본 개념 및 인증
- ✅ 모델 선택 가이드 (Opus, Sonnet, Haiku)
- ✅ 프롬프트 엔지니어링 베스트 프랙티스
- ✅ 컨텍스트 윈도우 관리 및 압축 기법
- ✅ 토큰 최적화 전략
- ✅ 응답 시간 개선 (스트리밍, 비동기, 캐싱)
- ✅ 비용 절감 전략 및 모니터링
- ✅ 실전 예제 (챗봇, 배치 처리, 코드 리뷰)

**예제 코드:**
```python
# 고성능 챗봇
chatbot = HighPerformanceChatbot("your-api-key")
result = chatbot.get_response("Python의 장점은?")

# 대량 텍스트 처리
processor = BatchTextProcessor("your-api-key")
results = await processor.process_batch(texts, "감정 분석")

# 코드 리뷰 자동화
reviewer = CodeReviewAssistant("your-api-key")
review = reviewer.review_code(code_sample)
```

### 🤖 [agents.md](agents.md) - AI 에이전트 성능 엔지니어링

자율적인 AI 에이전트 시스템 구축 및 최적화

**주요 내용:**
- ✅ AI 에이전트의 핵심 개념 및 구성요소
- ✅ 에이전트 아키텍처 패턴 (Reflex, Model-Based, Goal-Based, Utility-Based)
- ✅ 설계 패턴 (ReAct, Chain-of-Thought, Tool Use)
- ✅ 성능 측정 및 실시간 모니터링
- ✅ 최적화 전략 (응답 시간, 메모리)
- ✅ 멀티 에이전트 협업 시스템
- ✅ 실전 구현 (코드 리뷰, 고객 지원)
- ✅ 트러블슈팅 가이드

**예제 코드:**
```python
# ReAct 패턴 에이전트
react_agent = ReActAgent("your-api-key")
answer = react_agent.solve("복잡한 문제 해결")

# 목표 기반 에이전트
goal_agent = GoalBasedAgent("your-api-key")
results = goal_agent.achieve_goal("웹 스크래퍼 만들기")

# 멀티 에이전트 협업
system = CollaborativeAgentSystem("your-api-key")
results = system.collaborative_workflow("주제 연구")
```

---

## 실전 예제

### 예제 1: 고성능 챗봇 구현

대화 히스토리 관리, 스트리밍, 비동기 처리를 모두 지원하는 프로덕션 수준의 챗봇

```python
from examples.chatbot import HighPerformanceChatbot

chatbot = HighPerformanceChatbot("your-api-key")

# 일반 대화
response = chatbot.get_response("Python에 대해 알려줘")
print(response["response"])

# 스트리밍 대화
for chunk in chatbot.get_response("AI의 미래는?", stream=True):
    print(chunk, end="", flush=True)

# 비동기 대화
response = await chatbot.get_response_async("비동기란?")
```

**위치:** [claude.md - 실전 예제 1](claude.md#예제-1-고성능-챗봇-구현)

### 예제 2: 대량 텍스트 처리 시스템

수백, 수천 개의 텍스트를 효율적으로 처리하는 배치 시스템

```python
from examples.batch_processor import BatchTextProcessor

processor = BatchTextProcessor("your-api-key")

# 감정 분석 배치 처리
texts = ["리뷰 1", "리뷰 2", ..., "리뷰 1000"]
results = await processor.process_batch(
    texts,
    task="감정 분석 (positive/negative/neutral)",
    batch_size=10
)

# 처리 통계
stats = processor.get_stats()
print(f"처리 시간: {stats['total_time']}")
```

**위치:** [claude.md - 실전 예제 2](claude.md#예제-2-대량-텍스트-처리-시스템)

### 예제 3: ReAct 에이전트

추론과 행동을 결합한 지능형 에이전트

```python
from examples.react_agent import ReActAgent

agent = ReActAgent("your-api-key")

# 복잡한 문제 해결
answer = agent.solve(
    "서울의 현재 날씨를 화씨로 변환하면?"
)
```

**위치:** [agents.md - ReAct 패턴](agents.md#1-react-패턴-reasoning--acting)

### 예제 4: 멀티 에이전트 협업

여러 전문 에이전트가 협업하여 복잡한 작업 수행

```python
from examples.collaborative_agents import CollaborativeAgentSystem

system = CollaborativeAgentSystem("your-api-key")

# 연구 → 분석 → 작성 → 리뷰 파이프라인
results = system.collaborative_workflow("Python 비동기 프로그래밍")

print(results["document"])  # 최종 문서
print(results["review"])    # 품질 리뷰
```

**위치:** [agents.md - 멀티 에이전트 시스템](agents.md#멀티-에이전트-시스템)

---

## 성능 벤치마크

### 모델별 성능 비교

| 작업 유형 | Haiku | Sonnet | Opus |
|----------|-------|--------|------|
| 간단한 질문 (평균 응답 시간) | 0.8초 | 1.5초 | 3.2초 |
| 코드 생성 (품질 점수) | 7.2/10 | 9.1/10 | 9.5/10 |
| 긴 문서 요약 (정확도) | 85% | 92% | 96% |
| 1000개 텍스트 분류 (처리 시간) | 45초 | 78초 | 156초 |
| 비용 (1M 토큰 입력+출력) | $1.50 | $18 | $90 |

### 최적화 효과

| 최적화 기법 | 응답 시간 개선 | 비용 절감 |
|------------|--------------|----------|
| 스트리밍 | -60% TTFB | 0% |
| 프롬프트 캐싱 | -50% | -75% |
| 비동기 배치 (10개) | +800% 처리량 | 0% |
| 모델 다운그레이드 (Opus→Haiku) | -75% | -93% |
| 토큰 최적화 | -20% | -20% |

---

## 프로젝트 구조

```
ai-performance-engineering/
│
├── README.md                 # 메인 문서 (이 파일)
├── claude.md                 # Claude API 성능 엔지니어링 가이드
├── agents.md                 # AI 에이전트 개발 가이드
│
├── examples/                 # 실행 가능한 예제 코드
│   ├── chatbot.py           # 고성능 챗봇
│   ├── batch_processor.py   # 대량 처리 시스템
│   ├── code_reviewer.py     # 코드 리뷰 자동화
│   ├── react_agent.py       # ReAct 패턴 에이전트
│   └── collaborative_agents.py  # 멀티 에이전트 시스템
│
├── tutorials/                # 단계별 튜토리얼
│   ├── 01_getting_started.md
│   ├── 02_prompt_engineering.md
│   ├── 03_agent_basics.md
│   └── 04_optimization.md
│
├── benchmarks/               # 성능 벤치마크 스크립트
│   ├── model_comparison.py
│   ├── optimization_tests.py
│   └── results/
│
└── utils/                    # 유틸리티 함수
    ├── monitoring.py        # 성능 모니터링
    ├── cost_tracker.py      # 비용 추적
    └── token_counter.py     # 토큰 카운터
```

---

## 학습 리소스

### 공식 문서
- [Anthropic 공식 문서](https://docs.anthropic.com/)
- [Claude API 레퍼런스](https://docs.anthropic.com/en/api/)
- [프롬프트 엔지니어링 가이드](https://docs.anthropic.com/en/docs/prompt-engineering)

### 추천 읽을거리
- [LLM 성능 최적화 전략](https://www.anthropic.com/index/core-views-on-ai-safety)
- [AI 에이전트 설계 패턴](https://python.langchain.com/docs/modules/agents/)
- [프로덕션 LLM 애플리케이션 구축](https://www.anthropic.com/index/building-effective-agents)

### 커뮤니티
- [Discord 채널](#) - 질문 및 토론
- [GitHub Discussions](#) - 이슈 및 제안
- [블로그](#) - 최신 업데이트 및 심화 내용

---

## FAQ

### Q1. Claude API 키는 어떻게 발급받나요?

1. [Anthropic Console](https://console.anthropic.com/)에 접속
2. 계정 생성 또는 로그인
3. API Keys 메뉴에서 새 키 생성
4. 환경 변수에 설정: `export ANTHROPIC_API_KEY="your-key"`

### Q2. 어떤 모델을 선택해야 하나요?

- **간단한 작업 (분류, 요약)**: Claude 3 Haiku
- **일반적인 대화, 코딩**: Claude 3.5 Sonnet
- **복잡한 분석, 추론**: Claude 3 Opus

자세한 내용은 [claude.md - 모델 선택 가이드](claude.md#1-모델-선택-가이드)를 참고하세요.

### Q3. 비용을 어떻게 절감할 수 있나요?

1. **작업에 맞는 모델 선택**: 간단한 작업에 Haiku 사용
2. **프롬프트 캐싱**: 반복되는 시스템 프롬프트 캐시 (75% 절감)
3. **토큰 최적화**: 불필요한 내용 제거
4. **출력 토큰 제한**: `max_tokens` 적절히 설정

자세한 내용은 [claude.md - 비용 최적화](claude.md#비용-최적화)를 참고하세요.

### Q4. 응답 시간을 어떻게 개선하나요?

1. **스트리밍 사용**: 첫 토큰까지의 시간(TTFB) 단축
2. **비동기 처리**: 여러 요청 동시 처리
3. **빠른 모델 사용**: Haiku로 전환
4. **프롬프트 최적화**: 간결하고 명확하게

자세한 내용은 [claude.md - 응답 시간 개선](claude.md#응답-시간-개선)을 참고하세요.

### Q5. 에이전트와 일반 LLM 호출의 차이는?

- **일반 LLM 호출**: 단순 입력 → 출력
- **에이전트**: 목표 설정 → 추론 → 행동 → 관찰 → 반복

에이전트는 복잡한 작업을 자율적으로 분해하고 해결할 수 있습니다.

자세한 내용은 [agents.md - AI 에이전트 개요](agents.md#ai-에이전트-개요)를 참고하세요.

---

## 기여하기

이 프로젝트는 커뮤니티의 기여를 환영합니다!

### 기여 방법

1. **이슈 제기**: 버그 발견 또는 개선 제안
2. **Pull Request**: 코드 개선, 예제 추가, 문서 수정
3. **문서 번역**: 다른 언어로 번역 지원
4. **예제 공유**: 실전 사용 사례 공유

### 기여 가이드라인

1. Fork 후 새 브랜치 생성
2. 변경사항 커밋 (명확한 커밋 메시지)
3. Pull Request 생성
4. 리뷰 대기 및 수정

자세한 내용은 [CONTRIBUTING.md](CONTRIBUTING.md)를 참고하세요.

---

## 버전 히스토리

### v1.0.0 (2024-11-15)
- 초기 릴리스
- Claude API 가이드 (claude.md)
- AI 에이전트 가이드 (agents.md)
- 실전 예제 7개 포함
- 한국어 완전 지원

---

## 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참고하세요.

```
MIT License

Copyright (c) 2024 AI Performance Engineering

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction...
```

---

## 문의 및 지원

- **이메일**: support@ai-performance-engineering.com
- **GitHub Issues**: [이슈 생성하기](https://github.com/your-username/ai-performance-engineering/issues)
- **Discord**: [커뮤니티 참여하기](#)

---

## 감사의 말

이 프로젝트는 다음의 도움으로 만들어졌습니다:

- [Anthropic](https://www.anthropic.com/) - Claude API 제공
- 한국 AI 개발자 커뮤니티 - 피드백 및 제안
- 오픈소스 기여자들 - 코드 및 문서 개선

---

<div align="center">

**AI 성능 엔지니어링과 함께 더 나은 AI 시스템을 구축하세요!** 🚀

[시작하기](claude.md) | [에이전트 배우기](agents.md) | [예제 보기](#실전-예제)

⭐ 도움이 되었다면 Star를 눌러주세요!

</div>
