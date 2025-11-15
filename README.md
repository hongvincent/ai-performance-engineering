# AI 성능 엔지니어링 가이드 🚀

> OpenAI API와 LangChain/LangGraph를 활용한 고성능 AI 시스템 구축을 위한 종합 가이드

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Language: Korean](https://img.shields.io/badge/Language-한국어-red.svg)](README.md)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)

---

## 소개

이 레포지토리는 **한국 개발자들이 AI 성능 엔지니어링을 쉽게 학습**할 수 있도록 제작된 종합 가이드입니다. **OpenAI API**와 **LangChain/LangGraph**를 활용한 최신 AI 애플리케이션 개발 방법을 다룹니다.

### ✨ 2025년 최신 기술 스택

- 🔥 **OpenAI GPT-4.1 시리즈**: 1M 토큰 컨텍스트, 향상된 성능
- 🛠️ **LangChain 1.0 & LangGraph 1.0**: 안정적인 프로덕션 프레임워크
- ⚡ **최신 모델**: GPT-4.1, GPT-4.1 mini, GPT-4.1 nano
- 🎯 **실전 예제**: 즉시 실행 가능한 코드 제공

### 왜 이 가이드가 필요한가요?

- 📚 **한국어로 작성된 실전 중심 콘텐츠**: 번역투가 아닌 자연스러운 한국어 설명
- 💡 **즉시 사용 가능한 코드 예제**: 복사-붙여넣기로 바로 테스트 가능
- 🎯 **성능과 비용 최적화**: 실무에서 바로 적용 가능한 최적화 기법
- 🔧 **단계별 학습 구조**: 기초부터 고급까지 체계적 학습 경로
- 🆕 **최신 기술**: 2025년 1월 기준 최신 모델 및 프레임워크

---

## 목차

1. [빠른 시작](#빠른-시작)
2. [학습 가이드](#학습-가이드)
3. [주요 문서](#주요-문서)
4. [실전 예제](#실전-예제)
5. [프로젝트 구조](#프로젝트-구조)
6. [라이선스](#라이선스)

---

## 빠른 시작

### 사전 요구사항

- **Python 3.10 이상** (LangGraph 요구사항)
- **OpenAI API 키** ([발급 받기](https://platform.openai.com/api-keys))

### 설치

```bash
# 레포지토리 클론
git clone https://github.com/hongvincent/ai-performance-engineering.git
cd ai-performance-engineering

# 의존성 설치
pip install openai langchain langchain-openai langgraph

# 환경 변수 설정
export OPENAI_API_KEY="your-api-key-here"
```

### 첫 번째 예제 실행

```python
from openai import OpenAI

client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "당신은 도움이 되는 AI 어시스턴트입니다."},
        {"role": "user", "content": "안녕하세요!"}
    ]
)

print(response.choices[0].message.content)
```

### 예제 실행

```bash
# 기본 채팅 테스트
python examples/01_basic_chat.py

# 스트리밍 채팅
python examples/02_streaming_chat.py

# JSON 모드 (구조화된 출력)
python examples/03_json_mode.py
```

---

## 학습 가이드

### 학습 경로

```
1. OpenAI API 기초 (openai.md)
   ↓
2. LangChain & LangGraph (agents-langgraph.md)
   ↓
3. 실전 프로젝트 (examples/)
   ↓
4. 성능 최적화
```

### 난이도별 학습

#### 🟢 초급 (1-2주)
- OpenAI API 기본 사용법
- 프롬프트 엔지니어링 기초
- 간단한 채팅 애플리케이션

**추천 섹션:**
- [openai.md - OpenAI API 기본 개념](openai.md#openai-api-기본-개념)
- [openai.md - 프롬프트 엔지니어링](openai.md#프롬프트-엔지니어링)
- [examples/01_basic_chat.py](examples/01_basic_chat.py)

#### 🟡 중급 (2-4주)
- LangChain으로 고급 Chain 구축
- Function Calling 활용
- 비동기 처리 및 배치 작업

**추천 섹션:**
- [openai.md - 성능 최적화 전략](openai.md#성능-최적화-전략)
- [agents-langgraph.md - LangChain 기초](agents-langgraph.md#langchain-기초)
- [openai.md - 비용 최적화](openai.md#비용-최적화)

#### 🔴 고급 (4주 이상)
- LangGraph로 멀티 에이전트 시스템 설계
- RAG (Retrieval-Augmented Generation)
- 프로덕션 환경 구축 및 모니터링

**추천 섹션:**
- [agents-langgraph.md - LangGraph 1.0 소개](agents-langgraph.md#langgraph-10-소개)
- [agents-langgraph.md - 에이전트 패턴](agents-langgraph.md#에이전트-패턴)
- [agents-langgraph.md - 실전 구현](agents-langgraph.md#실전-구현)

---

## 주요 문서

### 📘 [openai.md](openai.md) - OpenAI API를 활용한 AI 성능 엔지니어링

OpenAI API를 최대한 활용하기 위한 완벽 가이드 (2025년 최신 모델 반영)

**주요 내용:**
- ✅ OpenAI API 기본 개념 및 인증
- ✅ **최신 모델 가이드** (GPT-4.1, GPT-4o, GPT-3.5 Turbo)
- ✅ 모델별 가격 및 성능 비교
- ✅ 프롬프트 엔지니어링 베스트 프랙티스
- ✅ JSON 모드 & Function Calling
- ✅ 스트리밍 & 비동기 처리
- ✅ 토큰 최적화 전략
- ✅ 비용 최적화 및 모니터링
- ✅ 실전 예제 (챗봇, 배치 처리, 분류 시스템)

**예제 코드:**
```python
# 비동기 배치 처리
processor = BatchProcessor(model="gpt-4.1-nano")
results = await processor.process_batch(prompts, batch_size=10)

# JSON 모드로 구조화된 출력
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}],
    response_format={"type": "json_object"}
)

# 스트리밍
for chunk in client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "안녕하세요"}],
    stream=True
):
    print(chunk.choices[0].delta.content, end="")
```

### 🤖 [agents-langgraph.md](agents-langgraph.md) - LangChain & LangGraph를 활용한 AI 에이전트 개발

LangChain 1.0 & LangGraph 1.0 기반 최신 AI 에이전트 개발 가이드

**주요 내용:**
- ✅ **LangChain 1.0 & LangGraph 1.0** 완벽 가이드
- ✅ LangChain 기초 (Prompts, Chains, Tools)
- ✅ LangGraph 핵심 개념 (State, Nodes, Edges)
- ✅ **ReAct 에이전트** 구현
- ✅ **멀티 에이전트 협업** 시스템
- ✅ **RAG** (Retrieval-Augmented Generation)
- ✅ 메모리를 가진 대화형 에이전트
- ✅ 성능 최적화 (스트리밍, 비동기, 캐싱)

**예제 코드:**
```python
# LangGraph 기반 에이전트
from langgraph.graph import StateGraph, END

workflow = StateGraph(AgentState)
workflow.add_node("agent", call_agent)
workflow.add_node("tools", tool_node)

app = workflow.compile()
result = app.invoke({"messages": [HumanMessage(content="안녕하세요")]})

# RAG 구현
from langchain.chains import RetrievalQA

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever()
)
answer = qa_chain.invoke("질문")

# 멀티 에이전트 협업
multi_agent_app = create_multi_agent_system()
result = multi_agent_app.invoke({"messages": ["주제"]})
```

---

## 실전 예제

### 📁 examples/ 디렉토리

모든 예제는 즉시 실행 가능하며, 실제 OpenAI API를 호출합니다.

#### 1. 기본 채팅 ([examples/01_basic_chat.py](examples/01_basic_chat.py))

OpenAI API를 사용한 가장 기본적인 대화 예제

```bash
python examples/01_basic_chat.py
```

**기능:**
- 다양한 질문에 대한 응답 테스트
- 토큰 사용량 추적
- 에러 처리

#### 2. 스트리밍 채팅 ([examples/02_streaming_chat.py](examples/02_streaming_chat.py))

실시간으로 응답을 받는 스트리밍 예제

```bash
python examples/02_streaming_chat.py
```

**기능:**
- 실시간 응답 스트리밍
- TTFB (Time To First Byte) 최소화
- 사용자 경험 개선

#### 3. JSON 모드 ([examples/03_json_mode.py](examples/03_json_mode.py))

구조화된 JSON 출력을 받는 예제

```bash
python examples/03_json_mode.py
```

**기능:**
- 감정 분석
- 구조화된 데이터 추출
- JSON 스키마 정의

---

## 프로젝트 구조

```
ai-performance-engineering/
│
├── README.md                      # 메인 문서 (이 파일)
├── openai.md                      # OpenAI API 가이드
├── agents-langgraph.md            # LangChain/LangGraph 가이드
├── .env.example                   # 환경 변수 예제
│
├── examples/                      # 실행 가능한 예제 코드
│   ├── 01_basic_chat.py          # 기본 채팅
│   ├── 02_streaming_chat.py      # 스트리밍 채팅
│   └── 03_json_mode.py           # JSON 모드
│
├── tutorials/                     # 단계별 튜토리얼 (예정)
│   ├── 01_getting_started.md
│   ├── 02_prompt_engineering.md
│   └── 03_agents.md
│
├── benchmarks/                    # 성능 벤치마크 (예정)
│   └── model_comparison.py
│
└── utils/                         # 유틸리티 함수 (예정)
    ├── monitoring.py
    └── cost_tracker.py
```

---

## 성능 벤치마크

### OpenAI 모델 비교 (2025년 기준)

| 모델 | 컨텍스트 | 입력 비용 | 출력 비용 | 속도 | 추천 용도 |
|-----|---------|----------|----------|------|----------|
| **GPT-4.1** | 1M | 높음 | 높음 | 중간 | 복잡한 추론, 코딩 |
| **GPT-4.1 mini** | 1M | 낮음 | 낮음 | 빠름 | 범용 작업 |
| **GPT-4.1 nano** | 1M | 매우 낮음 | 매우 낮음 | 매우 빠름 | 대량 처리 |
| **GPT-4o** | 128K | 중간 | 중간 | 빠름 | 멀티모달 |
| **GPT-4o mini** | 128K | 매우 낮음 | 매우 낮음 | 빠름 | 간단한 작업 |
| **GPT-3.5 Turbo** | 16K | 낮음 | 낮음 | 빠름 | 레거시 지원 |

### 가격 정보 (1M 토큰 기준)

| 모델 | 입력 | 출력 |
|-----|------|------|
| GPT-3.5 Turbo | $0.50 | $1.50 |
| GPT-4o mini | $0.15 | $0.60 |
| GPT-4 Turbo | $10.00 | $10.00 |
| GPT-4 | $30.00 | $60.00 |

---

## 학습 리소스

### 공식 문서
- [OpenAI 공식 문서](https://platform.openai.com/docs)
- [OpenAI API 레퍼런스](https://platform.openai.com/docs/api-reference)
- [LangChain 문서](https://docs.langchain.com/)
- [LangGraph 문서](https://docs.langchain.com/oss/python/langgraph/overview)

### 추천 읽을거리
- [OpenAI Cookbook](https://cookbook.openai.com/)
- [LangChain Blog](https://blog.langchain.com/)
- [Prompt Engineering Guide](https://www.promptingguide.ai/)

---

## FAQ

### Q1. OpenAI API 키는 어떻게 발급받나요?

1. [OpenAI Platform](https://platform.openai.com/)에 접속
2. 계정 생성 또는 로그인
3. API Keys 메뉴에서 새 키 생성
4. **결제 정보 등록** (필수)
5. 환경 변수에 설정: `export OPENAI_API_KEY="your-key"`

⚠️ **보안 주의**: API 키를 코드에 직접 하드코딩하지 마세요. 환경 변수를 사용하세요.

### Q2. 어떤 모델을 선택해야 하나요?

- **간단한 작업** (분류, 요약): GPT-4o mini 또는 GPT-4.1 nano
- **일반적인 대화, 코딩**: GPT-4.1 mini
- **복잡한 분석, 추론**: GPT-4.1
- **이미지/오디오 처리**: GPT-4o

자세한 내용은 [openai.md - 모델 선택 가이드](openai.md#최신-모델-가이드-2025)를 참고하세요.

### Q3. 비용을 어떻게 절감할 수 있나요?

1. **작업에 맞는 모델 선택**: 간단한 작업에 nano/mini 사용
2. **토큰 최적화**: 불필요한 내용 제거
3. **프롬프트 최적화**: 간결하고 명확하게
4. **배치 처리**: 여러 요청을 효율적으로 처리

자세한 내용은 [openai.md - 비용 최적화](openai.md#비용-최적화)를 참고하세요.

### Q4. LangChain과 순수 OpenAI API의 차이는?

- **순수 OpenAI API**: 직접적인 제어, 낮은 추상화
- **LangChain**: 고수준 추상화, 빠른 프로토타이핑
- **LangGraph**: 복잡한 에이전트 워크플로우, 상태 관리

간단한 작업은 순수 API, 복잡한 에이전트는 LangGraph를 권장합니다.

### Q5. 실제 프로덕션 환경에서 주의할 점은?

1. **에러 처리**: 재시도 로직 구현
2. **레이트 리미트**: 요청 제한 준수
3. **비용 모니터링**: 사용량 추적
4. **보안**: API 키 안전하게 관리
5. **로깅**: LangSmith 등 모니터링 도구 활용

---

## 기여하기

이 프로젝트는 커뮤니티의 기여를 환영합니다!

### 기여 방법

1. **이슈 제기**: 버그 발견 또는 개선 제안
2. **Pull Request**: 코드 개선, 예제 추가, 문서 수정
3. **문서 번역**: 다른 언어로 번역 지원
4. **예제 공유**: 실전 사용 사례 공유

---

## 버전 히스토리

### v2.0.0 (2025-01-15)
- 🔥 **OpenAI API 기반으로 전면 전환**
- ✨ 최신 모델 반영 (GPT-4.1 시리즈)
- 🛠️ LangChain 1.0 & LangGraph 1.0 지원
- 📦 실행 가능한 예제 코드 추가
- 📚 문서 전면 재작성

### v1.0.0 (2024-11-15)
- 초기 릴리스 (Claude API 기반)

---

## 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.

```
MIT License

Copyright (c) 2025 AI Performance Engineering

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 문의 및 지원

- **GitHub Issues**: [이슈 생성하기](https://github.com/hongvincent/ai-performance-engineering/issues)
- **GitHub Discussions**: [토론 참여하기](https://github.com/hongvincent/ai-performance-engineering/discussions)

---

## 감사의 말

이 프로젝트는 다음의 도움으로 만들어졌습니다:

- [OpenAI](https://openai.com/) - GPT API 제공
- [LangChain](https://www.langchain.com/) - 에이전트 프레임워크
- 한국 AI 개발자 커뮤니티 - 피드백 및 제안
- 오픈소스 기여자들 - 코드 및 문서 개선

---

<div align="center">

**AI 성능 엔지니어링과 함께 더 나은 AI 시스템을 구축하세요!** 🚀

[OpenAI 가이드](openai.md) | [LangGraph 가이드](agents-langgraph.md) | [예제 보기](examples/)

⭐ 도움이 되었다면 Star를 눌러주세요!

Made with ❤️ for Korean Developers

</div>
