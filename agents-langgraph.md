# LangChain & LangGraph를 활용한 AI 에이전트 개발

## 목차
1. [개요](#개요)
2. [LangChain 기초](#langchain-기초)
3. [LangGraph 1.0 소개](#langgraph-10-소개)
4. [에이전트 패턴](#에이전트-패턴)
5. [실전 구현](#실전-구현)
6. [성능 최적화](#성능-최적화)

---

## 개요

**LangChain**과 **LangGraph**는 LLM 기반 애플리케이션과 에이전트를 구축하기 위한 최고의 프레임워크입니다.

### 2025년 최신 업데이트

- ✨ **LangChain 1.0 & LangGraph 1.0** 정식 출시
- 🔒 **안정성 보장**: 2.0까지 Breaking Changes 없음
- 📚 **통합 문서**: docs.langchain.com에서 모든 문서 제공
- 🚀 **Python 3.10+** 요구
- ⚡ **신규 기능**: 노드 캐싱, deferred nodes, pre/post hooks

---

## LangChain 기초

### 1. 설치

```bash
# 기본 설치
pip install langchain langchain-openai

# LangGraph 포함
pip install langgraph

# 전체 설치 (권장)
pip install langchain langchain-openai langgraph langchain-community
```

### 2. 기본 사용법

```python
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

# LLM 초기화
llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0.7
)

# 메시지 전송
messages = [
    SystemMessage(content="당신은 도움이 되는 AI 어시스턴트입니다."),
    HumanMessage(content="Python의 장점을 알려주세요")
]

response = llm.invoke(messages)
print(response.content)
```

### 3. 프롬프트 템플릿

```python
from langchain.prompts import ChatPromptTemplate

# 템플릿 정의
template = ChatPromptTemplate.from_messages([
    ("system", "당신은 {subject} 전문가입니다."),
    ("human", "{question}")
])

# 메시지 생성
messages = template.format_messages(
    subject="Python 프로그래밍",
    question="리스트 컴프리헨션이 뭔가요?"
)

response = llm.invoke(messages)
print(response.content)
```

### 4. Chain 사용

```python
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# 프롬프트 템플릿
prompt = PromptTemplate(
    input_variables=["product"],
    template="다음 제품에 대한 마케팅 문구를 작성하세요: {product}"
)

# Chain 생성
chain = LLMChain(llm=llm, prompt=prompt)

# 실행
result = chain.run(product="AI 챗봇 플랫폼")
print(result)
```

---

## LangGraph 1.0 소개

LangGraph는 **상태를 가진 멀티 액터 애플리케이션**을 LLM으로 구축하기 위한 라이브러리입니다.

### 핵심 개념

1. **State (상태)**: 그래프 전체에서 공유되는 데이터
2. **Nodes (노드)**: 작업을 수행하는 함수
3. **Edges (엣지)**: 노드 간의 연결
4. **Conditional Edges**: 조건부 분기

### 기본 구조

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator

# 1. 상태 정의
class AgentState(TypedDict):
    messages: Annotated[list, operator.add]
    next_action: str

# 2. 노드 함수 정의
def call_model(state: AgentState):
    """LLM 호출 노드"""
    messages = state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}

def should_continue(state: AgentState):
    """계속 진행 여부 결정"""
    last_message = state["messages"][-1]
    if "FINISH" in last_message.content:
        return "end"
    return "continue"

# 3. 그래프 구성
workflow = StateGraph(AgentState)

# 노드 추가
workflow.add_node("agent", call_model)

# 엣지 추가
workflow.set_entry_point("agent")
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "agent",
        "end": END
    }
)

# 컴파일
app = workflow.compile()

# 실행
result = app.invoke({
    "messages": [HumanMessage(content="안녕하세요!")]
})
```

---

## 에이전트 패턴

### 1. ReAct 에이전트 (Reasoning + Acting)

```python
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import Tool
from langchain_openai import ChatOpenAI

# 도구 정의
def search_tool(query: str) -> str:
    """검색 도구 (시뮬레이션)"""
    return f"'{query}'에 대한 검색 결과"

def calculator_tool(expression: str) -> str:
    """계산기 도구"""
    try:
        result = eval(expression)
        return f"결과: {result}"
    except Exception as e:
        return f"오류: {str(e)}"

tools = [
    Tool(
        name="Search",
        func=search_tool,
        description="정보를 검색할 때 사용합니다. 입력은 검색 쿼리입니다."
    ),
    Tool(
        name="Calculator",
        func=calculator_tool,
        description="수학 계산을 할 때 사용합니다. 입력은 Python 수식입니다."
    )
]

# ReAct 프롬프트
from langchain.prompts import PromptTemplate

react_prompt = PromptTemplate.from_template("""
다음 도구를 사용하여 질문에 답하세요:

{tools}

다음 형식을 사용하세요:

Question: 답해야 할 질문
Thought: 무엇을 해야 할지 생각
Action: 사용할 도구 [{tool_names}]
Action Input: 도구에 전달할 입력
Observation: 도구의 결과
... (Thought/Action/Action Input/Observation을 반복)
Thought: 이제 최종 답을 알았습니다
Final Answer: 원래 질문에 대한 최종 답변

시작!

Question: {input}
Thought: {agent_scratchpad}
""")

# 에이전트 생성
llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
agent = create_react_agent(llm, tools, react_prompt)

# AgentExecutor
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=5
)

# 실행
result = agent_executor.invoke({
    "input": "25 곱하기 4는 얼마인가요?"
})
print(result["output"])
```

### 2. LangGraph 기반 커스텀 에이전트

```python
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage, AIMessage
from typing import TypedDict, Annotated, Sequence
import operator

# 상태 정의
class AgentState(TypedDict):
    messages: Annotated[Sequence[HumanMessage | AIMessage], operator.add]

# 도구를 LangChain Tool로 변환
from langchain.tools import tool

@tool
def get_weather(location: str) -> str:
    """특정 위치의 날씨를 가져옵니다."""
    return f"{location}의 날씨: 맑음, 22°C"

@tool
def calculate(expression: str) -> float:
    """수학 계산을 수행합니다."""
    return eval(expression)

tools = [get_weather, calculate]

# 도구 실행 노드
tool_node = ToolNode(tools)

# LLM 설정 (도구 바인딩)
llm_with_tools = llm.bind_tools(tools)

# 에이전트 노드
def call_agent(state: AgentState):
    """에이전트 호출"""
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

# 조건부 엣지: 계속 또는 종료
def should_continue(state: AgentState):
    """도구 호출이 필요한지 확인"""
    last_message = state["messages"][-1]

    # 도구 호출이 있으면 계속
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "continue"

    # 없으면 종료
    return "end"

# 그래프 구성
workflow = StateGraph(AgentState)

workflow.add_node("agent", call_agent)
workflow.add_node("tools", tool_node)

workflow.set_entry_point("agent")

workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "tools",
        "end": END
    }
)

workflow.add_edge("tools", "agent")

# 컴파일
app = workflow.compile()

# 실행
response = app.invoke({
    "messages": [HumanMessage(content="서울의 날씨는 어때?")]
})

for message in response["messages"]:
    print(f"{message.type}: {message.content}")
```

### 3. 멀티 에이전트 협업

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Literal

# 상태 정의
class MultiAgentState(TypedDict):
    messages: list
    next_agent: str
    final_output: str

# 각 에이전트 노드
def researcher(state: MultiAgentState):
    """연구 에이전트"""
    messages = state["messages"]
    prompt = f"다음 주제를 조사하세요: {messages[-1]}"

    response = llm.invoke([HumanMessage(content=prompt)])

    return {
        "messages": messages + [f"[연구 결과] {response.content}"],
        "next_agent": "analyzer"
    }

def analyzer(state: MultiAgentState):
    """분석 에이전트"""
    research_result = state["messages"][-1]
    prompt = f"다음 연구 결과를 분석하세요: {research_result}"

    response = llm.invoke([HumanMessage(content=prompt)])

    return {
        "messages": state["messages"] + [f"[분석 결과] {response.content}"],
        "next_agent": "writer"
    }

def writer(state: MultiAgentState):
    """작성 에이전트"""
    analysis_result = state["messages"][-1]
    prompt = f"다음 분석을 바탕으로 보고서를 작성하세요: {analysis_result}"

    response = llm.invoke([HumanMessage(content=prompt)])

    return {
        "messages": state["messages"] + [f"[최종 보고서] {response.content}"],
        "next_agent": "end",
        "final_output": response.content
    }

# 라우터: 다음 에이전트 결정
def route_agent(state: MultiAgentState) -> Literal["researcher", "analyzer", "writer", "end"]:
    """다음 에이전트로 라우팅"""
    next_agent = state.get("next_agent", "researcher")

    if next_agent == "end":
        return END

    return next_agent

# 그래프 구성
workflow = StateGraph(MultiAgentState)

workflow.add_node("researcher", researcher)
workflow.add_node("analyzer", analyzer)
workflow.add_node("writer", writer)

workflow.set_entry_point("researcher")

workflow.add_conditional_edges(
    "researcher",
    route_agent
)
workflow.add_conditional_edges(
    "analyzer",
    route_agent
)
workflow.add_conditional_edges(
    "writer",
    route_agent
)

# 컴파일
multi_agent_app = workflow.compile()

# 실행
result = multi_agent_app.invoke({
    "messages": ["Python 비동기 프로그래밍"],
    "next_agent": "researcher"
})

print("\n=== 최종 결과 ===")
print(result["final_output"])
```

---

## 실전 구현

### 예제 1: RAG (Retrieval-Augmented Generation)

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# 1. 문서 로드 및 분할
documents = [
    "Python은 1991년 귀도 반 로섬이 개발한 프로그래밍 언어입니다.",
    "Python은 간결하고 읽기 쉬운 문법을 가지고 있습니다.",
    "Python은 데이터 과학과 AI 분야에서 널리 사용됩니다."
]

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20
)

splits = text_splitter.create_documents(documents)

# 2. 임베딩 및 벡터 저장소
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = FAISS.from_documents(splits, embeddings)

# 3. RAG Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 2})
)

# 실행
query = "Python은 누가 만들었나요?"
result = qa_chain.invoke(query)
print(result["result"])
```

### 예제 2: 메모리를 가진 대화형 에이전트

```python
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# 메모리 활성화
memory = MemorySaver()

# 상태 정의
class ConversationState(TypedDict):
    messages: Annotated[list, operator.add]
    user_info: dict

def chatbot(state: ConversationState):
    """대화 처리"""
    messages = state["messages"]
    user_info = state.get("user_info", {})

    # 사용자 정보를 컨텍스트에 포함
    context = f"사용자 정보: {user_info}\n\n"
    full_prompt = context + messages[-1].content

    response = llm.invoke([HumanMessage(content=full_prompt)])

    return {"messages": [response]}

# 그래프 구성
workflow = StateGraph(ConversationState)
workflow.add_node("chatbot", chatbot)
workflow.set_entry_point("chatbot")
workflow.add_edge("chatbot", END)

# 메모리와 함께 컴파일
app = workflow.compile(checkpointer=memory)

# 세션 ID로 대화 관리
config = {"configurable": {"thread_id": "user_123"}}

# 대화 1
result1 = app.invoke({
    "messages": [HumanMessage(content="제 이름은 홍길동입니다")],
    "user_info": {"name": "홍길동"}
}, config)

# 대화 2 (이전 컨텍스트 기억)
result2 = app.invoke({
    "messages": [HumanMessage(content="제 이름이 뭐라고 했죠?")]
}, config)

print(result2["messages"][-1].content)
```

---

## 성능 최적화

### 1. 스트리밍

```python
# LangGraph 스트리밍
for chunk in app.stream({
    "messages": [HumanMessage(content="긴 이야기를 들려주세요")]
}):
    print(chunk)
```

### 2. 비동기 실행

```python
import asyncio

async def async_agent():
    """비동기 에이전트"""
    result = await app.ainvoke({
        "messages": [HumanMessage(content="안녕하세요")]
    })
    return result

# 실행
# result = asyncio.run(async_agent())
```

### 3. 캐싱 (노드 캐싱)

LangGraph 1.0의 새로운 기능인 노드 캐싱을 사용하여 중복 계산을 피합니다.

```python
from langgraph.graph import StateGraph

# 캐싱 활성화 (자동)
workflow = StateGraph(AgentState)
# 노드 캐싱은 기본적으로 활성화됨
```

---

## 디버깅 및 모니터링

### 1. Verbose 모드

```python
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,  # 상세 로그 출력
    max_iterations=5
)
```

### 2. LangSmith (프로덕션 모니터링)

```python
import os

# LangSmith 설정
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "your-langsmith-api-key"
os.environ["LANGCHAIN_PROJECT"] = "my-project"

# 이제 모든 실행이 LangSmith에 기록됨
```

---

## 추가 리소스

- [LangChain 공식 문서](https://docs.langchain.com/)
- [LangGraph 문서](https://docs.langchain.com/oss/python/langgraph/overview)
- [LangSmith](https://smith.langchain.com/)

---

**작성일**: 2025-01-15
**버전**: 1.0 (LangChain 1.0, LangGraph 1.0 기반)
**라이선스**: MIT
