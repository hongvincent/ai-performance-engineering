# AI 에이전트 성능 엔지니어링

## 목차
1. [AI 에이전트 개요](#ai-에이전트-개요)
2. [에이전트 아키텍처](#에이전트-아키텍처)
3. [에이전트 설계 패턴](#에이전트-설계-패턴)
4. [성능 측정 및 모니터링](#성능-측정-및-모니터링)
5. [최적화 전략](#최적화-전략)
6. [멀티 에이전트 시스템](#멀티-에이전트-시스템)
7. [실전 구현 예제](#실전-구현-예제)
8. [트러블슈팅](#트러블슈팅)

---

## AI 에이전트 개요

### 에이전트란?

AI 에이전트는 환경을 인식하고, 의사결정을 내리며, 목표를 달성하기 위해 행동하는 자율적인 시스템입니다.

```
┌─────────────────────────────────────┐
│         AI 에이전트                  │
│                                     │
│  ┌──────────┐      ┌──────────┐   │
│  │  인식    │──────▶│ 추론     │   │
│  │(Perceive)│      │(Reason)  │   │
│  └──────────┘      └──────────┘   │
│       ▲                  │         │
│       │                  ▼         │
│  ┌──────────┐      ┌──────────┐   │
│  │  환경    │◀──────│ 행동     │   │
│  │(Observe) │      │(Act)     │   │
│  └──────────┘      └──────────┘   │
└─────────────────────────────────────┘
```

### 에이전트의 핵심 구성요소

| 구성요소 | 설명 | 예시 |
|---------|------|------|
| **목표(Goal)** | 에이전트가 달성하려는 목적 | "사용자 질문에 정확히 답변하기" |
| **인식(Perception)** | 환경으로부터 정보 수집 | 사용자 입력, 센서 데이터 |
| **추론(Reasoning)** | 의사결정 및 계획 수립 | LLM 기반 추론, 규칙 엔진 |
| **행동(Action)** | 환경에 영향을 주는 실행 | API 호출, 데이터베이스 쿼리 |
| **메모리(Memory)** | 과거 경험 저장 및 활용 | 대화 히스토리, 벡터 DB |

---

## 에이전트 아키텍처

### 1. 단순 반응형 에이전트 (Simple Reflex Agent)

가장 기본적인 형태로, 현재 인식에만 기반하여 행동합니다.

```python
class SimpleReflexAgent:
    """규칙 기반 단순 에이전트"""

    def __init__(self):
        self.rules = {
            "인사": "안녕하세요! 무엇을 도와드릴까요?",
            "감사": "천만에요! 더 도움이 필요하시면 말씀해주세요.",
            "종료": "좋은 하루 되세요!"
        }

    def perceive(self, user_input: str) -> str:
        """사용자 입력 인식"""
        user_input = user_input.lower()

        if any(word in user_input for word in ["안녕", "hello", "hi"]):
            return "인사"
        elif any(word in user_input for word in ["감사", "고마워", "thanks"]):
            return "감사"
        elif any(word in user_input for word in ["종료", "exit", "bye"]):
            return "종료"
        else:
            return "기타"

    def act(self, perception: str) -> str:
        """규칙에 따라 행동"""
        return self.rules.get(perception, "죄송합니다. 이해하지 못했습니다.")

    def run(self, user_input: str) -> str:
        """에이전트 실행"""
        perception = self.perceive(user_input)
        action = self.act(perception)
        return action


# 사용 예시
agent = SimpleReflexAgent()
print(agent.run("안녕하세요"))  # "안녕하세요! 무엇을 도와드릴까요?"
```

### 2. 모델 기반 에이전트 (Model-Based Agent)

내부 상태를 유지하며 환경의 모델을 가집니다.

```python
from typing import Dict, List, Any
import anthropic

class ModelBasedAgent:
    """상태를 유지하는 모델 기반 에이전트"""

    def __init__(self, api_key: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.state = {
            "conversation_history": [],
            "user_context": {},
            "current_topic": None
        }

    def update_state(self, user_input: str, response: str):
        """내부 상태 업데이트"""
        self.state["conversation_history"].append({
            "user": user_input,
            "assistant": response
        })

        # 대화 히스토리가 너무 길면 오래된 것 제거
        if len(self.state["conversation_history"]) > 10:
            self.state["conversation_history"] = \
                self.state["conversation_history"][-10:]

    def perceive(self, user_input: str) -> Dict[str, Any]:
        """사용자 입력과 현재 상태를 함께 인식"""
        return {
            "input": user_input,
            "history": self.state["conversation_history"],
            "context": self.state["user_context"]
        }

    def reason(self, perception: Dict[str, Any]) -> str:
        """LLM을 사용한 추론"""
        # 대화 히스토리를 메시지 형식으로 변환
        messages = []
        for turn in perception["history"]:
            messages.append({"role": "user", "content": turn["user"]})
            messages.append({"role": "assistant", "content": turn["assistant"]})

        # 현재 사용자 입력 추가
        messages.append({"role": "user", "content": perception["input"]})

        # Claude에 요청
        response = self.client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            system="당신은 도움이 되는 AI 어시스턴트입니다. 대화 맥락을 고려하여 답변하세요.",
            messages=messages
        )

        return response.content[0].text

    def act(self, response: str) -> str:
        """응답 반환"""
        return response

    def run(self, user_input: str) -> str:
        """에이전트 실행 사이클"""
        # 1. 인식
        perception = self.perceive(user_input)

        # 2. 추론
        response = self.reason(perception)

        # 3. 행동
        action = self.act(response)

        # 4. 상태 업데이트
        self.update_state(user_input, action)

        return action


# 사용 예시
agent = ModelBasedAgent("your-api-key")
print(agent.run("파이썬에 대해 알려줘"))
print(agent.run("그럼 자바는?"))  # 이전 맥락을 기억함
```

### 3. 목표 기반 에이전트 (Goal-Based Agent)

명확한 목표를 가지고 계획을 수립합니다.

```python
from typing import List, Dict
import anthropic

class GoalBasedAgent:
    """목표 지향적 에이전트"""

    def __init__(self, api_key: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.goal = None
        self.plan = []
        self.completed_steps = []

    def set_goal(self, goal: str):
        """목표 설정"""
        self.goal = goal
        self.plan = []
        self.completed_steps = []

    def create_plan(self) -> List[str]:
        """목표 달성을 위한 계획 수립"""
        planning_prompt = f"""
        다음 목표를 달성하기 위한 단계별 계획을 수립해주세요:

        목표: {self.goal}

        계획을 다음 형식으로 작성하세요:
        1. [첫 번째 단계]
        2. [두 번째 단계]
        3. [세 번째 단계]
        ...

        각 단계는 구체적이고 실행 가능해야 합니다.
        """

        response = self.client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            messages=[{"role": "user", "content": planning_prompt}]
        )

        # 응답에서 단계 추출
        plan_text = response.content[0].text
        steps = []
        for line in plan_text.split('\n'):
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('-')):
                # 번호나 불릿 제거
                step = line.split('.', 1)[-1].split('-', 1)[-1].strip()
                if step:
                    steps.append(step)

        self.plan = steps
        return steps

    def execute_step(self, step: str) -> Dict[str, Any]:
        """단일 단계 실행"""
        execution_prompt = f"""
        다음 작업을 수행하세요:

        작업: {step}

        전체 목표: {self.goal}
        이미 완료된 단계: {', '.join(self.completed_steps) if self.completed_steps else '없음'}

        작업 결과를 상세히 설명해주세요.
        """

        response = self.client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=2048,
            messages=[{"role": "user", "content": execution_prompt}]
        )

        result = {
            "step": step,
            "result": response.content[0].text,
            "success": True
        }

        self.completed_steps.append(step)
        return result

    def achieve_goal(self, goal: str) -> List[Dict[str, Any]]:
        """목표 달성 프로세스 전체 실행"""
        self.set_goal(goal)

        print(f"목표 설정: {goal}\n")

        # 1. 계획 수립
        print("계획 수립 중...")
        plan = self.create_plan()

        print("\n수립된 계획:")
        for i, step in enumerate(plan, 1):
            print(f"{i}. {step}")

        # 2. 계획 실행
        results = []
        print("\n계획 실행 시작:\n")

        for i, step in enumerate(plan, 1):
            print(f"[{i}/{len(plan)}] {step}")
            result = self.execute_step(step)
            results.append(result)
            print(f"✓ 완료\n")

        print(f"목표 달성 완료!")
        return results


# 사용 예시
agent = GoalBasedAgent("your-api-key")
results = agent.achieve_goal("Python으로 간단한 웹 스크래퍼 만들기")

for result in results:
    print(f"\n단계: {result['step']}")
    print(f"결과: {result['result'][:200]}...")
```

### 4. 유틸리티 기반 에이전트 (Utility-Based Agent)

여러 선택지 중 최선을 평가하여 선택합니다.

```python
from typing import List, Dict, Tuple
import anthropic

class UtilityBasedAgent:
    """효용 함수를 사용하는 에이전트"""

    def __init__(self, api_key: str):
        self.client = anthropic.Anthropic(api_key=api_key)

    def generate_options(self, situation: str) -> List[str]:
        """주어진 상황에 대한 선택지 생성"""
        prompt = f"""
        다음 상황에서 가능한 행동 선택지 5가지를 제안해주세요:

        상황: {situation}

        각 선택지를 다음 형식으로 작성하세요:
        1. [선택지 1]
        2. [선택지 2]
        ...
        """

        response = self.client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}]
        )

        # 선택지 파싱
        options = []
        for line in response.content[0].text.split('\n'):
            line = line.strip()
            if line and line[0].isdigit():
                option = line.split('.', 1)[-1].strip()
                if option:
                    options.append(option)

        return options

    def evaluate_option(self, situation: str, option: str) -> float:
        """선택지의 효용 평가 (0-10점)"""
        eval_prompt = f"""
        다음 상황에서 이 선택지의 효용성을 0-10점으로 평가해주세요:

        상황: {situation}
        선택지: {option}

        평가 기준:
        - 목표 달성 가능성 (40%)
        - 리스크 (30%)
        - 비용 효율성 (20%)
        - 실행 가능성 (10%)

        점수만 숫자로 답변하세요 (예: 8.5)
        """

        response = self.client.messages.create(
            model="claude-3-haiku-20240307",  # 평가는 빠른 모델 사용
            max_tokens=10,
            messages=[{"role": "user", "content": eval_prompt}]
        )

        try:
            score = float(response.content[0].text.strip())
            return min(max(score, 0), 10)  # 0-10 범위로 제한
        except:
            return 5.0  # 기본값

    def choose_best_action(self, situation: str) -> Tuple[str, float, List[Dict]]:
        """최적의 행동 선택"""
        print(f"상황 분석: {situation}\n")

        # 1. 선택지 생성
        print("가능한 선택지 생성 중...")
        options = self.generate_options(situation)

        # 2. 각 선택지 평가
        print("\n선택지 평가 중...")
        evaluations = []

        for option in options:
            score = self.evaluate_option(situation, option)
            evaluations.append({
                "option": option,
                "score": score
            })
            print(f"  - {option}: {score:.1f}점")

        # 3. 최고 점수 선택지 선택
        best = max(evaluations, key=lambda x: x["score"])

        print(f"\n최선의 선택: {best['option']} ({best['score']:.1f}점)")

        return best["option"], best["score"], evaluations


# 사용 예시
agent = UtilityBasedAgent("your-api-key")

situation = "새로운 프로젝트를 시작하려는데 팀원들의 기술 스택이 다양합니다."
best_action, score, all_options = agent.choose_best_action(situation)

print(f"\n권장 행동: {best_action}")
print(f"확신도: {score}/10")
```

---

## 에이전트 설계 패턴

### 1. ReAct 패턴 (Reasoning + Acting)

추론과 행동을 번갈아 수행하는 패턴입니다.

```python
import anthropic
from typing import Dict, List, Any
import json

class ReActAgent:
    """ReAct 패턴을 구현한 에이전트"""

    def __init__(self, api_key: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.tools = {
            "search": self.search_tool,
            "calculate": self.calculate_tool,
            "get_weather": self.weather_tool
        }

    def search_tool(self, query: str) -> str:
        """검색 도구 (시뮬레이션)"""
        # 실제로는 검색 API를 호출
        return f"'{query}'에 대한 검색 결과: [시뮬레이션된 결과]"

    def calculate_tool(self, expression: str) -> str:
        """계산 도구"""
        try:
            result = eval(expression)
            return f"계산 결과: {result}"
        except Exception as e:
            return f"계산 오류: {str(e)}"

    def weather_tool(self, location: str) -> str:
        """날씨 조회 도구 (시뮬레이션)"""
        return f"{location}의 현재 날씨: 맑음, 기온 22°C"

    def reason_and_act(self, task: str, max_iterations: int = 5) -> str:
        """ReAct 루프 실행"""
        context = f"과제: {task}\n\n"
        iteration = 0

        while iteration < max_iterations:
            iteration += 1

            # Reasoning (추론)
            reasoning_prompt = f"""
{context}

사용 가능한 도구:
- search(query): 정보 검색
- calculate(expression): 수식 계산
- get_weather(location): 날씨 조회

다음 형식으로 생각하고 행동하세요:

Thought: [현재 상황 분석 및 다음 단계]
Action: [도구명(인자)]

또는 답을 알고 있다면:
Thought: [최종 답변 도출 과정]
Answer: [최종 답변]

현재 진행 상황을 고려하여 다음 단계를 결정하세요.
"""

            response = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=500,
                messages=[{"role": "user", "content": reasoning_prompt}]
            )

            response_text = response.content[0].text
            context += f"반복 {iteration}:\n{response_text}\n\n"

            # Answer가 있으면 종료
            if "Answer:" in response_text:
                answer = response_text.split("Answer:")[-1].strip()
                return answer

            # Action 파싱 및 실행
            if "Action:" in response_text:
                action_line = [line for line in response_text.split('\n')
                             if line.strip().startswith("Action:")][0]
                action_text = action_line.split("Action:")[-1].strip()

                # 도구 호출 파싱
                tool_name = action_text.split('(')[0].strip()
                tool_args = action_text.split('(')[1].split(')')[0].strip().strip('"\'')

                # 도구 실행
                if tool_name in self.tools:
                    observation = self.tools[tool_name](tool_args)
                    context += f"Observation: {observation}\n\n"
                else:
                    context += f"Observation: 오류 - '{tool_name}'은(는) 존재하지 않는 도구입니다.\n\n"

        return "최대 반복 횟수에 도달했습니다. 답변을 찾지 못했습니다."

    def solve(self, task: str) -> str:
        """과제 해결"""
        print(f"과제: {task}\n")
        print("=" * 60)

        answer = self.reason_and_act(task)

        print("=" * 60)
        print(f"\n최종 답변: {answer}")

        return answer


# 사용 예시
agent = ReActAgent("your-api-key")
agent.solve("서울의 현재 날씨가 섭씨로 몇 도인지 화씨로 변환하면 얼마인가요?")
```

### 2. Chain-of-Thought 패턴

단계별 추론을 통해 복잡한 문제를 해결합니다.

```python
import anthropic

class ChainOfThoughtAgent:
    """사고의 연쇄(Chain-of-Thought)를 사용하는 에이전트"""

    def __init__(self, api_key: str):
        self.client = anthropic.Anthropic(api_key=api_key)

    def solve_with_cot(self, problem: str) -> Dict[str, str]:
        """Chain-of-Thought를 사용한 문제 해결"""

        cot_prompt = f"""
다음 문제를 단계별로 풀어주세요:

{problem}

다음 형식으로 답변하세요:

## 문제 이해
[문제가 요구하는 것을 명확히 설명]

## 해결 단계
1. [첫 번째 단계]
2. [두 번째 단계]
3. [세 번째 단계]
...

## 최종 답변
[명확한 답변]
"""

        response = self.client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=2048,
            messages=[{"role": "user", "content": cot_prompt}]
        )

        return {
            "problem": problem,
            "reasoning": response.content[0].text,
            "model": response.model
        }

    def solve_with_few_shot_cot(self, problem: str, examples: List[Dict]) -> str:
        """Few-shot Chain-of-Thought"""

        # 예제 구성
        examples_text = ""
        for i, ex in enumerate(examples, 1):
            examples_text += f"""
예제 {i}:
문제: {ex['problem']}

풀이:
{ex['solution']}

답: {ex['answer']}

---
"""

        prompt = f"""
다음 예제를 참고하여 문제를 풀어주세요:

{examples_text}

이제 다음 문제를 같은 방식으로 풀어주세요:

문제: {problem}

풀이:
"""

        response = self.client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}]
        )

        return response.content[0].text


# 사용 예시
agent = ChainOfThoughtAgent("your-api-key")

# Zero-shot CoT
problem = "한 가게에서 사과 12개를 샀는데, 3개가 상했습니다. 상하지 않은 사과를 4명이 똑같이 나눠 가진다면 한 명당 몇 개씩 가지게 되나요?"
result = agent.solve_with_cot(problem)
print(result["reasoning"])

# Few-shot CoT
examples = [
    {
        "problem": "바구니에 빨간 공 5개와 파란 공 3개가 있습니다. 전체 공의 개수는?",
        "solution": "1. 빨간 공: 5개\n2. 파란 공: 3개\n3. 전체 = 5 + 3",
        "answer": "8개"
    }
]

new_problem = "책장에 소설 책 15권과 만화책 8권이 있습니다. 전체 책의 개수는?"
result = agent.solve_with_few_shot_cot(new_problem, examples)
print(result)
```

### 3. 도구 사용 패턴 (Tool Use)

외부 도구를 활용하여 기능을 확장합니다.

```python
import anthropic
from typing import List, Dict, Any, Callable

class ToolUseAgent:
    """도구 사용 에이전트"""

    def __init__(self, api_key: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.tools_registry: Dict[str, Callable] = {}
        self.tools_definitions: List[Dict] = []

    def register_tool(self, name: str, description: str,
                     parameters: Dict, function: Callable):
        """도구 등록"""
        self.tools_registry[name] = function
        self.tools_definitions.append({
            "name": name,
            "description": description,
            "input_schema": {
                "type": "object",
                "properties": parameters,
                "required": list(parameters.keys())
            }
        })

    def execute_tool(self, tool_name: str, tool_input: Dict) -> Any:
        """도구 실행"""
        if tool_name in self.tools_registry:
            return self.tools_registry[tool_name](**tool_input)
        else:
            return {"error": f"도구 '{tool_name}'을(를) 찾을 수 없습니다."}

    def run(self, user_message: str, max_iterations: int = 5) -> str:
        """도구를 사용하여 작업 수행"""
        messages = [{"role": "user", "content": user_message}]

        for iteration in range(max_iterations):
            # Claude에 요청
            response = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=2048,
                tools=self.tools_definitions,
                messages=messages
            )

            # 응답 처리
            if response.stop_reason == "end_turn":
                # 최종 답변
                final_response = next(
                    (block.text for block in response.content
                     if hasattr(block, "text")),
                    None
                )
                return final_response

            elif response.stop_reason == "tool_use":
                # 도구 사용 요청
                messages.append({"role": "assistant", "content": response.content})

                # 도구 실행
                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        tool_name = block.name
                        tool_input = block.input

                        print(f"[도구 사용] {tool_name}({tool_input})")

                        # 도구 실행
                        result = self.execute_tool(tool_name, tool_input)

                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": str(result)
                        })

                # 도구 결과를 메시지에 추가
                messages.append({"role": "user", "content": tool_results})

        return "최대 반복 횟수에 도달했습니다."


# 사용 예시
def get_current_time(timezone: str = "UTC") -> str:
    """현재 시간 조회"""
    from datetime import datetime
    return datetime.now().isoformat()

def search_database(query: str) -> List[Dict]:
    """데이터베이스 검색 (시뮬레이션)"""
    return [
        {"id": 1, "title": "결과 1", "content": "..."},
        {"id": 2, "title": "결과 2", "content": "..."}
    ]

def send_email(to: str, subject: str, body: str) -> Dict:
    """이메일 전송 (시뮬레이션)"""
    print(f"이메일 전송: {to} / {subject}")
    return {"status": "sent", "message_id": "12345"}


# 에이전트 생성 및 도구 등록
agent = ToolUseAgent("your-api-key")

agent.register_tool(
    name="get_current_time",
    description="현재 시간을 조회합니다.",
    parameters={
        "timezone": {
            "type": "string",
            "description": "시간대 (예: UTC, Asia/Seoul)"
        }
    },
    function=get_current_time
)

agent.register_tool(
    name="search_database",
    description="데이터베이스에서 정보를 검색합니다.",
    parameters={
        "query": {
            "type": "string",
            "description": "검색 쿼리"
        }
    },
    function=search_database
)

agent.register_tool(
    name="send_email",
    description="이메일을 전송합니다.",
    parameters={
        "to": {"type": "string", "description": "수신자 이메일"},
        "subject": {"type": "string", "description": "제목"},
        "body": {"type": "string", "description": "본문"}
    },
    function=send_email
)

# 실행
result = agent.run("현재 서울 시간을 확인하고, 'AI' 키워드로 데이터베이스를 검색해주세요.")
print(f"\n최종 결과: {result}")
```

---

## 성능 측정 및 모니터링

### 1. 주요 성능 지표

```python
import time
from typing import Dict, List
from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class PerformanceMetrics:
    """성능 지표 데이터 클래스"""
    agent_name: str
    task_name: str
    start_time: float = field(default_factory=time.time)
    end_time: float = 0
    success: bool = False
    error_message: str = ""
    llm_calls: int = 0
    total_tokens: int = 0
    tool_calls: int = 0

    def duration(self) -> float:
        """실행 시간 (초)"""
        return self.end_time - self.start_time if self.end_time > 0 else 0

    def tokens_per_second(self) -> float:
        """초당 토큰 처리량"""
        duration = self.duration()
        return self.total_tokens / duration if duration > 0 else 0


class PerformanceMonitor:
    """에이전트 성능 모니터링"""

    def __init__(self):
        self.metrics_history: List[PerformanceMetrics] = []

    def start_task(self, agent_name: str, task_name: str) -> PerformanceMetrics:
        """작업 시작"""
        metrics = PerformanceMetrics(
            agent_name=agent_name,
            task_name=task_name
        )
        return metrics

    def end_task(self, metrics: PerformanceMetrics, success: bool = True,
                 error: str = ""):
        """작업 종료"""
        metrics.end_time = time.time()
        metrics.success = success
        metrics.error_message = error
        self.metrics_history.append(metrics)

    def get_statistics(self) -> Dict:
        """통계 생성"""
        if not self.metrics_history:
            return {}

        total_tasks = len(self.metrics_history)
        successful_tasks = sum(1 for m in self.metrics_history if m.success)
        total_duration = sum(m.duration() for m in self.metrics_history)
        total_tokens = sum(m.total_tokens for m in self.metrics_history)
        total_llm_calls = sum(m.llm_calls for m in self.metrics_history)

        return {
            "total_tasks": total_tasks,
            "successful_tasks": successful_tasks,
            "success_rate": f"{(successful_tasks/total_tasks)*100:.1f}%",
            "total_duration": f"{total_duration:.2f}초",
            "average_duration": f"{total_duration/total_tasks:.2f}초",
            "total_tokens": total_tokens,
            "average_tokens": f"{total_tokens/total_tasks:.0f}",
            "total_llm_calls": total_llm_calls,
            "average_llm_calls": f"{total_llm_calls/total_tasks:.1f}"
        }

    def print_report(self):
        """성능 리포트 출력"""
        stats = self.get_statistics()

        print("\n" + "="*60)
        print("성능 모니터링 리포트")
        print("="*60)

        for key, value in stats.items():
            print(f"{key:20}: {value}")

        print("="*60 + "\n")


# 사용 예시
monitor = PerformanceMonitor()

# 작업 1
metrics1 = monitor.start_task("SearchAgent", "데이터 검색")
# ... 작업 수행 ...
metrics1.llm_calls = 2
metrics1.total_tokens = 1500
metrics1.tool_calls = 1
time.sleep(0.5)  # 시뮬레이션
monitor.end_task(metrics1, success=True)

# 작업 2
metrics2 = monitor.start_task("AnalysisAgent", "데이터 분석")
# ... 작업 수행 ...
metrics2.llm_calls = 3
metrics2.total_tokens = 2500
time.sleep(0.8)  # 시뮬레이션
monitor.end_task(metrics2, success=True)

# 리포트 출력
monitor.print_report()
```

### 2. 실시간 모니터링 대시보드

```python
from collections import deque
from typing import Deque
import threading
import time

class RealtimeMonitor:
    """실시간 성능 모니터링"""

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.latencies: Deque[float] = deque(maxlen=window_size)
        self.throughputs: Deque[float] = deque(maxlen=window_size)
        self.error_count = 0
        self.total_requests = 0
        self.lock = threading.Lock()

    def record_request(self, latency: float, tokens: int, success: bool = True):
        """요청 기록"""
        with self.lock:
            self.latencies.append(latency)
            self.throughputs.append(tokens / latency if latency > 0 else 0)
            self.total_requests += 1

            if not success:
                self.error_count += 1

    def get_current_stats(self) -> Dict:
        """현재 통계"""
        with self.lock:
            if not self.latencies:
                return {}

            return {
                "avg_latency": f"{sum(self.latencies)/len(self.latencies):.3f}초",
                "p95_latency": f"{sorted(self.latencies)[int(len(self.latencies)*0.95)]:.3f}초",
                "p99_latency": f"{sorted(self.latencies)[int(len(self.latencies)*0.99)]:.3f}초",
                "avg_throughput": f"{sum(self.throughputs)/len(self.throughputs):.0f} tokens/s",
                "error_rate": f"{(self.error_count/max(self.total_requests,1))*100:.2f}%",
                "total_requests": self.total_requests
            }

    def print_dashboard(self):
        """대시보드 출력"""
        stats = self.get_current_stats()

        if not stats:
            print("데이터 없음")
            return

        print("\n" + "="*60)
        print(f"실시간 모니터링 대시보드 ({datetime.now().strftime('%H:%M:%S')})")
        print("="*60)

        for key, value in stats.items():
            print(f"{key:20}: {value}")

        print("="*60 + "\n")


# 사용 예시
monitor = RealtimeMonitor()

# 요청 시뮬레이션
import random

for i in range(20):
    latency = random.uniform(0.5, 2.0)
    tokens = random.randint(500, 2000)
    success = random.random() > 0.1  # 90% 성공률

    monitor.record_request(latency, tokens, success)

    if i % 5 == 0:
        monitor.print_dashboard()

    time.sleep(0.1)
```

---

## 최적화 전략

### 1. 응답 시간 최적화

```python
import anthropic
import asyncio
from typing import List

class OptimizedAgent:
    """최적화된 에이전트"""

    def __init__(self, api_key: str):
        self.sync_client = anthropic.Anthropic(api_key=api_key)
        self.async_client = anthropic.AsyncAnthropic(api_key=api_key)

    # 전략 1: 스트리밍 사용
    def stream_response(self, prompt: str):
        """스트리밍으로 빠른 첫 토큰 반환"""
        with self.sync_client.messages.stream(
            model="claude-3-haiku-20240307",  # 빠른 모델
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}]
        ) as stream:
            for text in stream.text_stream:
                yield text

    # 전략 2: 비동기 배치 처리
    async def batch_process(self, prompts: List[str]) -> List[str]:
        """여러 요청 동시 처리"""
        tasks = [
            self.async_client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=512,
                messages=[{"role": "user", "content": p}]
            )
            for p in prompts
        ]

        responses = await asyncio.gather(*tasks)
        return [r.content[0].text for r in responses]

    # 전략 3: 프롬프트 캐싱
    def cached_request(self, system_prompt: str, user_query: str):
        """시스템 프롬프트 캐싱"""
        message = self.sync_client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            system=[
                {
                    "type": "text",
                    "text": system_prompt,
                    "cache_control": {"type": "ephemeral"}
                }
            ],
            messages=[{"role": "user", "content": user_query}]
        )

        return message.content[0].text

# 사용 예시
agent = OptimizedAgent("your-api-key")

# 1. 스트리밍
print("스트리밍 응답:")
for chunk in agent.stream_response("Python의 장점을 알려주세요"):
    print(chunk, end="", flush=True)

# 2. 배치 처리
prompts = ["질문 1", "질문 2", "질문 3"]
results = asyncio.run(agent.batch_process(prompts))
for i, result in enumerate(results, 1):
    print(f"\n응답 {i}: {result}")

# 3. 캐싱
system = "당신은 Python 전문가입니다..."  # 긴 시스템 프롬프트
response = agent.cached_request(system, "리스트 컴프리헨션이 뭔가요?")
print(response)
```

### 2. 메모리 최적화

```python
from collections import deque
from typing import List, Dict
import sys

class MemoryOptimizedAgent:
    """메모리 효율적인 에이전트"""

    def __init__(self, max_history: int = 10):
        # deque 사용으로 자동 크기 제한
        self.conversation = deque(maxlen=max_history)
        self.summary_threshold = 5

    def add_message(self, role: str, content: str):
        """메시지 추가"""
        self.conversation.append({"role": role, "content": content})

        # 일정 크기 이상이면 요약
        if len(self.conversation) >= self.summary_threshold * 2:
            self.compress_history()

    def compress_history(self):
        """오래된 대화 요약으로 압축"""
        # 오래된 절반을 요약
        old_messages = list(self.conversation)[:self.summary_threshold]

        # 간단한 요약 (실제로는 LLM 사용)
        summary = f"[요약] {len(old_messages)}개 메시지 압축됨"

        # 새로운 대화로 교체
        new_conversation = deque(maxlen=self.conversation.maxlen)
        new_conversation.append({"role": "system", "content": summary})

        for msg in list(self.conversation)[self.summary_threshold:]:
            new_conversation.append(msg)

        self.conversation = new_conversation

    def get_memory_usage(self) -> int:
        """메모리 사용량 (바이트)"""
        return sys.getsizeof(self.conversation)


# 사용 예시
agent = MemoryOptimizedAgent(max_history=10)

for i in range(20):
    agent.add_message("user", f"질문 {i}")
    agent.add_message("assistant", f"답변 {i}")

print(f"대화 길이: {len(agent.conversation)}")
print(f"메모리 사용량: {agent.get_memory_usage()} bytes")
```

---

## 멀티 에이전트 시스템

### 협업 에이전트 패턴

```python
import anthropic
from typing import List, Dict
from enum import Enum

class AgentRole(Enum):
    RESEARCHER = "researcher"
    ANALYZER = "analyzer"
    WRITER = "writer"
    REVIEWER = "reviewer"

class CollaborativeAgentSystem:
    """협업하는 멀티 에이전트 시스템"""

    def __init__(self, api_key: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.agents = {
            AgentRole.RESEARCHER: {
                "name": "연구 에이전트",
                "system_prompt": "당신은 정보 수집 전문가입니다. 주어진 주제에 대해 철저히 조사합니다.",
                "model": "claude-3-5-sonnet-20241022"
            },
            AgentRole.ANALYZER: {
                "name": "분석 에이전트",
                "system_prompt": "당신은 데이터 분석 전문가입니다. 수집된 정보를 분석하고 인사이트를 도출합니다.",
                "model": "claude-3-5-sonnet-20241022"
            },
            AgentRole.WRITER: {
                "name": "작성 에이전트",
                "system_prompt": "당신은 기술 문서 작성 전문가입니다. 분석 결과를 명확한 문서로 작성합니다.",
                "model": "claude-3-5-sonnet-20241022"
            },
            AgentRole.REVIEWER: {
                "name": "리뷰 에이전트",
                "system_prompt": "당신은 품질 관리 전문가입니다. 최종 결과물을 검토하고 개선점을 제안합니다.",
                "model": "claude-3-opus-20240229"  # 높은 품질의 리뷰
            }
        }

    def run_agent(self, role: AgentRole, task: str, context: str = "") -> str:
        """단일 에이전트 실행"""
        agent = self.agents[role]

        full_prompt = f"{context}\n\n{task}" if context else task

        response = self.client.messages.create(
            model=agent["model"],
            max_tokens=2048,
            system=agent["system_prompt"],
            messages=[{"role": "user", "content": full_prompt}]
        )

        return response.content[0].text

    def collaborative_workflow(self, topic: str) -> Dict[str, str]:
        """협업 워크플로우 실행"""
        results = {}

        print(f"주제: {topic}\n")
        print("="*60)

        # 1. 연구 단계
        print("\n[1/4] 연구 에이전트 실행 중...")
        research_task = f"'{topic}'에 대해 조사하고 핵심 정보를 정리해주세요."
        research_result = self.run_agent(AgentRole.RESEARCHER, research_task)
        results["research"] = research_result
        print(f"✓ 완료 (길이: {len(research_result)} 자)")

        # 2. 분석 단계
        print("\n[2/4] 분석 에이전트 실행 중...")
        analysis_task = f"다음 연구 결과를 분석하고 주요 인사이트를 도출해주세요:\n\n{research_result}"
        analysis_result = self.run_agent(AgentRole.ANALYZER, analysis_task)
        results["analysis"] = analysis_result
        print(f"✓ 완료 (길이: {len(analysis_result)} 자)")

        # 3. 작성 단계
        print("\n[3/4] 작성 에이전트 실행 중...")
        writing_task = f"""
다음 분석 결과를 바탕으로 기술 문서를 작성해주세요:

{analysis_result}

형식:
# {topic}

## 개요
## 주요 내용
## 결론
"""
        writing_result = self.run_agent(AgentRole.WRITER, writing_task)
        results["document"] = writing_result
        print(f"✓ 완료 (길이: {len(writing_result)} 자)")

        # 4. 리뷰 단계
        print("\n[4/4] 리뷰 에이전트 실행 중...")
        review_task = f"""
다음 문서를 리뷰하고 개선점을 제안해주세요:

{writing_result}

리뷰 항목:
- 정확성
- 완전성
- 명확성
- 구조
"""
        review_result = self.run_agent(AgentRole.REVIEWER, review_task)
        results["review"] = review_result
        print(f"✓ 완료 (길이: {len(review_result)} 자)")

        print("\n" + "="*60)
        print("모든 단계 완료!")

        return results


# 사용 예시
system = CollaborativeAgentSystem("your-api-key")
results = system.collaborative_workflow("Python 비동기 프로그래밍")

print("\n\n최종 문서:")
print("="*60)
print(results["document"])

print("\n\n리뷰:")
print("="*60)
print(results["review"])
```

---

*(계속...)*

## 실전 구현 예제

### 예제 1: 자동 코드 리뷰 에이전트

```python
import anthropic
from pathlib import Path
from typing import List, Dict

class CodeReviewAgent:
    """코드 리뷰 자동화 에이전트"""

    def __init__(self, api_key: str):
        self.client = anthropic.Anthropic(api_key=api_key)

    def review_file(self, file_path: Path) -> Dict:
        """단일 파일 리뷰"""
        code = file_path.read_text()
        language = file_path.suffix[1:]

        prompt = f"""
다음 {language} 코드를 리뷰해주세요:

```{language}
{code}
```

리뷰 결과를 다음 JSON 형식으로 제공하세요:
{{
  "score": 0-10,
  "issues": [
    {{"severity": "high/medium/low", "line": 숫자, "message": "설명"}}
  ],
  "suggestions": ["제안1", "제안2"]
}}
"""

        response = self.client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}]
        )

        return {
            "file": str(file_path),
            "review": response.content[0].text
        }
```

### 예제 2: 고객 지원 에이전트

```python
class CustomerSupportAgent:
    """고객 지원 자동화 에이전트"""

    def __init__(self, api_key: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.conversation_history = []

    def classify_intent(self, message: str) -> str:
        """고객 의도 분류"""
        prompt = f"""
다음 고객 메시지의 의도를 분류하세요:
"{message}"

분류 옵션:
- question: 질문
- complaint: 불만
- request: 요청
- feedback: 피드백

하나의 단어로만 답변하세요.
"""

        response = self.client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=10,
            messages=[{"role": "user", "content": prompt}]
        )

        return response.content[0].text.strip().lower()

    def generate_response(self, message: str, intent: str) -> str:
        """응답 생성"""
        system_prompts = {
            "question": "당신은 친절한 고객 지원 담당자입니다. 질문에 명확히 답변하세요.",
            "complaint": "당신은 공감능력이 뛰어난 고객 지원 담당자입니다. 불만을 경청하고 해결책을 제시하세요.",
            "request": "당신은 효율적인 고객 지원 담당자입니다. 요청사항을 처리하는 방법을 안내하세요.",
            "feedback": "당신은 감사하는 고객 지원 담당자입니다. 피드백에 감사하고 개선 의지를 보이세요."
        }

        response = self.client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=512,
            system=system_prompts.get(intent, system_prompts["question"]),
            messages=[{"role": "user", "content": message}]
        )

        return response.content[0].text
```

---

## 트러블슈팅

### 일반적인 문제 및 해결

| 문제 | 원인 | 해결책 |
|------|------|--------|
| 느린 응답 시간 | 큰 모델 사용 | Haiku 모델로 전환 또는 스트리밍 사용 |
| 높은 비용 | 불필요한 토큰 | 프롬프트 최적화, 캐싱 활용 |
| 불안정한 결과 | 높은 temperature | temperature 낮추기 (0.5-0.7) |
| 컨텍스트 초과 | 긴 대화 히스토리 | 요약 또는 슬라이딩 윈도우 |
| 도구 사용 실패 | 잘못된 파라미터 | 도구 스키마 명확히 정의 |

---

## 결론

효과적인 AI 에이전트 개발의 핵심:

1. **명확한 아키텍처**: 목적에 맞는 에이전트 패턴 선택
2. **성능 모니터링**: 지속적인 측정과 개선
3. **효율적인 리소스 관리**: 비용과 성능의 균형
4. **확장 가능한 설계**: 멀티 에이전트 협업 고려

## 추가 리소스

- [LangChain 문서](https://python.langchain.com/)
- [AutoGPT](https://github.com/Significant-Gravitas/AutoGPT)
- [Agent 프로토콜](https://agentprotocol.ai/)

---

**작성일**: 2024-11-15
**버전**: 1.0
**라이선스**: MIT
