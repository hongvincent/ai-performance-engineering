#!/usr/bin/env python3
"""
예제 5: Function Calling (도구 사용)
AI가 외부 함수를 호출하여 작업을 수행하는 예제
"""

import os
import json
from openai import OpenAI
from datetime import datetime

def get_current_weather(location: str, unit: str = "celsius") -> dict:
    """현재 날씨를 가져옵니다 (시뮬레이션)"""
    # 실제로는 날씨 API를 호출
    weather_data = {
        "서울": {"temp": 15, "condition": "맑음"},
        "부산": {"temp": 18, "condition": "흐림"},
        "제주": {"temp": 20, "condition": "비"}
    }

    data = weather_data.get(location, {"temp": 22, "condition": "알 수 없음"})

    if unit == "fahrenheit":
        data["temp"] = data["temp"] * 9/5 + 32

    return {
        "location": location,
        "temperature": data["temp"],
        "unit": unit,
        "condition": data["condition"],
        "timestamp": datetime.now().isoformat()
    }

def calculate(expression: str) -> dict:
    """수학 계산을 수행합니다"""
    try:
        result = eval(expression)
        return {
            "expression": expression,
            "result": result,
            "success": True
        }
    except Exception as e:
        return {
            "expression": expression,
            "error": str(e),
            "success": False
        }

def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
        return

    print("="*60)
    print("예제 5: Function Calling (도구 사용)")
    print("="*60)

    client = OpenAI(api_key=api_key)

    # 사용 가능한 함수 정의
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "특정 위치의 현재 날씨를 가져옵니다",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "도시 이름, 예: 서울, 부산"
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "온도 단위"
                        }
                    },
                    "required": ["location"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "calculate",
                "description": "수학 계산을 수행합니다",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "계산할 수식, 예: '2 + 2', '10 * 5'"
                        }
                    },
                    "required": ["expression"]
                }
            }
        }
    ]

    # 테스트 질문들
    questions = [
        "서울의 현재 날씨는 어때?",
        "25 곱하기 4는 얼마야?",
        "부산 날씨를 화씨로 알려줘"
    ]

    for i, question in enumerate(questions, 1):
        print(f"\n[질문 {i}] {question}")
        print("-" * 60)

        # 첫 번째 API 호출
        messages = [{"role": "user", "content": question}]

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )

        response_message = response.choices[0].message

        # 도구 호출 확인
        if response_message.tool_calls:
            # 메시지 히스토리에 추가
            messages.append(response_message)

            # 각 도구 호출 처리
            for tool_call in response_message.tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)

                print(f"🔧 도구 호출: {function_name}")
                print(f"📥 인자: {function_args}")

                # 함수 실행
                if function_name == "get_current_weather":
                    function_response = get_current_weather(**function_args)
                elif function_name == "calculate":
                    function_response = calculate(**function_args)
                else:
                    function_response = {"error": "Unknown function"}

                print(f"📤 결과: {function_response}")

                # 함수 결과를 메시지에 추가
                messages.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": json.dumps(function_response, ensure_ascii=False)
                })

            # 두 번째 API 호출 (함수 결과 포함)
            second_response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages
            )

            final_answer = second_response.choices[0].message.content
            print(f"\n💬 최종 답변: {final_answer}")
        else:
            # 도구 호출 없이 직접 답변
            print(f"💬 답변: {response_message.content}")

    print("\n" + "="*60)
    print("✅ Function Calling 예제 완료!")
    print("="*60)

if __name__ == "__main__":
    main()
