#!/usr/bin/env python3
"""
예제 1: 기본 채팅
OpenAI API를 사용한 가장 기본적인 대화 예제
"""

import os
from openai import OpenAI

def main():
    # API 키 설정
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
        return

    print("="*60)
    print("예제 1: 기본 채팅")
    print("="*60)

    # 클라이언트 초기화
    client = OpenAI(api_key=api_key)

    # 테스트 메시지
    test_messages = [
        "안녕하세요! 당신은 누구인가요?",
        "Python의 주요 장점 3가지를 알려주세요.",
        "간단한 Hello World 코드를 작성해주세요."
    ]

    for i, message in enumerate(test_messages, 1):
        print(f"\n[질문 {i}] {message}")
        print("-" * 60)

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",  # 비용 효율적인 모델
                messages=[
                    {"role": "system", "content": "당신은 도움이 되는 AI 어시스턴트입니다."},
                    {"role": "user", "content": message}
                ],
                max_tokens=500,
                temperature=0.7
            )

            # 응답 출력
            answer = response.choices[0].message.content
            print(f"[응답] {answer}")

            # 토큰 사용량
            usage = response.usage
            print(f"\n📊 토큰 사용: 입력={usage.prompt_tokens}, "
                  f"출력={usage.completion_tokens}, "
                  f"합계={usage.total_tokens}")

        except Exception as e:
            print(f"❌ 오류 발생: {e}")

    print("\n" + "="*60)
    print("✅ 테스트 완료!")
    print("="*60)

if __name__ == "__main__":
    main()
