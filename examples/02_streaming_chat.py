#!/usr/bin/env python3
"""
예제 2: 스트리밍 채팅
실시간으로 응답을 받는 스트리밍 예제
"""

import os
from openai import OpenAI

def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
        return

    print("="*60)
    print("예제 2: 스트리밍 채팅")
    print("="*60)

    client = OpenAI(api_key=api_key)

    prompt = "Python으로 간단한 웹 서버를 만드는 방법을 단계별로 설명해주세요."

    print(f"\n[질문] {prompt}")
    print("-" * 60)
    print("[응답] ", end="", flush=True)

    try:
        stream = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            max_tokens=1000
        )

        full_response = ""
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content
                print(content, end="", flush=True)
                full_response += content

        print("\n" + "-" * 60)
        print(f"📏 응답 길이: {len(full_response)} 문자")
        print("✅ 스트리밍 완료!")

    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")

if __name__ == "__main__":
    main()
