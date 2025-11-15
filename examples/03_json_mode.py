#!/usr/bin/env python3
"""
예제 3: JSON 모드 (구조화된 출력)
JSON 형식으로 응답을 받는 예제
"""

import os
import json
from openai import OpenAI

def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
        return

    print("="*60)
    print("예제 3: JSON 모드 (구조화된 출력)")
    print("="*60)

    client = OpenAI(api_key=api_key)

    # 테스트 리뷰
    reviews = [
        "이 제품 정말 훌륭해요! 강력 추천합니다.",
        "배송이 너무 느렸어요. 실망스럽네요.",
        "가격 대비 괜찮은 것 같아요."
    ]

    for i, review in enumerate(reviews, 1):
        print(f"\n[리뷰 {i}] {review}")
        print("-" * 60)

        prompt = f"""
다음 리뷰를 분석하고 JSON 형식으로 답변하세요:

리뷰: "{review}"

JSON 형식:
{{
  "sentiment": "positive/negative/neutral",
  "confidence": 0.0-1.0,
  "keywords": ["키워드1", "키워드2"],
  "category": "제품/배송/가격/기타"
}}
"""

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                max_tokens=200
            )

            result = json.loads(response.choices[0].message.content)
            print("[분석 결과]")
            print(json.dumps(result, ensure_ascii=False, indent=2))

        except Exception as e:
            print(f"❌ 오류 발생: {e}")

    print("\n" + "="*60)
    print("✅ 테스트 완료!")
    print("="*60)

if __name__ == "__main__":
    main()
