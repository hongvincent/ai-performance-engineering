#!/usr/bin/env python3
"""
예제 4: 비동기 배치 처리
여러 요청을 동시에 효율적으로 처리하는 예제
"""

import os
import asyncio
from openai import AsyncOpenAI
from typing import List
import time

async def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
        return

    print("="*60)
    print("예제 4: 비동기 배치 처리")
    print("="*60)

    # 비동기 클라이언트 생성
    client = AsyncOpenAI(api_key=api_key)

    # 처리할 프롬프트 리스트
    prompts = [
        "Python의 주요 장점 3가지를 간단히 설명해주세요.",
        "JavaScript의 주요 특징을 알려주세요.",
        "TypeScript가 JavaScript와 다른 점은?",
        "React의 주요 개념을 설명해주세요.",
        "Node.js는 무엇인가요?"
    ]

    print(f"\n총 {len(prompts)}개의 요청을 비동기로 처리합니다.\n")

    # 시작 시간 기록
    start_time = time.time()

    # 비동기 작업 생성
    async def process_single(prompt: str, index: int):
        """단일 프롬프트 처리"""
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200
        )
        return {
            "index": index,
            "prompt": prompt,
            "response": response.choices[0].message.content,
            "tokens": response.usage.total_tokens
        }

    # 모든 작업을 동시에 실행
    tasks = [process_single(prompt, i) for i, prompt in enumerate(prompts, 1)]
    results = await asyncio.gather(*tasks)

    # 종료 시간 기록
    end_time = time.time()
    total_time = end_time - start_time

    # 결과 출력
    for result in results:
        print(f"[요청 {result['index']}] {result['prompt'][:40]}...")
        print(f"[응답] {result['response'][:100]}...")
        print(f"📊 토큰: {result['tokens']}")
        print("-" * 60)

    # 통계 출력
    total_tokens = sum(r['tokens'] for r in results)
    print(f"\n⏱️  총 처리 시간: {total_time:.2f}초")
    print(f"📊 총 토큰 사용: {total_tokens}")
    print(f"⚡ 평균 처리 시간: {total_time/len(prompts):.2f}초/요청")
    print(f"🚀 처리량: {len(prompts)/total_time:.2f} 요청/초")

    print("\n" + "="*60)
    print("✅ 비동기 배치 처리 완료!")
    print("="*60)

if __name__ == "__main__":
    asyncio.run(main())
