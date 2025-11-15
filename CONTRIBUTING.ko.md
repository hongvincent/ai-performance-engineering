# AI Systems Performance Engineering에 기여하기

AI Systems Performance Engineering 레포지토리에 기여하는 데 관심을 가져주셔서 감사합니다! 이 가이드는 코드, 문서, 예제 및 개선 사항 기여를 시작하는 데 도움을 드립니다.

## 기여 방법

우리는 다양한 형태의 커뮤니티 기여를 환영합니다:

- **코드 예제**: 새로운 CUDA 커널, PyTorch 최적화, 성능 스크립트
- **문서화**: README 파일, 코드 주석, 튜토리얼 개선
- **성능 최적화**: 더 나은 알고리즘, 메모리 최적화, 프로파일링 도구
- **버그 수정**: 기존 코드의 이슈, 호환성 문제
- **아키텍처 지원**: Blackwell 워크플로우 확장 또는 새로운 GPU 제품군을 위한 도구 추가
- **테스트**: 단위 테스트, 성능 벤치마크, 검증 스크립트

## 시작하기

### 사전 요구사항

- CUDA를 지원하는 NVIDIA GPU
- Python 3.8+
- CUDA를 지원하는 PyTorch
- Git

### 개발 환경 설정

```bash
# 레포지토리를 포크하고 클론
git clone https://github.com/your-username/ai-performance-engineering.git
cd ai-performance-engineering

# 기여를 위한 새 브랜치 생성
git checkout -b feature/your-feature-name

# 개발 의존성 설치
pip install -r code/ch1/requirements.txt
```

## 기여 가이드라인

### 코드 스타일

- **Python**: PEP 8 스타일 가이드라인 준수
- **CUDA**: 일관된 명명 규칙 및 적절한 에러 처리 사용
- **Shell 스크립트**: 적절한 에러 처리를 사용하는 bash (`set -e`)
- **주석**: 복잡한 로직에 명확하고 설명적인 주석 추가

### 파일 조직

- **새로운 예제**: 적절한 챕터 디렉토리에 배치 (`code/chX/`)
- **도구**: `tools/` 디렉토리에 추가
- **스크립트**: `scripts/` 또는 관련 챕터 디렉토리에 추가
- **문서화**: 관련 README 파일 업데이트

### 아키텍처 지원

메인 브랜치는 **Blackwell B200/B300 (SM100)**을 독점적으로 타겟팅합니다. 새로운 예제는 기본적으로 `ARCH ?= sm_100`을 사용하고 CUDA 12.9 툴체인을 상속해야 합니다. 다른 GPU에 대한 지원을 프로토타입하는 경우, 명확하게 문서화된 플래그 뒤에 두거나 별도의 브랜치로 제출하세요.

## 개발 워크플로우

### 1. 기여 유형 선택

#### **코드 예제**
- 새로운 CUDA 커널 또는 PyTorch 최적화 생성
- 성능 프로파일링 스크립트 추가
- 새로운 알고리즘 또는 기법 구현

#### **문서화**
- 더 나은 설명으로 README 파일 개선
- 코드 주석 및 docstring 추가
- 튜토리얼 또는 가이드 생성

#### **성능 최적화**
- 더 나은 성능을 위해 기존 코드 최적화
- 새로운 프로파일링 도구 추가
- 메모리 사용량 또는 연산 효율성 개선

### 2. 개발 프로세스

```bash
# 변경 사항 작성
# 코드를 철저히 테스트

# 테스트 실행 (해당하는 경우)
python -m pytest tests/

# 코드 스타일 확인
black code/
flake8 code/
```

### 3. 변경 사항 테스트

#### **성능 테스트**
```bash
# 성능 벤치마크 실행
./code/build_all.sh

# 변경 사항 프로파일링
python scripts/profile_harness.py --profile nsys --profile pytorch --examples ch6_add_parallel --output-root profiles/test_run

# 베이스라인과 비교
python tools/comprehensive_profiling.py
```

#### **호환성 테스트**
- Blackwell B200/B300 하드웨어에서 실행 확인
- PyTorch 2.9 nightly/cu129 환경 확인
- CUDA 12.9 툴킷 호환성 보장

### 4. 기여 제출

```bash
# 변경 사항 추가
git add .

# 설명적인 메시지로 커밋
git commit -m "Add new CUDA kernel for memory optimization

- Implements coalesced memory access pattern
- Targets NVIDIA Blackwell B200/B300
- Includes performance benchmarks
- Adds comprehensive documentation"

# 포크에 푸시
git push origin feature/your-feature-name
```

## Pull Request 가이드라인

### 제출 전

- [ ] Blackwell 하드웨어(또는 시뮬레이터)에서 **철저히 테스트**
- [ ] 필요한 경우 **문서 업데이트**
- [ ] 복잡한 코드에 **주석 추가**
- [ ] 최적화에 대한 **성능 벤치마크 포함**
- [ ] **명명 규칙** 및 코드 스타일 준수
- [ ] **관련 README 파일 업데이트**

### Pull Request 템플릿

```markdown
## 설명
변경 사항에 대한 간략한 설명

## 변경 유형
- [ ] 새로운 기능 (코드 예제, 최적화)
- [ ] 버그 수정
- [ ] 문서 업데이트
- [ ] 성능 개선
- [ ] Blackwell 워크플로우 개선

## 테스트
- [ ] Blackwell B200/B300 (sm_100)에서 테스트됨
- [ ] 성능 벤치마크 포함됨
- [ ] 문서 업데이트됨

## 성능 영향
- **이전**: [베이스라인 메트릭]
- **이후**: [개선된 메트릭]
- **개선**: [백분율/설명]

## 추가 참고사항
추가 컨텍스트 또는 고려사항
```

## 아키텍처 가이드라인

### 새로운 GPU 지원 추가

새로운 GPU 아키텍처에 대한 지원을 추가할 때:

1. **아키텍처 감지 스크립트 업데이트**
2. **새로운 아키텍처 상수 추가**
3. **타겟 하드웨어에서 테스트**
4. **문서 업데이트**

### Blackwell 이후 확장

추가 아키텍처를 실험하는 경우, 변경 사항을 명확히 문서화하고 기본 Blackwell 워크플로우가 성능 저하되지 않도록 하세요. `main`을 간결하게 유지하기 위해 아키텍처별 차이점에 대해 별도의 브랜치를 유지하는 것을 고려하세요.

## 성능 기여 가이드라인

### 벤치마킹 표준

- **베이스라인**: 항상 베이스라인 성능 포함
- **다중 실행**: 벤치마크를 여러 번 실행
- **하드웨어 사양**: 테스트 하드웨어 문서화
- **환경**: CUDA/PyTorch 버전 명시

### 벤치마크 형식 예제

```python
# 성능 벤치마크 예제
import time
import torch

def benchmark_kernel():
    # 설정
    device = torch.device('cuda')
    size = 1024 * 1024

    # 워밍업
    for _ in range(10):
        # 여기에 커널 코드
        pass

    # 벤치마크
    start = time.time()
    for _ in range(100):
        # 여기에 커널 코드
        pass
    end = time.time()

    # 보고
    avg_time = (end - start) / 100
    throughput = size / avg_time
    print(f"Average time: {avg_time:.6f}s")
    print(f"Throughput: {throughput:.2f} ops/s")
```

## 버그 리포트

### 이슈 보고

버그를 보고할 때 다음을 포함해 주세요:

- **하드웨어**: GPU 모델, 드라이버 버전
- **소프트웨어**: CUDA 버전, PyTorch 버전
- **단계**: 명확한 재현 단계
- **예상 vs 실제**: 예상한 것 vs 실제 발생한 것
- **로그**: 에러 메시지 및 로그

### 이슈 템플릿

```markdown
## 버그 설명
이슈에 대한 명확한 설명

## 재현 단계
1. 단계 1
2. 단계 2
3. 단계 3

## 예상 동작
예상했던 동작

## 실제 동작
실제 발생한 동작

## 환경
- GPU: [모델]
- CUDA: [버전]
- PyTorch: [버전]
- OS: [버전]

## 추가 컨텍스트
기타 관련 정보
```

## 문서화 기여

### README 업데이트

문서를 업데이트할 때:

- **명확성**: 설명을 명확하고 간결하게 작성
- **예제**: 실용적인 코드 예제 포함
- **링크**: 관련 링크 및 참조 추가
- **구조**: 일관된 포맷팅 유지

### 코드 주석

- **목적**: 코드가 무엇을 하는지 설명
- **매개변수**: 함수 매개변수 문서화
- **반환값**: 반환값 문서화
- **복잡성**: 복잡한 알고리즘 설명

## 기여 아이디어

### 우선순위가 높은 영역

- **새로운 CUDA 커널**: 최적화된 구현
- **PyTorch 최적화**: 프레임워크별 개선
- **프로파일링 도구**: 더 나은 성능 분석
- **아키텍처 지원**: 새로운 GPU 호환성
- **문서화**: 튜토리얼 및 가이드

### 기여 예제

- **메모리 최적화**: 새로운 메모리 액세스 패턴
- **커널 퓨전**: 여러 연산 결합
- **Tensor Core 사용**: 최적화된 행렬 연산
- **Stream 관리**: 더 나은 비동기 실행
- **분산 학습**: 멀티 GPU 최적화

## 도움 받기

### 커뮤니티 리소스

- **Issues**: 질문은 GitHub Issues를 사용
- **Discussions**: 아이디어에 대한 토론 시작
- **Meetups**: 월간 모임 참여
- **YouTube**: 비디오 튜토리얼 확인

### 연락처

- **GitHub Issues**: 버그 및 기능 요청
- **Discussions**: 질문 및 아이디어
- **Email**: 비공개 또는 민감한 사항

## 라이선스

이 프로젝트에 기여함으로써, 귀하의 기여가 프로젝트와 동일한 라이선스(MIT License) 하에 라이선스될 것에 동의하는 것입니다.

## 인정

기여자는 다음에서 인정받게 됩니다:

- **README.md**: 중요한 기여에 대해
- **릴리스 노트**: 각 릴리스마다
- **문서화**: 관련 섹션에서
- **커뮤니티**: 모임 및 프레젠테이션에서

---

AI Performance Engineering 커뮤니티에 기여해 주셔서 감사합니다.
