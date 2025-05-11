# Large Scale Meta Optimization
데이터 기반 방법은 일반적으로 확장성(scalability)에 유리한데, 메타러닝도 대규모 문제에 잘 작동할까?

## Intro
- 대규모 메타 최적화를 왜 고려해야 하는가?
- 이 기법이 사용될 수 있는 실제 응용 사례
- 접근법
  - Truncated backpropagation (절단된 역전파)
  - Gradient-free optimization (그레디언트를 쓰지 않는 최적화, 예: 진화 알고리즘)
- Direct Backpropagation
  - 메타러닝(meta-learning)의 일반적인 동작 방식
  - 유연성이 높으나 메모리 사용량이 크고, 계산 그래프가 깊을수록 비용 증가
    - 메타파라미터는 단순히 초기값만이 아닌 학습 함수 전체의 구성 요소가 되기에 훨씬 복잡
    - 계산 그래프가 커지는 상황
      1. 큰 네트워크
      2. 많은 gradient step(MAML 계열)
      3. 2차 최적화를 포함
  - 모델이 너무 크거나 학습 단계가 너무 많을 때는 직접 역전파가 불가능해지므로, 다른 대안(예: truncated BP, gradient-free methods)이 필요

## 대규모 메타 최적화를 위한 방법
검증 성능을 최대화하기 위해 어떤 것을 최적화할 것인가?라는 관점에서 분류

1. Hyperparameter Optimization
   - 검증 성능을 높이기 위한 **하이퍼파라미터**를 최적화
2. Hyperparameter Optimization (Synthetic Data 기반)
   - **학습 데이터**를 생성하는 하이퍼파라미터를 최적화
3. Neural Architecture Search (NAS)
   - 검증 성능을 높이기 위해 네트워크 **아키텍처**를 최적화
4. Optimizer Learning (옵티마이저 학습)
   - 검증 성능을 높이기 위한 **옵티마이저** 자체를 학습

### Truncated backpropagation
- 긴 시퀀스 연산 또는 계산 그래프를 메모리 효율적으로 처리하기 위해 사용하는 대표적인 전략 중 하나
- 긴 계산 그래프를 작은 조각으로 잘라서,  각 조각마다 따로 역전파(backpropagation)를 수행하는 방식
- 장점
  - 간단하고 효율적
- 단점
  - 편향된 추정값
  - 장기의존성을 학습할 수 없음
  - 길이의 선택은 항상 trade-off
    - 짧으면 빠르지만 학습 성능 저하, 고차 정보 반영 못함, 이전 task 경험 잊음
    - 길면 정확하지만 메모리 폭증

### Gradient-free Optimization
그래디언트를 직접 계산하지 않고도 최적화를 수행하는 방법이 필요

- 대안 기법으로 Evolution Strategies
  - 여러 파라미터 세트를 무작위로 만들어 평가 → 더 잘 수행한 쪽을 선택해 다음 세대를 생성
  - 많은 시도가 필요할 수 있으며 수렴 속도가 느려서 효율이 낮을 수 있음. noise에 민감할 수도 있음.
- 프로세스
  1. 파라미터 초기화
  2. 표본 샘플링(다양한 파라미터 조합)
  3. 평가 및 선택(최상위 성능만 선택)
  4. 업데이트(세대 갱신)
- 초기 파라미터 자체를 ES로 최적화 하려면? 고차원 파라미터 공간에서는 좋은 초기값 찾기가 어려워서 수렴 속도 느려짐.
- 장점
  - 메모리 고정(gradient free)
  - 병렬화 쉬움
  - 내부 연산이 미분 불가능해도 괜찮다.
- 단점
  - 고차원 파라미터 공간에서 성능 저하
  - 복잡한 구조에서 수렴 어려움

### 다른 대안들
1. Implicit Differentiation(암시적 미분)
   - 내부 최적화 결과(최종 값)만 가지고 meta-gradient를 계산
   - 중간 학습 과정은 저장하지 않음 → 메모리 절약
2. Forward-mode Differentiation(정방향 미분)
   - 체인 룰을 정방향으로 적용
   - 일반적인 딥러닝 환경에서는 역방향이 더 효율적인 경우가 많음