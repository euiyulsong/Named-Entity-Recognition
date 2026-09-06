## Multi-turn / Multi-hop DST 실험 결과 요약

### 실험 설정

동일한 **hard subset 100개**에서 비교:

| 모델            | 학습 여부 | 평가 난이도                          |
| ------------- | ----: | ------------------------------- |
| LLM Zero-shot |     X | multi-domain + implicit + hop≥2 |
| LLM SFT       |     O | 동일 hard subset                  |

핵심 metric은 slot-value pair 기준 `Precision / Recall / F1`과 belief state 전체가 완전히 맞는 `Joint EM`.

---

## 결과

| Model             | Precision | Recall |         F1 | Joint EM |
| ----------------- | --------: | -----: | ---------: | -------: |
| **LLM Zero-shot** |    0.1641 | 0.0519 | **0.0788** |   0.0000 |
| **LLM SFT**       |    0.5541 | 0.6444 | **0.5959** |   0.0700 |

SFT 후 F1이:

```text
0.0788 → 0.5959
```

로 **+0.5171 absolute**, 약 **7.6배** 상승했다.

Recall 상승이 특히 크다.

```text
Zero-shot recall = 0.0519
SFT recall       = 0.6444
```

즉 zero-shot 모델은 과거 turn에 남아 있는 slot-value를 대부분 놓쳤지만, 학습 후에는 **누적 dialogue state를 훨씬 잘 복원**하게 됐다.

---

## Multi-turn / implicit case

hard set 자체가 전부 implicit example이라:

```text
has_implicit:
Zero-shot F1 = 0.0788
SFT F1       = 0.5959
```

이다.

즉 단순히 현재 문장에서 entity를 뽑는 NER 능력 향상이 아니라,

```text
previous dialogue
      ↓
carry-over / accumulated constraints
      ↓
current belief state
```

를 학습한 효과가 크다고 볼 수 있다.

---

## Explicit 정보가 일부 있는 경우

| Model     | Explicit F1 |
| --------- | ----------: |
| Zero-shot |      0.0764 |
| **SFT**   |  **0.6469** |

SFT 모델은 explicit entity가 있는 경우 가장 높은 성능을 보였다.

```text
Precision = 0.6165
Recall    = 0.6805
F1        = 0.6469
```

즉 현재 turn의 정보와 과거 turn 정보를 같이 활용하는 상황에서는 상당히 안정적이다.

---

## Hop distance별 결과

| Model     |   Hop 2 F1 |  Hop 3+ F1 |
| --------- | ---------: | ---------: |
| Zero-shot |     0.0408 |     0.0933 |
| **SFT**   | **0.5878** | **0.5993** |

SFT 후:

```text
Hop2 : 0.0408 → 0.5878
Hop3+: 0.0933 → 0.5993
```

로 크게 상승했다.

특히 중요한 점은 SFT 모델이:

```text
Hop2  = 0.5878
Hop3+ = 0.5993
```

으로 **context가 더 멀어져도 거의 성능이 떨어지지 않았다는 것**이다.

현재 실험 범위에서는:

> multi-hop distance 자체보다 task-specific DST 학습 여부가 훨씬 큰 영향을 준다.

라고 해석하는 게 적절하다.

다만 `max_hop` 기준 grouping이라 sample 난이도가 완전히 통제된 것은 아니므로, **“3-hop reasoning이 2-hop보다 더 쉽다”라고 해석하면 안 된다.**

---

## Multi-domain 결과

hard subset은 전부 multi-domain dialogue다.

```text
Zero-shot F1 = 0.0788
SFT F1       = 0.5959
```

따라서 SFT는 단순 single-domain pattern뿐 아니라 여러 domain의 slot을 동시에 유지하는 것도 상당히 학습했다.

예를 들어:

```text
hotel-area
hotel-pricerange
restaurant-food
train-destination
```

같은 서로 다른 domain의 constraint를 현재 state 안에서 함께 관리해야 하는 상황에서도 성능이 유지된다는 의미다.

---

## Joint EM은 여전히 낮음

SFT의 pair F1은:

```text
0.5959
```

인데 Joint EM은:

```text
0.0700
```

이다.

즉 개별 slot-value는 약 60% 수준으로 맞추지만, **belief state 전체를 하나도 빠짐없이 완벽하게 맞춘 turn은 7%**뿐이다.

DST에서는 꽤 자연스러운 차이다.

예를 들어 gold가 6개 slot이고 모델이 5개를 맞히면 pair F1은 높게 나오지만:

```text
Joint EM = 0
```

이기 때문이다.

따라서 현재 병목은:

> “대화를 전혀 이해하지 못함”

보다는

> **“전체 state를 완전하게 복원하는 completeness”**

쪽에 더 가깝다.

---

## 핵심 결론

이번 결과는 꽤 명확하다.

> **0.8B급 LLM은 zero-shot으로 어려운 multi-turn/multi-hop DST를 거의 못 풀지만, task-specific SFT를 하면 성능이 크게 향상된다.**

특히:

```text
Overall F1 : 0.0788 → 0.5959
Hop2 F1    : 0.0408 → 0.5878
Hop3+ F1   : 0.0933 → 0.5993
```

이라는 결과 때문에,

**multi-turn / multi-hop 능력이 pretraining만으로 자동으로 충분히 생기는 것은 아니고, 작은 규모라도 명시적인 DST 학습이 매우 중요하다**

는 결론을 낼 수 있다.

한 줄로 요약하면:

> **Task-specific SFT dominates zero-shot generalization on hard multi-turn DST: +51.7 F1 points overall, with similarly large gains at hop 2 and hop 3+.**
