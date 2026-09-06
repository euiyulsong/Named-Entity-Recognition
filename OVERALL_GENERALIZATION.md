## Multi-Entity 1,000개 학습 결과 분석

### 실험 요약

이번 실험에서는 CoNLL-2003 train에서 **multi-entity 문장 1,000개만 사용해 학습**한 뒤, 이전의 **single-entity-only 학습 결과**와 비교했다.

평가 split은 동일하게 유지했다.

* `single_entity`: entity 1개
* `exactly_2_entities`: entity 정확히 2개
* `3plus_entities`: entity 3개 이상
* `multi_entity`: entity 2개 이상 전체
* `multi_entity_same_type`: entity는 여러 개지만 type은 1종류
* `multi_type`: entity가 여러 개이고 type도 2종류 이상

비교 모델은:

* BERT-base token classification
* Qwen generative NER

이다.

---

## 전체 결과

| Model / Train              |     Single |  Exactly 2 |         3+ | Multi overall | Same-type Multi | Multi-type |
| -------------------------- | ---------: | ---------: | ---------: | ------------: | --------------: | ---------: |
| BERT / single-entity train | **0.8775** |     0.7413 |     0.7825 |        0.7659 |      **0.8732** |     0.7356 |
| BERT / multi-1000 train    |     0.6767 |     0.8338 |     0.8064 |        0.8174 |          0.8622 |     0.8050 |
| Qwen / single-entity train | **0.8688** |     0.5606 |     0.3213 |        0.4262 |          0.5246 |     0.3964 |
| Qwen / multi-1000 train    |     0.7046 | **0.8576** | **0.8324** |    **0.8433** |          0.8365 | **0.8427** |

가장 큰 패턴은 명확하다.

> **학습 데이터의 entity cardinality가 테스트 성능 분포를 매우 강하게 결정한다.**

single-entity만 학습하면 single test에 강하고, multi-entity를 학습하면 multi test에 강해진다.

특히 Qwen에서 이 현상이 매우 크게 나타났다.

---

## 1. Qwen은 multi-entity 1,000개 학습만으로 multi 성능이 급상승

Qwen 결과 변화가 가장 극적이다.

| Split           | Single-train | Multi-1000 |          변화 |
| --------------- | -----------: | ---------: | ----------: |
| single          |       0.8688 |     0.7046 |     -0.1642 |
| exactly 2       |       0.5606 | **0.8576** | **+0.2970** |
| 3+              |       0.3213 | **0.8324** | **+0.5111** |
| multi overall   |       0.4262 | **0.8433** | **+0.4171** |
| same-type multi |       0.5246 | **0.8365** | **+0.3119** |
| multi-type      |       0.3964 | **0.8427** | **+0.4463** |

특히 `3plus_entities`는

$$
0.3213 \rightarrow 0.8324
$$

로 **+0.5111 absolute F1** 상승했다.

`multi_type`도:

$$
0.3964 \rightarrow 0.8427
$$

로 크게 상승했다.

즉 이전 Qwen single-only SFT의 낮은 multi performance는 단순히 모델이 multi-entity reasoning을 못해서라기보다, **학습 중 multi-output 구조를 전혀 보지 못한 영향이 매우 컸다**고 해석하는 게 자연스럽다.

---

## 2. 이전 Qwen 실패는 output-cardinality bias 영향이 컸다고 볼 수 있음

Single-entity SFT에서는 모든 target이 사실상 이런 구조였다.

```json
[
  {"text": "Google", "type": "ORG"}
]
```

즉 항상 **list length = 1**이었다.

반면 multi-entity 학습에서는 모델이 이런 target도 학습하게 된다.

```json
[
  {"text": "John", "type": "PER"},
  {"text": "Google", "type": "ORG"},
  {"text": "London", "type": "LOC"}
]
```

그러자 multi-entity F1이:

$$
0.4262 \rightarrow 0.8433
$$

로 거의 두 배가 되었다.

따라서 이전 실험에서 관찰한 큰 generalization gap은 단순한 semantic compositional failure뿐 아니라,

$$
\text{output cardinality mismatch}
$$

의 영향이 상당히 컸다고 볼 수 있다.

즉 Qwen은 entity concept 자체보다도:

> **“몇 개를 출력해야 하는가”라는 output structure를 training distribution에서 강하게 학습하는 경향**

을 보인다.

---

## 3. Qwen multi-1000은 multi split 전반에서 거의 균일하게 강함

Multi-1000 Qwen의 결과는:

```text
exactly 2 entities       0.8576
3+ entities              0.8324
multi entity             0.8433
same-type multi          0.8365
multi-type               0.8427
```

로 거의 모두 **0.83~0.86** 범위다.

이건 중요한 결과다.

Single-only train 때는:

```text
same-type multi    0.5246
multi-type         0.3964
```

로 distinct type composition에서 추가 하락이 컸다.

하지만 multi-entity 학습 후에는:

```text
same-type multi    0.8365
multi-type         0.8427
```

로 오히려 거의 차이가 없다.

즉 **multi-output 경험만 주어지면 서로 다른 entity type의 composition 자체는 Qwen에게 큰 문제가 아니었다**고 볼 수 있다.

이전의:

> multi-type composition 자체가 어렵다

라는 해석은 Qwen에 대해서는 수정해야 한다.

더 정확하게는:

> **single-only SFT가 multi-output 구조를 학습시키지 않았기 때문에 multi-type에서도 크게 실패했던 것**

에 가깝다.

---

## 4. 대신 single-entity 성능은 떨어짐

Qwen:

$$
0.8688 \rightarrow 0.7046
$$

BERT:

$$
0.8775 \rightarrow 0.6767
$$

로 둘 다 single test 성능은 크게 감소했다.

즉 반대 방향의 distribution shift도 존재한다.

```text
single-only train
    → single 강함 / multi 약함

multi-only train
    → multi 강함 / single 약함
```

이 결과는 모델이 단순히 더 일반적인 NER 능력을 얻는다기보다 **training cardinality distribution에 맞춰 specialization**된다는 것을 보여준다.

---

## 5. BERT도 multi 학습 효과가 있지만 Qwen보다 훨씬 작음

BERT 변화는:

```text
single          -0.2008
exactly 2       +0.0925
3+              +0.0239
multi overall   +0.0515
same-type       -0.0110
multi-type      +0.0694
```

이다.

즉 multi-entity 1,000개 학습으로 multi 성능은 개선되지만 그 크기는 비교적 작다.

특히:

$$
F1_{multi}: 0.7659 \rightarrow 0.8174
$$

약 +0.052 정도다.

반면 Qwen은:

$$
0.4262 \rightarrow 0.8433
$$

약 +0.417이다.

이는 BERT가 single-only training에서도 이미 multi-entity에 상당히 잘 generalize하고 있었기 때문이다.

---

## 6. BERT와 Qwen의 inductive bias 차이가 매우 선명해짐

BERT는 token classification이다.

각 token마다:

$$
p(y_i \mid x_1,\ldots,x_n)
=
\text{softmax}(Wh_i+b)
$$

를 계산한다.

따라서 training sentence에 entity가 1개밖에 없더라도, test에서 entity가 여러 개 등장하면 같은 token-level classifier를 여러 위치에 적용할 수 있다.

그래서 single-only BERT도:

```text
single             0.8775
same-type multi    0.8732
multi-type         0.7356
```

처럼 상대적으로 잘 generalize했다.

반면 Qwen은:

$$
p(\text{entire output sequence}\mid x)
$$

를 학습한다.

따라서 entity 개수가 바뀌면 단순히 token classification을 여러 번 적용하는 게 아니라 **출력 sequence의 구조와 길이 자체를 바꿔야 한다.**

이 때문에 Qwen은 training cardinality에 훨씬 민감하게 반응한 것으로 보인다.

---

## 7. Multi-1000에서는 Qwen이 오히려 BERT보다 높음

흥미롭게도 multi-entity 학습 후에는 Qwen이 대부분의 multi split에서 BERT를 앞선다.

| Split           | BERT Multi-1000 | Qwen Multi-1000 |
| --------------- | --------------: | --------------: |
| exactly 2       |          0.8338 |      **0.8576** |
| 3+              |          0.8064 |      **0.8324** |
| multi overall   |          0.8174 |      **0.8433** |
| same-type multi |      **0.8622** |          0.8365 |
| multi-type      |          0.8050 |      **0.8427** |

특히 `multi_type`:

$$
0.8050 \rightarrow 0.8427
$$

로 Qwen이 약 +0.038 높다.

즉 Qwen이 inherently NER generalization에 약한 것은 아니다.

오히려:

> **적절한 multi-entity supervision만 주어지면 generative LLM도 매우 높은 multi-entity performance를 달성한다.**

라고 보는 게 맞다.

---

## 8. Qwen의 precision/recall도 균형이 좋아짐

Multi-1000 Qwen의 주요 결과:

```text
multi_entity
Precision = 0.8509
Recall    = 0.8359
F1        = 0.8433
```

`multi_type`도:

```text
Precision = 0.8517
Recall    = 0.8338
F1        = 0.8427
```

이다.

이전 zero-shot이나 single-only SFT에서 나타났던 심한 over-extraction / under-extraction 문제가 크게 줄고, precision과 recall이 거의 균형을 이룬다.

즉 multi-output examples를 1,000개만 줘도 **output calibration 자체가 상당히 좋아진 것**으로 보인다.

---

## 9. Sentence Exact Match도 크게 회복됨

Qwen Multi-1000의 exact match:

```text
single             0.6237
exactly 2          0.7445
3+                 0.5096
multi overall      0.6440
same-type multi    0.7220
multi-type         0.6142
```

이전 single-only SFT에서는 multi split들의 exact match가 거의 0이었던 것과 완전히 다르다.

즉 multi-1000 학습 후에는 단순히 entity 일부를 맞추는 수준이 아니라, **문장 전체 entity set을 완벽하게 생성하는 능력도 실제로 생겼다.**

이 결과 역시 output-cardinality hypothesis를 강하게 지지한다.

---

## 10. 가장 중요한 결론

이번 실험은 이전 결과의 해석을 상당히 명확하게 바꿔준다.

처음 single-only 실험만 보면:

> Qwen은 single entity skill을 multi entity로 composition하지 못한다.

고 볼 수 있었다.

하지만 multi-entity 1,000개만 학습했더니:

$$
F1_{multi}=0.8433
$$

까지 즉시 회복되었다.

따라서 더 정확한 결론은:

> **Generative NER 모델은 training-time output cardinality에 매우 민감하며, single-output supervision만으로는 multi-output generalization이 약하다. 하지만 소량의 multi-entity supervision만 주어져도 그 gap은 대부분 해소된다.**

반면 BERT는 token-wise prediction 구조 덕분에 single-only supervision에서도 multi-entity generalization이 이미 상당히 강하다.

---

## 최종 요약

### BERT

```text
Single-only training
    ↓
Multi에도 상당히 잘 generalize

Multi-1000 training
    ↓
Multi 성능은 소폭 개선
Single 성능은 하락
```

즉 **cardinality 변화에 비교적 robust**하다.

### Qwen

```text
Single-only training
    ↓
Single 매우 강함
Multi 크게 실패

Multi-1000 training
    ↓
Multi 매우 강함
Single 성능은 감소
```

즉 **training output cardinality에 매우 민감**하다.

이를 가장 간단하게 표현하면:

$$
\boxed{
\text{BERT: local token-level generalization}
}
$$

$$
\boxed{
\text{Qwen: output-structure-dependent specialization}
}
$$

이번 결과에서는 **Qwen의 multi-entity failure가 모델의 근본적인 compositional reasoning 부족이라기보다는 training distribution과 output structure mismatch에서 상당 부분 발생했다는 증거가 강하다.**
