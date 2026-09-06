
## NER Cardinality Generalization: Fair 1,000-Sample Comparison

### 1. 실험 목적

NER 학습에서 **single-entity supervision과 multi-entity supervision이 일반화 성능에 어떤 영향을 주는지** 비교했다.

이번 비교에서는 학습 데이터 수와 epoch을 동일하게 맞췄다.

* **Single-1000**: entity가 정확히 1개인 문장 1,000개
* **Multi-1000**: entity가 2개 이상인 문장 1,000개
* 동일한 CoNLL-2003 test split 사용
* 동일 epoch
* 동일한 기본 모델 및 학습 설정
* Qwen은 추가로 **Zero-shot** 결과와 비교

평가 split:

| Split                    | 의미                         |
| ------------------------ | -------------------------- |
| `single_entity`          | entity가 정확히 1개             |
| `exactly_2_entities`     | entity가 정확히 2개             |
| `3plus_entities`         | entity가 3개 이상              |
| `multi_entity`           | entity가 2개 이상 전체           |
| `multi_entity_same_type` | entity 여러 개, type은 한 종류    |
| `multi_type`             | entity 여러 개, type도 두 종류 이상 |

Qwen zero-shot 결과는 별도 fine-tuning 없이 prompt만 사용한 결과이다.

---

# 2. Qwen: Zero-shot vs Single-1000 vs Multi-1000

## F1 비교

| Test split         | Zero-shot | Single-1000 | Multi-1000 | 최고              |
| ------------------ | --------: | ----------: | ---------: | --------------- |
| Single entity      |    0.2749 |  **0.8405** |     0.7046 | **Single-1000** |
| Exactly 2 entities |    0.4080 |      0.5288 | **0.8576** | **Multi-1000**  |
| 3+ entities        |    0.4792 |      0.3152 | **0.8324** | **Multi-1000**  |
| Multi entity       |    0.4518 |      0.4081 | **0.8433** | **Multi-1000**  |
| Same-type multi    |    0.4357 |      0.4644 | **0.8365** | **Multi-1000**  |
| Multi-type         |    0.4543 |      0.3923 | **0.8427** | **Multi-1000**  |

Zero-shot 수치는 원본 Qwen 실험 문서의 결과이고, Multi-1000 수치는 multi-entity 1,000개 학습 실험 결과다.

---

## 3. Qwen의 핵심 결과: 학습 cardinality에 매우 민감함

### Single-entity test

```text
Zero-shot     0.2749
Single-1000   0.8405
Multi-1000    0.7046
```

Single-1000 학습이 가장 좋다.

Zero-shot 대비:

$$
0.2749 \rightarrow 0.8405
$$

로 **+0.5656 F1**이 상승했다.

즉 entity 하나를 추출하는 task는 single-entity supervision 1,000개만으로도 매우 잘 학습된다.

---

### Multi-entity test

반대로 multi test에서는 결과가 완전히 뒤집힌다.

```text
Zero-shot     0.4518
Single-1000   0.4081
Multi-1000    0.8433
```

Multi-1000은 Single-1000보다:

$$
0.8433-0.4081
=
\mathbf{+0.4352}
$$

높다.

상대적으로는 약:

$$
\frac{0.8433-0.4081}{0.4081}
\approx
106.7\%
$$

상승이다.

즉 **학습량을 동일하게 1,000개로 통제해도 multi-entity examples를 직접 학습하는 것이 압도적으로 중요하다.**

---

# 4. Qwen Single-1000의 가장 특징적인 현상: 높은 Precision, 낮은 Recall

Single-1000 Qwen의 multi test를 보면:

| Split         | Precision |     Recall |     F1 |
| ------------- | --------: | ---------: | -----: |
| Exactly 2     |    0.7933 |     0.3966 | 0.5288 |
| 3+            |    0.8309 | **0.1945** | 0.3152 |
| Multi overall |    0.8063 | **0.2731** | 0.4081 |
| Multi-type    |    0.8228 | **0.2576** | 0.3923 |

이 패턴은 굉장히 명확하다.

모델이 잘못된 entity를 마구 생성하는 게 주된 문제가 아니다.

오히려:

> **찾은 entity는 꽤 정확하지만, 여러 개 중 대부분을 출력하지 않는다.**

예를 들어 gold가:

```text
John       PER
Google     ORG
London     LOC
```

일 때 Single-1000 Qwen은 대략:

```text
John       PER
```

하나만 제대로 출력하는 식의 behavior를 보일 가능성이 높다.

즉 **under-extraction**이 핵심 failure mode다.

---

# 5. Sentence Exact Match가 이를 더 강하게 보여줌

Single-1000 Qwen:

```text
single entity        0.8405

exactly 2            0.0000
3+                    0.0000
multi entity          0.0000
same-type multi       0.0000
multi-type            0.0000
```

Single entity에서는 **84.05%의 문장을 완전히 맞추는데**, multi entity가 되는 순간 complete entity set을 맞춘 문장이 하나도 없다.

반면 Multi-1000 Qwen은:

```text
single             0.6237
exactly 2          0.7445
3+                 0.5096
multi overall      0.6440
same-type multi    0.7220
multi-type         0.6142
```

로 multi-output을 실제로 완성할 수 있게 된다.

이 결과는 Qwen의 문제를 단순한 label classification failure보다 **output cardinality / structured generation behavior**로 보는 근거가 매우 강하다.

---

# 6. Zero-shot이 Single-1000보다 오히려 나은 영역도 있음

특히:

```text
               Zero     Single-1000

3+             .4792       .3152
multi          .4518       .4081
multi-type     .4543       .3923
```

이다.

즉 single-only fine-tuning을 하고 나서 오히려 pretrained zero-shot보다 multi generalization이 나빠지는 영역이 있다.

이는 매우 흥미로운 결과다.

Single-1000 SFT가 모델에게:

> “NER에서는 하나의 entity를 찾아서 하나를 반환한다”

라는 강한 task pattern을 주면서, pretrained 모델이 갖고 있던 variable-cardinality generation 능력을 일부 억제한 것으로 해석할 수 있다.

따라서 Qwen에서는:

$$
\text{SFT}
\neq
\text{항상 더 좋은 generalization}
$$

이다.

보다 정확히는:

$$
\text{SFT} \rightarrow
\text{training distribution specialization}
$$

이라고 보는 게 맞다.

---

# 7. Multi-1000은 이 문제를 거의 완전히 해결

Multi-1000 Qwen:

```text
exactly 2       0.8576
3+              0.8324
multi           0.8433
same-type       0.8365
multi-type      0.8427
```

multi condition 안에서는 전부 **0.83~0.86** 수준으로 매우 안정적이다.

특히:

```text
same-type    0.8365
multi-type   0.8427
```

가 거의 동일하다.

따라서 multi-output을 학습시킨 뒤에는:

> PER + PER

과

> PER + ORG + LOC

사이에 사실상 큰 난이도 차이가 없다.

즉 Qwen의 이전 multi-type 실패를

> “서로 다른 entity type을 composition할 능력이 부족하다”

라고 해석하는 것은 적절하지 않다.

오히려:

> **multi-output 구조 자체를 학습하지 않았던 것이 더 큰 병목이었다.**

는 설명이 이번 fair experiment에서도 강하게 지지된다.

---

# 8. BERT: Single-1000 vs Multi-1000

BERT도 학습량 1,000개로 동일하게 맞추면 상당히 명확한 차이가 발생한다.

| Test split      | Single-1000 | Multi-1000 | Multi − Single |
| --------------- | ----------: | ---------: | -------------: |
| Single entity   |      0.6448 | **0.6767** |        +0.0319 |
| Exactly 2       |      0.5394 | **0.8338** |        +0.2944 |
| 3+              |      0.4229 | **0.8064** |        +0.3835 |
| Multi overall   |      0.4690 | **0.8174** |        +0.3484 |
| Same-type multi |      0.5743 | **0.8622** |        +0.2879 |
| Multi-type      |      0.4402 | **0.8050** |        +0.3648 |

Multi-1000 BERT 결과는 기존 multi-entity 실험에서 보고된 값이다.

놀랍게도 **BERT도 1,000개로 sample size를 통제하자 single-only training의 multi generalization이 기존 4,996개 실험보다 상당히 낮아졌다.**

---

# 9. 기존 BERT 해석도 일부 수정해야 함

이전 4,996개 single-entity BERT 결과에서는:

```text
single       0.8775
same-type    0.8732
multi-type   0.7356
```

이어서:

> BERT는 single-entity training만으로 multi에 매우 잘 generalize한다.

고 해석할 수 있었다. 기존 문서에도 이 방향의 해석이 기록돼 있다.

하지만 **Single-1000으로 맞추면**:

```text
single       0.6448
same-type    0.5743
multi-type   0.4402
```

로 내려간다.

따라서 더 정확한 결론은:

> **BERT의 token-wise inductive bias가 multi generalization을 돕기는 하지만, 충분한 single-entity training coverage 역시 중요하다.**

이다.

즉 기존 4,996개 결과의 강한 generalization 중 일부는 architecture뿐 아니라 **더 많은 training examples**의 효과였다.

---

# 10. 그래도 Qwen이 cardinality shift에 더 민감함

동일하게 Single-1000에서 `single → multi`로 넘어갈 때:

### BERT

$$
0.6448
\rightarrow
0.4690
$$

absolute gap:

$$
-0.1758
$$

### Qwen

$$
0.8405
\rightarrow
0.4081
$$

absolute gap:

$$
-0.4324
$$

즉 Qwen의 감소가 약 **2.46배 더 크다.**

Multi-type에서도:

### BERT

$$
0.6448 \rightarrow 0.4402
$$

$$
\Delta=-0.2046
$$

### Qwen

$$
0.8405 \rightarrow 0.3923
$$

$$
\Delta=-0.4482
$$

이다.

따라서 sample size까지 통제한 뒤에도:

> **Generative Qwen은 BERT token classifier보다 training-time entity cardinality에 훨씬 민감하다.**

는 결론은 유지된다.

---

# 11. BERT와 Qwen의 inductive bias 차이

### BERT

BERT NER에서는 각 위치마다:

$$
p(y_i|x)
=
\operatorname{softmax}(Wh_i+b)
$$

로 NER label을 결정한다.

즉 entity가 하나에서 여러 개로 늘어나더라도 기본 operation은:

> 각 token을 분류

하는 것으로 동일하다.

따라서 training cardinality가 달라져도 구조적인 output format 변화는 없다.

---

### Qwen

Qwen은:

$$
p(y_1,\ldots,y_T|x)
=
\prod_t p(y_t|y_{<t},x)
$$

형태로 전체 JSON sequence를 생성한다.

Single training:

```json
[
  {"text":"John","type":"PER"}
]
```

Multi training:

```json
[
  {"text":"John","type":"PER"},
  {"text":"Google","type":"ORG"},
  {"text":"London","type":"LOC"}
]
```

는 단순히 entity classifier를 반복하는 것이 아니라 **output sequence 자체가 달라지는 문제**다.

그래서 Qwen은 training output 구조의 영향을 훨씬 많이 받는다.

---

# 12. 가장 공정한 최종 비교

## Single-1000 → Multi test

```text
BERT    0.4690
Qwen    0.4081
```

BERT가 더 강하다.

## Multi-1000 → Multi test

```text
BERT    0.8174
Qwen    0.8433
```

이번에는 Qwen이 더 강하다.

즉 재미있는 crossover가 생긴다.

```text
                         Single training       Multi training
                         ───────────────       ──────────────
Multi test

BERT                         0.4690               0.8174
Qwen                         0.4081               0.8433
```

이를 해석하면:

> **BERT는 unseen cardinality로의 generalization이 상대적으로 더 좋고, Qwen은 target cardinality를 실제로 학습했을 때 더 높은 ceiling을 보인다.**

라고 정리할 수 있다.

---

# 13. Multi-type에서도 동일한 crossover

```text
                         Single-1000       Multi-1000
BERT                        0.4402            0.8050
Qwen                        0.3923            0.8427
```

Single supervision만 있을 때는:

$$
BERT > Qwen
$$

하지만 multi supervision을 주면:

$$
Qwen > BERT
$$

로 역전된다.

이것도 꽤 중요한 결과다.

---

# 14. Qwen의 세 조건을 한눈에 보면

```text
                     Zero-shot    Single-1000    Multi-1000
Single                 0.2749        0.8405         0.7046

Exactly 2              0.4080        0.5288         0.8576
3+                     0.4792        0.3152         0.8324
Multi overall          0.4518        0.4081         0.8433

Same-type              0.4357        0.4644         0.8365
Multi-type             0.4543        0.3923         0.8427
```

이 표 하나로 거의 전체 실험이 설명된다.

### Zero-shot

* NER 절대 성능은 낮음
* cardinality 변화에 특별히 한쪽으로 강하게 specialization되어 있지는 않음

### Single-1000

* Single entity에 매우 강함
* Multi에서는 recall이 급락
* 3+ entity에서는 zero-shot보다도 나빠짐

### Multi-1000

* Multi 전 구간에서 압도적
* Precision/Recall 균형도 좋음
* 대신 single 성능은 Single-1000보다 낮음

---

# 15. 최종 결론

이번에는 **Single과 Multi 모두 1,000 examples, 동일 epoch**으로 맞췄기 때문에 이전보다 훨씬 강한 결론을 낼 수 있다.

### 첫째, training cardinality 자체가 성능을 강하게 결정한다.

$$
\boxed{
\text{Train cardinality}
\rightarrow
\text{Test cardinality specialization}
}
$$

Single examples를 학습하면 single test에 강하고, multi examples를 학습하면 multi test에 강하다.

---

### 둘째, 이 현상은 Qwen에서 훨씬 강하다.

Single-1000에서:

```text
BERT
single → multi:
0.6448 → 0.4690
Δ = -0.1758

Qwen
single → multi:
0.8405 → 0.4081
Δ = -0.4324
```

따라서 generative LLM이 token classifier보다 cardinality distribution shift에 훨씬 민감하다.

---

### 셋째, Qwen의 문제는 multi-type reasoning 자체가 아니다.

Multi-1000에서는:

```text
same-type   0.8365
multi-type  0.8427
```

로 사실상 동일하다.

따라서 Qwen single-only 모델의 multi-type 실패는:

$$
\text{semantic composition failure}
$$

라기보다는

$$
\boxed{
\text{output-cardinality / output-structure mismatch}
}
$$

의 영향이 훨씬 크다.

---

### 넷째, BERT와 Qwen은 서로 다른 장점을 가진다.

$$
\boxed{
\text{BERT}
=
\text{better unseen-cardinality generalization}
}
$$

$$
\boxed{
\text{Qwen}
=
\text{higher performance when target output structure is supervised}
}
$$

실제로 multi test에서:

```text
Single-1000:
BERT 0.4690 > Qwen 0.4081

Multi-1000:
Qwen 0.8433 > BERT 0.8174
```

로 crossover가 관찰된다.

---

## 한 줄 요약

> **동일한 1,000개 학습량으로 통제해도 single-entity supervision은 multi-entity extraction으로 완전히 일반화되지 않았으며, 특히 generative Qwen에서 그 gap이 매우 컸다. 반면 multi-entity examples를 직접 학습하면 Qwen은 BERT보다 높은 multi-entity 성능까지 달성했다. 이는 generative NER에서 semantic label knowledge뿐 아니라 training-time output cardinality와 구조가 핵심적인 generalization factor임을 보여준다.**
