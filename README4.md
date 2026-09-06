````markdown
## Qwen3.5 NER Generalization Experiment

### 실험 목적

Qwen3.5 계열 모델이 **single-entity NER supervision만으로 multi-entity / multi-type NER에 얼마나 일반화하는지** 확인했다.

비교 조건은 다음 두 가지다.

- **Zero-shot**: pretrained Qwen을 별도 NER 학습 없이 prompting만 사용
- **Single-entity SFT**: CoNLL-2003 train 중 entity가 정확히 1개인 example만 사용해 fine-tuning

평가 split은 다음과 같이 구성했다.

- `single_entity`: entity 1개
- `exactly_2_entities`: entity 정확히 2개
- `3plus_entities`: entity 3개 이상
- `multi_entity`: entity 2개 이상
- `multi_entity_same_type`: entity는 여러 개지만 type은 1종류
- `multi_type`: entity가 여러 개이며 type도 2종류 이상

---

## 전체 결과

| Split | Zero-shot F1 | SFT F1 | SFT - Zero | Zero-shot Exact | SFT Exact |
|---|---:|---:|---:|---:|---:|
| single entity | 0.2749 | **0.8688** | **+0.5939** | 0.2058 | **0.8688** |
| exactly 2 entities | 0.4080 | **0.5606** | +0.1526 | 0.1545 | 0.0000 |
| 3+ entities | **0.4792** | 0.3213 | **-0.1579** | 0.0916 | 0.0000 |
| multi entity | **0.4518** | 0.4262 | -0.0256 | 0.1301 | 0.0000 |
| multi entity, same type | 0.4357 | **0.5246** | +0.0889 | 0.2122 | 0.0000 |
| multi type | **0.4543** | 0.3964 | **-0.0579** | 0.0969 | 0.0000 |

---

## 1. Single-entity fine-tuning 효과는 매우 큼

가장 직접적인 효과는 IID 조건에서 나타난다.

```text
single_entity

Zero-shot F1 : 0.2749
SFT F1       : 0.8688

ΔF1          : +0.5939
````

즉 single-entity example만으로 SFT했을 때 F1이 약 **0.27 → 0.87**로 크게 상승했다.

이는 fine-tuning을 통해 모델이 CoNLL의 annotation convention과 entity schema를 매우 잘 학습했다는 의미다.

특히 sentence exact match 역시:

```text
0.2058 → 0.8688
```

로 상승했다.

따라서 **atomic NER skill 자체는 single-entity supervision만으로 충분히 학습 가능**하다고 볼 수 있다.

---

## 2. 하지만 SFT 후 compositional generalization이 크게 무너짐

가장 중요한 결과다.

SFT 모델의 성능은 다음과 같다.

```text
single entity    : 0.8688
same-type multi  : 0.5246
multi-type       : 0.3964
```

single entity를 기준으로 하면:

```text
same-type multi gap = -0.3442
multi-type gap      = -0.4725
```

상대 감소율은 각각:

```text
same-type multi : 39.6%
multi-type      : 54.4%
```

즉 모델은 single entity를 매우 잘 풀게 되었지만, entity가 여러 개 등장하는 순간 generalization gap이 상당히 커졌다.

특히 서로 다른 type을 동시에 처리해야 하는 `multi_type`에서는:

$$
0.8688 \rightarrow 0.3964
$$

로 절반 이하 수준까지 떨어졌다.

따라서 이 결과는 다음과 같이 요약할 수 있다.

> **Single-entity SFT strongly specializes the model to the atomic training distribution, but this specialization does not compose reliably when multiple entities must be extracted jointly.**

---

## 3. Zero-shot은 반대로 multi-entity에서 더 잘 나옴

Zero-shot 결과는 특이한 패턴을 보인다.

```text
single entity    : 0.2749
same-type multi  : 0.4357
multi-type       : 0.4543
3+ entities      : 0.4792
```

즉 multi-entity가 single-entity보다 오히려 성능이 높다.

Generalization gap도 실제로 양수다.

```text
same-type gap = +0.1608
multi-type gap = +0.1794
```

이걸 **“zero-shot 모델이 multi-entity에 더 잘 generalize한다”**고 바로 해석하면 조금 위험하다.

더 가능성 높은 설명은 **dataset composition과 evaluation 특성**이다.

예를 들어 multi-entity sentence는 named entity가 더 많이 포함되므로 모델이 일부 entity를 맞출 기회 자체가 많다.

F1은 sentence당 평균이 아니라 전체 entity 기준 micro metric이기 때문에:

```text
문장 A: entity 1개 중 0개 맞춤
문장 B: entity 5개 중 3개 맞춤
```

같은 상황에서 B가 훨씬 유리하다.

따라서 zero-shot에서 나타나는:

$$
F1_{multi} > F1_{single}
$$

은 strong compositional generalization의 직접적인 증거라기보다는 **entity density / difficulty distribution 차이의 영향**도 포함한다고 보는 게 안전하다.

---

## 4. SFT의 가장 중요한 현상: specialization-generalization trade-off

Zero-shot과 SFT를 직접 비교하면 패턴이 더 명확하다.

### Single entity

```text
Zero : 0.2749
SFT  : 0.8688
```

SFT 압승이다.

### Same-type multi

```text
Zero : 0.4357
SFT  : 0.5246
```

SFT가 여전히 이득이다.

### Multi-type

```text
Zero : 0.4543
SFT  : 0.3964
```

오히려 **zero-shot이 더 높다.**

### 3+ entities

```text
Zero : 0.4792
SFT  : 0.3213
```

차이가 더 커진다.

즉 training distribution과 멀어질수록:

```text
SFT advantage
    ↓
    ↓
zero-shot advantage
```

로 변한다.

이는 전형적인 **specialization vs generalization trade-off** 형태다.

---

## 5. 핵심 결과를 구조적으로 보면

```text
                     Zero-shot           SFT
                         │                 │
                         │                 │
single               0.2749            0.8688
                         │                 │
                         │              -39.6%
same-type multi      0.4357            0.5246
                         │                 │
                         │              -54.4%
multi-type           0.4543            0.3964
```

Zero-shot은 절대 성능은 낮지만 complexity 증가에 따른 degradation이 거의 없다.

반면 SFT 모델은 atomic task에서는 매우 높지만 multi-entity composition에서 급격하게 무너진다.

즉:

$$
\text{SFT improves task acquisition}
$$

하지만 동시에

$$
\text{SFT can reduce compositional robustness}
$$

라는 패턴이다.

---

## 6. Same-type보다 multi-type이 더 어려움

SFT 결과에서:

```text
same-type multi : 0.5246
multi-type      : 0.3964
```

차이는:

$$
0.5246 - 0.3964 = 0.1282
$$

정도다.

이는 단순히 entity 개수가 늘어나는 것보다 **서로 다른 label type을 동시에 composition하는 것이 더 어렵다**는 이전 BERT 실험과 방향성이 일치한다.

즉:

```text
PER + PER
```

보다는

```text
PER + ORG + LOC
```

같은 조합에서 더 크게 깨진다.

따라서 generalization complexity는 단순한:

$$
\#entities
$$

뿐만 아니라:

$$
\#distinct\ entity\ types
$$

에도 강하게 좌우된다고 볼 수 있다.

---

## 7. 3+ entity에서 SFT가 특히 크게 붕괴

가장 강한 OOD split은 `3plus_entities`다.

```text
Zero-shot : 0.4792
SFT       : 0.3213
```

SFT 기준 single-entity 대비:

```text
absolute gap = -0.5475
relative drop = 63.0%
```

즉 학습 당시 한 번에 entity 1개만 보던 모델에게 entity 3개 이상이 등장하면 성능이 크게 무너졌다.

이 결과는 다음 hypothesis를 강하게 지지한다.

> **Learning an entity extraction primitive does not necessarily imply that the model learns to repeatedly apply that primitive an arbitrary number of times.**

즉 atomic skill을 아는 것과 **iterative/compositional execution**은 별개의 능력일 수 있다.

---

## 8. Sentence Exact Match = 0은 특히 중요함

SFT 모델은 multi-entity split에서 sentence exact match가 전부 0이다.

```text
exactly 2 entities    : 0.0000
3+ entities           : 0.0000
multi entity          : 0.0000
same-type multi       : 0.0000
multi-type            : 0.0000
```

반면 F1은 0.32~0.56 정도가 나온다.

이건 모델이 entity를 **일부는 맞추지만 전체 구조를 완벽하게 맞추는 문장은 사실상 없었다**는 뜻이다.

예를 들어:

```text
Gold:
John    PER
Google  ORG
London  LOC

Prediction:
John    PER
Google  ORG
```

라면 F1은 꽤 높지만 sentence exact는 실패다.

따라서 application 관점에서는 현재 SFT 모델의 multi-entity robustness가 F1이 보여주는 것보다 훨씬 더 나쁠 수 있다.

---

## 9. 이전 BERT 결과와 비교

이전 BERT single-entity fine-tuning 결과는:

| Model                 |     Single | Same-type Multi | Multi-type |
| --------------------- | ---------: | --------------: | ---------: |
| BERT token classifier | **0.8775** |      **0.8738** | **0.7356** |
| Qwen SFT              |     0.8688 |          0.5246 |     0.3964 |

두 모델의 single-entity 성능은 거의 같다.

```text
BERT : 0.8775
Qwen : 0.8688
```

그런데 multi-entity에서는 차이가 매우 크다.

```text
same-type multi

BERT : 0.8738
Qwen : 0.5246
```

그리고 multi-type:

```text
BERT : 0.7356
Qwen : 0.3964
```

즉 이 실험에서는 오히려 **BERT token classifier가 generative Qwen보다 compositional generalization을 훨씬 잘 유지했다.**

가능한 이유는 architecture/interface 차이다.

BERT NER는 각 token에 대해:

$$
p(y_i|x)
$$

를 직접 예측하므로 동일한 local tagging operation을 여러 position에 반복하기 쉽다.

반면 generative LLM은:

$$
p(\text{JSON entity list}|x)
$$

전체 output sequence를 autoregressive하게 생성해야 한다.

따라서 training에서 항상 **entity 하나짜리 JSON array만 봤다면**

```json
[
  {"text": "...", "type": "..."}
]
```

라는 출력 구조 자체를 강하게 학습했을 가능성이 있다.

즉 현재 결과는 단순한 semantic generalization 실패가 아니라 **output-cardinality bias**까지 섞여 있을 가능성이 매우 높다.

---

## 10. 가장 중요한 추가 가설: cardinality bias

현재 결과에서 가장 의심되는 부분이다.

SFT train은 항상 entity 하나뿐이므로 target도 항상:

```json
[
  {"text": "Google", "type": "ORG"}
]
```

처럼 **JSON object가 정확히 1개**다.

그런데 test에서는:

```json
[
  {"text": "John", "type": "PER"},
  {"text": "Google", "type": "ORG"},
  {"text": "London", "type": "LOC"}
]
```

처럼 여러 개를 출력해야 한다.

따라서 모델은 사실:

> NER task를 single-hop으로 학습했다

기보다는

> **“항상 object 하나를 출력하는 task”를 학습했다**

일 수도 있다.

이게 맞다면 현재 실험은 semantic compositionality뿐 아니라 **output length/cardinality generalization**도 함께 측정하고 있다.

특히 multi-entity sentence exact match가 모두 0인 점은 이 가설과 잘 맞는다.

---

## 최종 결론

이번 실험의 가장 중요한 결론은 다음과 같다.

> **Single-entity SFT는 Qwen의 atomic NER 성능을 매우 크게 향상시키지만, 그 성능 향상은 multi-entity setting으로 자연스럽게 composition되지 않는다.**

구체적으로:

* single entity:

  * `0.2749 → 0.8688`
  * 매우 강한 specialization 효과
* same-type multi:

  * SFT F1 `0.5246`
  * single 대비 **39.6% 상대 하락**
* multi-type:

  * SFT F1 `0.3964`
  * single 대비 **54.4% 상대 하락**
* 3+ entity:

  * SFT F1 `0.3213`
  * single 대비 **63.0% 상대 하락**
* multi-type과 3+ entity에서는 오히려 zero-shot 모델이 SFT 모델보다 높음

따라서 이번 결과는 다음과 같이 요약할 수 있다.

> **Fine-tuning can substantially improve in-distribution atomic task performance while simultaneously reducing out-of-distribution compositional robustness.**

다만 현재 generative NER setup에서는 **single-entity training이 항상 1-item JSON target을 사용하기 때문에 output-cardinality bias가 결과에 상당히 개입했을 가능성**이 있다. 따라서 다음 실험에서는 semantic entity 수는 1개로 유지하면서 target format에는 dummy/non-entity 구조 또는 variable-length output을 주거나, 반대로 multi-output formatting만 학습시키는 control experiment를 추가하는 것이 필요하다.

### 가장 추천하는 다음 ablation

현재 결과에서 바로 이어서 해야 할 것은 세 조건 비교다.

```text
A. single entity + 항상 output 1개
B. single entity + variable output-format exposure
C. multi entity training
```

만약:

```text
A multi-type F1 = 0.40
B multi-type F1 = 0.65
C multi-type F1 = 0.80
```

처럼 나온다면 현재 failure 상당 부분이 **semantic compositionality가 아니라 output cardinality generalization 문제**였다는 걸 분리할 수 있다.

이 ablation까지 하면 실험 해석이 훨씬 강해진다.

```
```
