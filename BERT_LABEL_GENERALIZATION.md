결과가 꽤 선명해. 결론부터 말하면:

**single-entity로만 학습해도 “같은 타입 entity 여러 개”에는 거의 그대로 일반화되지만, “서로 다른 타입을 한 문장 안에서 조합”해야 하는 multi-type에서는 성능이 크게 무너진다.**

즉, 단순히 entity 수가 늘어서 망가지는 게 아니라 **label composition이 핵심 generalization bottleneck**으로 보여.

### 1. Single → multi-entity 자체는 어느 정도 일반화됨

기준 성능은:

* single entity F1 = **0.8775**
* multi entity F1 = **0.7661**
* 상대 F1 drop = **12.7%**

즉 single-entity만 학습했는데도 2개 이상 entity가 나오는 문장에서 F1 0.77 정도를 유지한다.

이건 BERT token classifier가

> "John" → PER
> "Google" → ORG

같이 개별적인 local tagging rule을 배운 뒤, 어느 정도는 한 문장 안에서 여러 번 적용할 수 있다는 뜻이야.

---

### 2. 가장 중요한 결과: same-type multi-entity는 거의 성능 저하가 없음

여기가 제일 재밌어.

| Test                    |         F1 |   single 대비 |
| ----------------------- | ---------: | ----------: |
| single entity           |     0.8775 |           - |
| multi entity, same type | **0.8738** | **-0.0037** |
| multi type              | **0.7356** | **-0.1418** |

`multi_entity_same_type`이 **0.8738**이야.

single-entity의 0.8775와 사실상 똑같아.

상대 감소도 겨우:

$$
\frac{0.8775 - 0.8738}{0.8775}
\approx 0.42\%
$$

즉 이런 건 잘 처리한다는 뜻:

```text
[John]PER met [Mary]PER.
```

학습할 때 PER가 한 개만 나왔어도, test 때 PER가 여러 개 등장하는 것은 거의 문제가 안 돼.

그래서 **“single entity만 학습해서 multi entity에 generalize 못 한다”**라고 결론 내리면 안 되고, 좀 더 정확하게는:

> **Entity multiplicity 자체에는 잘 generalize한다.**

라고 해야 해.

---

## 3. 반면 multi-type composition에서 크게 깨짐

`multi_type`:

* F1 = **0.7356**
* single 대비 absolute drop = **-0.1418**
* relative drop = **16.16%**

예를 들어 이런 경우겠지:

```text
[John]PER works at [Google]ORG in [London]LOC.
```

모델은 training에서 각각을 따로 본 적은 있어도,

```text
PER
ORG
LOC
```

가 **한 문장에 같이 등장하는 distribution**을 충분히 학습하지 못한 상태야.

결과적으로:

$$
\text{atomic label knowledge}
\not\Rightarrow
\text{perfect label composition}
$$

이라는 꽤 좋은 empirical result가 나온 거야.

---

## 4. `exactly_2_entities`가 `3plus_entities`보다 오히려 나쁨

조금 특이하게:

* exactly 2 entities = **0.7409**
* 3+ entities = **0.7825**

entity 수가 많을수록 monotonically 떨어지진 않았어.

이것 때문에 **“entity 수가 많으면 성능이 떨어진다”라고 해석하면 안 돼.**

오히려 아까 same-type 결과와 합쳐 보면 설명이 된다.

2 entities 집합 안에 이런 어려운 샘플이 많을 수 있어:

```text
PER + ORG
PER + LOC
ORG + LOC
```

반면 3+ entity dataset에는:

```text
PER + PER + PER
ORG + ORG + ORG
```

같은 반복적이고 쉬운 example도 들어갈 수 있음.

그러니까 complexity는

$$
\text{# entities}
$$

보다는

$$
\text{# distinct entity types}
$$

에 더 강하게 의존하는 것처럼 보여.

---

## 5. Exact Match를 보면 실제 degradation은 훨씬 큼

F1만 보면:

```text
single       0.8775
multi-type   0.7356
```

라서 “조금 떨어졌네” 정도로 보일 수 있는데, sentence exact match는:

```text
single       0.8696
multi-type   0.3681
```

이야.

즉 multi-type sentence의 **약 63%에서는 적어도 하나 이상의 token/entity tagging error가 발생**했다는 뜻이야.

차이는:

$$
0.3681 - 0.8696 = -0.5015
$$

무려 **50.15%p 감소**.

이게 실제 application 관점에서는 상당히 중요해.

예를 들어:

```text
John works for Google in London.
```

에서

```text
John    PER ✓
Google  ORG ✓
London  LOC ✗
```

이면 entity-level micro-F1에는 부분 점수를 받지만, structured extraction 전체 관점에서는 요청 하나가 실패한 거니까.

그래서 네 실험에서는 **F1과 exact match 둘 다 제시하는 게 맞아.**

---

## 6. Precision보다 Recall 하락이 더 큼

multi-type:

```text
precision = 0.7619
recall    = 0.7112
```

single:

```text
precision = 0.8690
recall    = 0.8861
```

변화량은 대략:

$$
\Delta P = -0.107
$$

$$
\Delta R = -0.175
$$

즉 multi-type에서는 모델이 엉뚱한 entity를 많이 추가하는 것보다 **gold entity를 놓치는 문제**가 더 커.

쉽게 말하면 모델 behavior가:

```text
John works at Google in London

John   -> PER ✓
Google -> ORG ✓
London -> O   ✗
```

같은 **under-extraction** 쪽으로 변한다는 가능성이 높아.

이건 `multi_type_errors.json`을 보면 확인 가능해.

---

## 7. Token accuracy가 높은 건 별로 중요하지 않음

token accuracy:

```text
single       0.9796
multi-type   0.9446
```

둘 다 아주 높아 보여.

그런데 NER에서는 대부분 token이 `O`잖아.

예를 들어:

```text
The president of Google visited London yesterday
 O      O      O   ORG      O    LOC     O
```

대부분 O라서 entity를 많이 놓쳐도 token accuracy가 높게 유지된다.

따라서 이 실험에서는:

**F1 > Exact Match >>> token accuracy**

순으로 중요하게 보면 돼.

---

# 이 결과가 말해주는 핵심

결과를 구조적으로 보면:

```text
                 F1
Single entity    0.8775
       │
       ├── same-type repetition
       │        ↓
       │      0.8738       거의 유지
       │
       └── different-type composition
                ↓
              0.7356       크게 하락
```

그래서 현재 실험에서 가장 강한 결론은:

> **A NER model trained only on atomic single-entity examples generalizes almost perfectly to repeated entities of the same type, but exhibits a substantial generalization gap when multiple distinct entity types must be composed within the same sentence.**

한국어로 하면:

> **single-entity 학습은 동일 entity type의 반복에는 거의 완벽하게 일반화되지만, 서로 다른 entity type이 동시에 등장하는 compositional setting에서는 약 16%의 상대 F1 하락이 발생했다.**

이게 네 결과의 핵심이야.

---

## 그리고 꽤 중요한 해석 하나

처음 질문이:

> single turn single hop으로 해도 multi-hop/multi-label에 generalize 되냐?

였잖아.

현재 결과로 답하면:

**부분적으로 yes.**

동일 skill을 여러 번 적용하는:

$$
A \rightarrow A+A+A
$$

에는 거의 완벽하게 generalize.

하지만 서로 다른 learned skill을 조합하는:

$$
A,\ B,\ C
\rightarrow
A+B+C
$$

에서는 꽤 큰 gap이 생김.

즉 이 결과는 사실 **compositional generalization** 실험으로 해석하는 게 가장 좋다.

---

### 다음으로 가장 가치 있는 ablation

이제 `multi_type`을 type 수별로 쪼갠 CSV를 보면 더 확실해져.

예를 들어 실제 결과가:

```text
1 type     0.87
2 types    0.76
3+ types   0.68
```

처럼 나오면,

$$
\text{F1} \downarrow
\quad \text{as} \quad
|\text{label types}| \uparrow
$$

라는 **composition complexity curve**까지 얻는다.

이게 나오면 지금 실험에서 가장 설득력 있는 결과가 될 거야.

그리고 다음 단계는 딱 **multi-turn**이야. 동일한 방식으로 single-turn만 train한 뒤 `current utterance only`와 `history + current utterance`를 테스트하면, **label compositional generalization과 contextual generalization을 완전히 분리해서** 볼 수 있어.
