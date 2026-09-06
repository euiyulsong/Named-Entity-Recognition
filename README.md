
# NER Model Comparison — CoNLL-2003
<img width="2200" height="1400" alt="positive_negative_distribution" src="https://github.com/user-attachments/assets/bf558223-ca9f-452f-bcb2-d69561e0e60b" />
## 1. Experiment Overview

CoNLL-2003를 이용해 네 가지 NER 모델을 비교했다.

* **BiLSTM**
* **BiLSTM + CRF**
* **BERT**
* **BERT + CRF**

NER는 각 token에 대해 BIO entity label을 예측하는 **token classification / sequence labeling** 문제로 구성했다.

---

## 2. Dataset Exploration

### Dataset Size

| Split      | Sentences |
| ---------- | --------: |
| Train      |    14,041 |
| Validation |     3,250 |
| Test       |     3,453 |

### NER Labels

총 **9개 클래스**를 사용한다.

| ID | Label  | Description             |
| -: | ------ | ----------------------- |
|  0 | O      | Entity가 아닌 token        |
|  1 | B-PER  | Person 시작               |
|  2 | I-PER  | Person 내부               |
|  3 | B-ORG  | Organization 시작         |
|  4 | I-ORG  | Organization 내부         |
|  5 | B-LOC  | Location 시작             |
|  6 | I-LOC  | Location 내부             |
|  7 | B-MISC | Miscellaneous entity 시작 |
|  8 | I-MISC | Miscellaneous entity 내부 |

`B-PER`, `I-PER` 등은 각각 **독립적인 classification class**로 사용했다.

### Training Tag Distribution

| Tag    | Token Count |
| ------ | ----------: |
| O      |     169,578 |
| B-LOC  |       7,140 |
| B-PER  |       6,600 |
| B-ORG  |       6,321 |
| I-PER  |       4,528 |
| I-ORG  |       3,704 |
| B-MISC |       3,438 |
| I-LOC  |       1,157 |
| I-MISC |       1,155 |

`O`가 압도적으로 많아 **class imbalance가 매우 큰 데이터셋**임을 확인할 수 있다.

### Sentence Length

| Statistic | Tokens |
| --------- | -----: |
| Mean      |  14.50 |
| P95       |     37 |
| Max       |    113 |

### Examples

```text
EU rejects German call to boycott British lamb .

EU      → B-ORG
German  → B-MISC
British → B-MISC
```

```text
Peter Blackburn

Peter     → B-PER
Blackburn → I-PER
```

```text
BRUSSELS 1996-08-22

BRUSSELS → B-LOC
```

BiLSTM vocabulary size는 **11,050**이었다.

---

# 3. Model Architecture

실험 구조는 다음과 같다.

```text
BiLSTM
Token → Embedding → BiLSTM → Linear → Softmax

BiLSTM + CRF
Token → Embedding → BiLSTM → Linear → CRF

BERT
Token → BERT → Linear → Softmax

BERT + CRF
Token → BERT → Linear → CRF
```

CRF 모델은 각 token을 완전히 독립적으로 결정하는 대신 **label transition과 전체 sequence score**를 함께 고려한다.

---

# 4. Final Results

Entity-level micro metrics 기준:

| Model        | Parameters |  Precision |     Recall |         F1 |
| ------------ | ---------: | ---------: | ---------: | ---------: |
| BiLSTM       |      1.68M |     0.5824 |     0.4924 |     0.5336 |
| BiLSTM + CRF |      1.68M |     0.6927 |     0.5595 | **0.6190** |
| BERT         |    108.32M |     0.8961 | **0.9254** | **0.9106** |
| BERT + CRF   |    108.32M | **0.9106** | **0.9254** | **0.9179** |

성능 순서는 명확했다.

```text
BERT + CRF   0.9179
BERT         0.9106
BiLSTM + CRF 0.6190
BiLSTM       0.5336
```

---

## 5. Effect of CRF

### BiLSTM

```text
BiLSTM       : 0.5336
BiLSTM + CRF : 0.6190

Δ F1 = +0.0854
```

약 **8.5 F1 point** 상승했다.

CRF의 효과가 상당히 크다. BiLSTM representation만으로 각 token을 독립적으로 분류하는 것보다 `B-PER → I-PER`, `B-ORG → I-ORG` 같은 **label sequence 관계를 추가로 모델링하는 것이 크게 도움**이 되었다.

### BERT

```text
BERT       : 0.9106
BERT + CRF : 0.9179

Δ F1 = +0.0073
```

BERT에서는 약 **0.7 F1 point** 상승에 그쳤다.

즉:

> **representation이 약한 BiLSTM에서는 CRF 효과가 크지만, contextual representation이 강한 BERT에서는 CRF의 추가 효과가 상대적으로 작았다.**

BERT의 self-attention이 이미 주변 token과 문장 전체 문맥을 강하게 encoding하기 때문이다.

---

# 6. Per-Entity Results

| Model        |   LOC F1 |  MISC F1 |   ORG F1 |   PER F1 |
| ------------ | -------: | -------: | -------: | -------: |
| BiLSTM       |     0.70 |     0.38 |     0.30 |     0.53 |
| BiLSTM + CRF | **0.74** | **0.55** | **0.42** | **0.64** |
| BERT         |     0.93 |     0.63 |     0.85 | **0.97** |
| BERT + CRF   | **0.94** | **0.68** | **0.86** | **0.97** |

CRF는 BiLSTM에서 모든 entity type의 성능을 개선했다.

특히:

```text
MISC: 0.38 → 0.55
ORG : 0.30 → 0.42
PER : 0.53 → 0.64
```

BERT에서도 CRF가 `LOC`, `MISC`, `ORG`를 소폭 개선했지만 이미 BERT 자체 성능이 높아서 차이는 작았다.

특히 `MISC`가 모든 모델에서 가장 어려운 category였다.

---

# 7. Training Behavior

Validation F1:

| Epoch | BiLSTM | BiLSTM+CRF |   BERT |   BERT+CRF |
| ----: | -----: | ---------: | -----: | ---------: |
|     1 | 0.4367 |     0.4816 | 0.8348 | **0.9058** |
|     2 | 0.6148 |     0.6522 | 0.9302 | **0.9348** |
|     3 | 0.6721 |     0.7205 | 0.9373 | **0.9475** |

BERT 계열은 **첫 epoch부터 BiLSTM 계열의 3 epoch 결과보다 높은 성능**을 보였다.

이는 pretrained contextual representation의 효과가 매우 크다는 것을 보여준다.

---

# 8. CRF Cost

CRF의 parameter overhead 자체는 거의 없다.

```text
BiLSTM
1,680,905

BiLSTM + CRF
1,681,004
→ +99 parameters


BERT
108,317,193

BERT + CRF
108,317,292
→ +99 parameters
```

반면 training time은 크게 증가했다.

```text
BiLSTM
~0.6 sec / epoch

BiLSTM + CRF
~5.2 sec / epoch


BERT
~7.3 sec / epoch

BERT + CRF
~14–16 sec / epoch
```

즉 CRF는 **parameter cost는 거의 없지만 sequential dynamic programming 때문에 computational cost가 증가**한다.

BERT에서는:

```text
F1
0.9106 → 0.9179

Training time
~7.3s → ~15s
```

이므로 약 2배의 학습 시간에 비해 성능 향상은 +0.7 F1 point 정도였다.

---

# 9. Key Findings

1. **BERT가 BiLSTM을 크게 앞섰다.**
   F1 `0.5336 → 0.9106`으로 pretrained contextual representation의 효과가 가장 컸다.

2. **CRF는 BiLSTM에서 매우 효과적이었다.**
   F1 `0.5336 → 0.6190`으로 +8.5 point 개선됐다.

3. **BERT에서도 CRF가 최고 성능을 기록했지만 개선폭은 작았다.**
   `0.9106 → 0.9179`, 약 +0.7 point였다.

4. **CRF의 parameter overhead는 사실상 무시할 수준이다.**
   두 모델 모두 +99 parameters뿐이었다.

5. **하지만 CRF의 computational overhead는 상당했다.**
   특히 BERT에서는 epoch 시간이 약 2배 증가했다.

6. 따라서 이번 실험에서는 **BERT+CRF가 최고 정확도**, **BERT가 성능/속도 trade-off 측면에서 가장 효율적인 모델**이라고 볼 수 있다.

### Conclusion

```text
Representation improvement
BiLSTM ──────────────────────────────→ BERT
0.5336                                0.9106
               +0.3770 F1

Sequence modeling
BiLSTM ──CRF──→ BiLSTM+CRF
0.5336          0.6190
                 +0.0854

BERT ──CRF──→ BERT+CRF
0.9106        0.9179
               +0.0073
```

이번 결과에서 가장 큰 성능 향상은 **CRF가 아니라 pretrained Transformer representation으로의 전환**에서 발생했다. CRF는 여전히 sequence consistency를 개선하지만, BERT처럼 강한 contextual encoder 위에서는 marginal gain이 상대적으로 작아졌다.
