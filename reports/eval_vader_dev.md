# Model Evaluation Report: vader

- **Dataset Split:** `DEV` (Fixed 70/30 split, seed=42)
- **Timestamp:** 2026-11-14 17:43:21 UTC
- **Total Sample Size:** 28
- **Sentiment Sample Size:** 23

## Overall Metrics

| Metric | Score | Target Threshold | Status |
| :--- | :---: | :---: | :---: |
| **Sentiment Accuracy** | **1.0000** | $\ge 0.85$ | PASS |
| **Macro-F1 Score** | **1.0000** | $\ge 0.80$ | PASS |
| **Relevance Accuracy** | **0.8214** | — | INFO |

## Per-Class Performance (scikit-learn)

| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Positive** | 1.0000 | 1.0000 | 1.0000 | 9.0 |
| **Negative** | 1.0000 | 1.0000 | 1.0000 | 5.0 |
| **Neutral** | 1.0000 | 1.0000 | 1.0000 | 9.0 |

## Confusion Matrix

| Actual \ Predicted | Predicted Positive | Predicted Negative | Predicted Neutral |
| :--- | :---: | :---: | :---: |
| **Actual Positive** | 9 | 0 | 0 |
| **Actual Negative** | 0 | 5 | 0 |
| **Actual Neutral**  | 0 | 0 | 9 |

## Slice Metrics by Source

| Source | Samples | Sentiment Accuracy | Macro-F1 | Relevance Accuracy |
| :--- | :---: | :---: | :---: | :---: |
| `news` | 16 | 1.0000 | 1.0000 | 0.7500 |
| `reddit` | 8 | 1.0000 | 1.0000 | 0.8750 |
| `youtube` | 4 | 1.0000 | 1.0000 | 1.0000 |

## Slice Metrics by Country

| Country | Samples | Sentiment Accuracy | Macro-F1 | Relevance Accuracy |
| :--- | :---: | :---: | :---: | :---: |
| `AU` | 1 | 1.0000 | 1.0000 | 1.0000 |
| `CA` | 3 | 1.0000 | 1.0000 | 1.0000 |
| `GB` | 7 | 1.0000 | 1.0000 | 0.8571 |
| `IN` | 6 | 1.0000 | 1.0000 | 0.6667 |
| `US` | 11 | 1.0000 | 1.0000 | 0.8182 |
