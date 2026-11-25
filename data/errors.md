# Error Analysis Report (Development Set)

> **Note:** This report inspects the **DEV set ONLY** (fixed 70/30 split, seed=42).
> The **TEST set** is held out and was NOT accessed during this analysis.

- **Model:** `v1-model`
- **Prompt Version:** `v1`
- **Timestamp:** 2026-11-18 11:55:06 UTC
- **Dev Set Items Analyzed:** 28
- **Total Disagreements:** 4
- **Error Rate on Dev Set:** 14.3%

## Summary of Disagreements

| Transition (My Label → Model Label) | Error Count | Sources Affected |
| :--- | :---: | :--- |
| `negative -> neutral` | 3 | `reddit` (2), `youtube` (1) |
| `neutral -> positive` | 1 | `news` (1) |

---

## Detailed Disagreements by Group

### Transition: `negative -> neutral`

#### Source: `reddit` (2 items)

**Item ID 9** (CA)
- **Title:** Nike app keeps crashing during checkouts
- **Text:** "Terrible shopping experience trying to buy the new trail running shoes."
- **My Label (Ground Truth):** `negative` (Relevant: `True`)
- **Model Label (Prediction):** `neutral` (Relevant: `True`)
- **Model Reason:** *"Retail app issue during checkout without critique of shoe design."*

**Item ID 6** (US)
- **Title:** Nike Pegasus sole detached after 3 weeks
- **Text:** "Extremely poor durability for a 150 dollar shoe. Very disappointed with build quality."
- **My Label (Ground Truth):** `negative` (Relevant: `True`)
- **Model Label (Prediction):** `neutral` (Relevant: `True`)
- **Model Reason:** *"Technical report of premature sole separation on Pegasus."*

#### Source: `youtube` (1 items)

**Item ID 30** (GB)
- **Title:** Nike Structure 25: too stiff and clunky
- **Text:** "Lacks cushioning and feels heavy compared to competing support shoes."
- **My Label (Ground Truth):** `negative` (Relevant: `True`)
- **Model Label (Prediction):** `neutral` (Relevant: `True`)
- **Model Reason:** *"Technical comparison noting firmness and weight differences."*

### Transition: `neutral -> positive`

#### Source: `news` (1 items)

**Item ID 32** (US)
- **Title:** Nike patent reveals new lacing mechanism
- **Text:** "Patent filing shows automated cable tensioning concept."
- **My Label (Ground Truth):** `neutral` (Relevant: `True`)
- **Model Label (Prediction):** `positive` (Relevant: `True`)
- **Model Reason:** *"Innovative tensioning mechanism highlights technical advancement."*
