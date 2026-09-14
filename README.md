# SENTINEL — AI Scam & Phishing Intelligence

SENTINEL is a defensive research and engineering prototype for analyzing suspicious messages and URLs. It combines a supervised DistilBERT text classifier, transparent social-engineering indicators, and a separate URL-risk model behind a FastAPI service with a Next.js frontend.

> **Research prototype · Defensive use**
>
> SENTINEL provides probabilistic risk analysis. A low-risk result is not proof that a message or URL is safe, and a high-risk result is not a substitute for human verification.

## Final Results at a Glance

### Final Text Model — DistilBERT Model D

| Evaluation               |   Accuracy |   Precision |     Recall |         F1 |    ROC-AUC |
| ------------------------ | ---------: | ----------: | ---------: | ---------: | ---------: |
| **Held-out test set**    | **98.54%** |  **98.42%** | **95.71%** | **97.05%** | **99.86%** |
| **OOD v3 fresh holdout** | **87.00%** | **100.00%** | **74.00%** | **85.06%** | **98.05%** |

**OOD v3:** 200 benign + 200 spam messages, **0 false positives**, **52 false negatives**.

**External phishing-positive evaluation:** DistilBERT phishing recall **83.83%** on Nazario.

**Robustness:** 4 of 42 unique base messages experienced at least one malicious→benign threshold flip = **9.52%**, bootstrap 95% CI **2.38%–19.05%**.

### URL Model

Primary PhiUSIIL domain-aware benchmark:

**99.63% accuracy · 99.95% precision · 99.20% recall · 99.58% F1 · 99.86% ROC-AUC**

A separate Faizann24-derived URL branch achieved **78.42% accuracy · 35.36% precision · 62.64% recall · 45.20% F1 · 83.74% ROC-AUC**. These are separate datasets and are not merged into one headline URL score.

---

## What SENTINEL Does

Given a message, SENTINEL:

1. estimates scam/phishing risk with the deployed DistilBERT classifier;
2. extracts transparent auxiliary manipulation signals;
3. extracts URLs from the message;
4. evaluates each detected URL with a separate URL model;
5. returns a combined evidence-oriented assessment to the frontend.

Current auxiliary signals:

* **Urgency** — pressure or deadline language.
* **Financial** — payment or financial-action language.
* **Authority** — organizational/security authority or impersonation cues.

These indicators are transparent heuristics. They are **supporting evidence, not neural-model explanations**.

## Architecture

```text
                         ┌──────────────────────┐
                         │   SENTINEL Frontend  │
                         │ Next.js + TypeScript  │
                         └──────────┬───────────┘
                                    │ POST /analyze
                                    ▼
                         ┌──────────────────────┐
                         │    FastAPI Backend   │
                         └──────────┬───────────┘
                                    │
                     ┌──────────────┴──────────────┐
                     ▼                             ▼
            ┌─────────────────┐          ┌─────────────────┐
            │  Text Pipeline  │          │   URL Pipeline  │
            │ DistilBERT D    │          │ URL extraction  │
            │ + rule signals  │          │ + URL classifier│
            └────────┬────────┘          └────────┬────────┘
                     │                            │
                     └─────────────┬──────────────┘
                                   ▼
                         ┌──────────────────────┐
                         │ Risk + Evidence Layer│
                         └──────────┬───────────┘
                                    ▼
                         ┌──────────────────────┐
                         │    SENTINEL UI       │
                         │ score / risk / URL / │
                         │ signals / evidence   │
                         └──────────────────────┘
```

### Technology

* **Frontend:** Next.js 16.3.4, React, TypeScript, Tailwind.
* **Backend:** FastAPI + Uvicorn.
* **ML runtime:** PyTorch 2.14.0.
* **Development accelerator:** Apple MPS.
* **Text model:** DistilBERT checkpoint `results/models/distilbert_short_benign`.
* **URL model:** logistic-regression pipeline with stored preprocessing artifacts.

---

## Text Dataset Construction

The main text corpus combines three sources:

| Source       |      Rows | Label 0 | Label 1 |
| ------------ | --------: | ------: | ------: |
| Nazario      |     1,565 |       0 |   1,565 |
| Ling         |     2,859 |   2,401 |     458 |
| SpamAssassin |     5,809 |   4,091 |   1,718 |
| **Combined** | **8,668** |       — |       — |

After deduplication:

* **8,660** normalized unique rows.
* **0** normalized overlap across sources.
* Source-aware split:

  * train **6,062**
  * validation **1,299**
  * test **1,299**
* Training distribution: **4,543 benign / 1,519 malicious**.

Dataset preparation includes missing-value checks, exact and normalized duplicate analysis, cross-source overlap checks, and source-aware splitting.

---

## Text Model Development

The final model was not selected from a single benchmark. The project compares a classical baseline and several DistilBERT variants:

| Model                                    |   Accuracy |  Precision |     Recall |         F1 |    ROC-AUC |
| ---------------------------------------- | ---------: | ---------: | ---------: | ---------: | ---------: |
| Majority baseline                        |     74.90% |          — |          — |          — |          — |
| TF-IDF + Logistic Regression             |     98.38% |     98.41% |     95.09% |     96.72% |     99.74% |
| **Original DistilBERT**                  | **98.54%** | **98.42%** | **95.71%** | **97.05%** | **99.92%** |
| DistilBERT + 1,500 benign                |     98.92% |     97.27% |     98.47% |     97.87% |     99.91% |
| DistilBERT + benign + targeted malicious |     98.77% |     97.84% |     97.24% |     97.54% |     99.77% |
| **Model D: + short-benign**              |     98.54% |     98.42% |     95.71% |     97.05% |     99.86% |

The **98.54% held-out accuracy** is the primary in-domain accuracy of the final deployed model. The **87.00% OOD v3 accuracy** is a separate post-development generalization result and should not replace or be conflated with the held-out score.

### Why Model D Was Selected

Model D added:

* 1,500 benign augmentation examples;
* 1,500 targeted malicious augmentation examples;
* 1,000 short-benign augmentation examples.

Its epoch-3 validation results were:

* Accuracy **99.23%**
* Precision **99.07%**
* Recall **97.85%**
* F1 **98.45%**
* ROC-AUC **99.96%**

It was selected because it substantially improved the development/OOD profile, particularly benign false-positive control, while retaining strong original held-out performance.

---

## Confusion Matrices

### Original Held-Out Text Test — Model D

```text
[[968, 5],
 [14, 312]]
```

### OOD v2 — Model D

```text
[[120, 0],
 [14, 106]]
```

### OOD v3 — Model D

```text
[[200, 0],
 [52, 148]]
```

OOD v3 therefore contains **200/200 correctly classified benign messages**, **148/200 correctly classified spam messages**, **0 false positives**, and **52 false negatives**.

---

## OOD v2 — Development Challenge

OOD v2 contains **240 messages**: 120 benign and 120 malicious, with balanced domain/register coverage.

| Model                 |   Accuracy |   Precision |     Recall |         F1 |    ROC-AUC |
| --------------------- | ---------: | ----------: | ---------: | ---------: | ---------: |
| Original DistilBERT   |     62.08% |      57.14% |     96.67% |     71.83% |     90.02% |
| + benign augmentation |     68.75% |      97.87% |     38.33% |     55.09% |     88.54% |
| 50/50 score fusion    |     80.42% |      93.98% |     65.00% |     76.85% |     90.43% |
| + targeted malicious  |     82.92% |      76.87% |     94.17% |     84.64% |     94.48% |
| **Model D**           | **94.17%** | **100.00%** | **88.33%** | **93.81%** | **99.65%** |

**Important:** OOD v2 was used to diagnose failure modes and guide augmentation, so it is treated as a **development challenge set**, not an unbiased final generalization estimate.

---

## OOD v3 — Fresh Post-Development Holdout

OOD v3 was constructed **after Model D development** from an unused portion of the UCI SMS pool after removing overlap with the augmentation data.

* 400 total messages.
* 200 ham / 200 spam.
* Random seed: **2026**.
* Dataset SHA-256: `5ab2bcf9b6acc6c73d507ecc56d76f9dfd3e0d2f2e972fc97d486660319f81c8`.
* Prediction file SHA-256: `bb299f93e70f6f9e7f12d201a63e7e1670af60da498362638af4ab80742a54a6`.

| Model               |   Accuracy |   Precision | Recall |         F1 |    ROC-AUC |    FP |     FN |
| ------------------- | ---------: | ----------: | -----: | ---------: | ---------: | ----: | -----: |
| Original DistilBERT |     68.75% |      63.16% | 90.00% |     74.23% |     86.66% |   105 |     20 |
| **Model D**         | **87.00%** | **100.00%** | 74.00% | **85.06%** | **98.05%** | **0** | **52** |

OOD v3 is the project's **primary post-development generalization estimate**, but it is specifically an **external SMS-spam generalization test**, not a modern phishing benchmark.

### OOD v3 Calibration

* PR-AUC: **98.65%**
* Brier score: **0.12385**
* ECE: **0.12860**

The raw model scores are therefore **not guaranteed to be calibrated probabilities**. A UI score such as `99.9%` should be interpreted as a model risk/confidence score, not as a literal 99.9% real-world probability.

---

## Cross-Source Text Evaluation

| Train source → Test source | Accuracy |     F1 | ROC-AUC |
| -------------------------- | -------: | -----: | ------: |
| Ling → SpamAssassin        |   43.56% | 49.95% |  85.70% |
| SpamAssassin → Ling        |   91.84% | 79.29% |  99.33% |

These results demonstrate that strong in-domain performance does not imply symmetric cross-corpus generalization.

---

## External Phishing Evaluation

Nazario is reserved as a phishing-positive-only external evaluation set.

| Model                        | Phishing recall |
| ---------------------------- | --------------: |
| TF-IDF + Logistic Regression |          80.89% |
| **DistilBERT**               |      **83.83%** |

Because the set is positive-only, ROC-AUC is not reported.

---

## Robustness Study

The robustness experiment tests whether malicious messages remain detectable after controlled linguistic rewrites that preserve the underlying intent.

### Transformation Families

1. Professional rewrite — improve grammar, structure, and naturalness.
2. Cue-reduced rewrite — reduce obvious scam-style cues while preserving objective and requested action.
3. Combined rewrite — apply both transformations.

### Screening Pipeline

```text
Original malicious message
          ↓
Controlled LLM transformation
          ↓
Placeholder validation
          ↓
Cosine similarity screening
          ↓
Manual semantic audit
          ↓
Model evaluation
          ↓
Before / after comparison
```

Cosine similarity is a screening signal rather than proof of semantic equivalence. The predefined screening threshold is **0.85**.

### Final Audit

* 100 malicious base messages sampled.
* 300 transformation attempts.
* 281 passed generation validation.
* 226 passed similarity screening.
* 60 candidates manually audited.
* 59 semantically verified transformation instances.
* 1 not assessed.

### Primary Analysis — Base-Message Level

Because multiple rewrites originate from the same source message, the **42 unique base messages** are the primary statistical unit.

* 4/42 base messages had ≥1 malicious→benign threshold flip.
* **Flip rate: 9.52%**
* Bootstrap 95% CI: **2.38%–19.05%**
* Original mean score: **0.98328**
* Transformed mean score: **0.92947**
* Mean delta: **−0.05381**
* Median delta: **−0.000098**
* One-sided Wilcoxon: **W = 766, p = 1.68×10⁻⁵**
* Bootstrap 95% CI for mean delta: **[−0.12047, −0.00185]**

### Secondary Transformation-Level Analysis

| Transformation | n | Mean Δ | Mean |Δ| | Wilcoxon p |
|---|---:|---:|---:|---:|
| Professional | 20 | -0.0656 | 0.1076 | 0.0136 |
| Cue-reduced | 20 | -0.0592 | 0.0729 | 0.0136 |
| Combined | 19 | -0.1579 | 0.1769 | 0.0053 |
| Overall | 59 | -0.0932 | 0.1182 | 0.000021 |

All 8 transformation-instance threshold flips were malicious→benign and came from 4 underlying base messages.

**Interpretation:** the classifier is measurably sensitive to semantically preserved linguistic rewriting. This is a controlled robustness challenge-set result, not a prevalence estimate.

---

## URL Intelligence

### Primary PhiUSIIL Branch

* Raw rows: **235,795**
* Test rows: **23,677**

| Metric    |     Result |
| --------- | ---------: |
| Accuracy  | **99.63%** |
| Precision | **99.95%** |
| Recall    | **99.20%** |
| F1        | **99.58%** |
| ROC-AUC   | **99.86%** |

### Separate Faizann24-Derived Branch

Provenance recorded in the project as:

`faizann24/Using-machine-learning-to-detect-malicious-URLs`

Cleaning audit:

* Raw rows: **420,464**
* Exact duplicate rows: **9,216**
* Duplicate URLs: **9,217**
* Normalized unique URLs: **406,955**
* Normalized duplicates: **13,509**
* One conflicting URL removed
* Final clean rows: **406,954**
* Good: **344,799**
* Bad: **62,155**
* Bad proportion: **15.27%**
* Test rows: **41,904**

| Metric    |     Result |
| --------- | ---------: |
| Accuracy  | **78.42%** |
| Precision | **35.36%** |
| Recall    | **62.64%** |
| F1        | **45.20%** |
| ROC-AUC   | **83.74%** |

The two URL branches are deliberately reported separately because they represent different datasets and evaluation conditions.

### Live URL Deployment Verification

Balanced sanity sample:

* 100 URLs.
* 50 good / 50 bad.
* Random seed **42**.

Results:

* Accuracy **72.00%**
* Precision **76.19%**
* Recall **64.00%**
* F1 **69.57%**
* False positives **10/50 good**
* False negatives **18/50 bad**

The live and offline scorers produced a maximum absolute probability difference of approximately **1.11×10⁻¹⁶**, verifying numerical consistency of the deployment path.

---

## Error Analysis and Known Failure Modes

### Short, Context-Poor Scam Messages

Final UI smoke testing revealed a specific limitation after short-benign augmentation. Several short scam messages were under-scored:

* `WINNER! Claim your free prize now. Reply YES to receive your reward.` → **1.6%**
* `Congratulations! You have won a prize. Claim it now.` → **0.0%**
* `You won $500! Reply YES to claim your prize.` → **3.2%**
* `URGENT: You have won a cash prize. Claim now.` → **0.0%**
* `Your account is locked. Verify now.` → **0.0%**
* `I love you. Send me money for my emergency.` → **0.0%**

In contrast, longer romance and prize/financial examples were scored at **100.0% high risk** during final application testing.

This suggests a meaningful **short-context blind spot** rather than a universal failure of the classifier.

### Other Limitations

* Dataset-specific language and source distributions can create corpus shortcuts.
* Cross-source performance can be highly asymmetric.
* OOD v3 shows **98.05% ROC-AUC but only 74.00% recall**, illustrating a ranking/classification trade-off.
* Raw risk scores are not universally calibrated.
* OOD v3 false negatives should not all be interpreted as missed phishing attacks because it is an SMS-spam holdout and includes out-of-scope content.
* Robustness transformations are controlled rewrites, not real attacker adaptations.
* Auxiliary signals are heuristics, not faithful neural-model explanations.
* URL analysis is lexical/feature-based and does not perform live reputation lookup, DNS resolution, or browser execution.

---

## Scam Taxonomy: Deliberately Not Used for Headline Claims

The repository contains a **60-row synthetic intent→category prototype** spanning ten categories, including family/friend impersonation, payment/invoice fraud, phishing/account takeover, romance scam, prize/refund/charity scams, and investment/crypto scam.

It achieved very high experimental classification scores, but it is too small and synthetic to support credible real-world category probabilities. Therefore:

> **SENTINEL does not currently claim real-world “Romance 87% / Financial 92% / Prize 14%” style category probabilities.**

A future multi-class taxonomy should be built only after obtaining sufficiently diverse, appropriately licensed, independently labeled scam-category data.

---

## Why This Evaluation Goes Beyond Accuracy

A high held-out score can hide dataset-specific shortcuts. SENTINEL therefore evaluates:

* in-domain held-out performance;
* cross-source generalization;
* external phishing-positive evaluation;
* development OOD challenge data;
* fresh post-development OOD data;
* domain-aware URL splits;
* calibration;
* error distributions;
* controlled linguistic robustness;
* deployment-path consistency.

The goal is not only to maximize a benchmark score, but to identify where that performance stops generalizing.

---

## Reproducibility

The current Python environment is pinned in `requirements.txt`:

```text
accelerate==1.14.0
datasets==5.0.1
joblib==1.6.0
numpy==2.4.6
openai==3.7.0
pandas==3.0.5
python-dotenv==1.2.3
scikit-learn==1.9.0
scipy==1.17.1
sentence-transformers==6.0.1
torch==2.14.0
transformers==5.16.1
matplotlib==3.10.8
```

Environment check:

```text
python -m pip check
No broken requirements found.
```

The OOD v3 evaluation scripts were rerun and reproduced the reported Model D metrics.

Large datasets, model weights, local environments, secrets, and row-level experimental outputs are excluded from version control.

---

## Quick Start

### Backend

From the repository root:

```bash
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Open:

`http://localhost:3000`

### Production Build

```bash
cd frontend
npm run build
```

---

## API

### `POST /analyze`

Request:

```json
{
  "message": "URGENT: Verify your bank account immediately."
}
```

The response includes:

* `risk_score`
* `risk_level`
* `model`
* `device`
* `signals`
* `combinations`
* `url_count`
* `url_max_probability`
* `url_mean_probability`
* `url_risk_level`

---

## Repository Structure

```text
ai-scam-phishing-intelligence/
├── README.md
├── requirements.txt
├── backend/
│   └── app/
│       ├── main.py
│       └── services/
│           ├── text_model.py
│           └── url_model.py
├── frontend/
│   ├── app/
│   ├── components/
│   └── lib/
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   └── evaluation/
├── results/
│   ├── dataset_investigation/
│   ├── phiusiil/
│   └── experiment_log.md
├── compare_ood_v3.py
└── evaluate_ood_v3.py
```

---

## Finalization Status

* Final text model selected.
* Held-out text evaluation.
* Cross-source evaluation.
* External phishing-positive evaluation.
* OOD v2 development challenge.
* OOD v3 fresh post-development holdout.
* Base-message-level robustness analysis.
* URL benchmark evaluation.
* Live URL deployment verification.
* FastAPI backend integration.
* Next.js frontend.
* Local backend smoke tests.
* Frontend production build.
* Reproducible Python dependency environment.
* Technical report.

---

## Future Work

Future improvements should be evaluated as new experimental versions rather than silently changing the frozen Model D results.

Potential directions:

* improve short-context scam detection without increasing benign false positives;
* obtain better-labeled multi-class scam data for category analysis;
* add calibrated risk scoring using a held-out calibration set;
* add richer URL intelligence without unsafe live execution;
* evaluate additional modern phishing corpora;
* deploy a public demonstration.

---

## License

Project code license: **to be selected before public publication**.

Dataset and pretrained-model terms remain governed by their respective sources and licenses.

## Author

**Simran**
