# AI Scam & Phishing Intelligence Assistant

An AI-assisted defensive system for detecting phishing and scam messages, analyzing suspicious URLs, and explaining the main risk indicators behind a prediction.

> Project status: Core text/URL modeling, OOD evaluation, robustness analysis, URL intelligence, FastAPI backend, and Next.js frontend are established. Final documentation, reproducibility packaging, and technical report preparation remain.

## Overview

Phishing and scam messages increasingly rely on social engineering rather than obviously malicious wording. This project investigates a layered approach for detecting suspicious messages, analyzing URLs, and providing interpretable supporting risk indicators.

### Research Questions

1. How well do classical and transformer-based models detect phishing/scam messages on held-out data?
2. How well do those models generalize across different email corpora and external phishing data?
3. How robust is detection when malicious messages are rewritten to sound more professional or less obviously suspicious while preserving their malicious intent?

## System Architecture

```text
Message + Optional URL
          |
          v
   +----------------------+
   | Text Analysis        |
   | TF-IDF / DistilBERT  |
   +----------------------+
          |
          +--------------------+
          |                    |
          v                    v
  Manipulation Signals     URL Analysis
  - urgency                - URL structure
  - financial requests     - suspicious characters
  - authority cues         - subdomains
                         - encoded characters
                         - IP/port indicators
          |                    |
          +---------+----------+
                    |
                    v
          Risk / Evidence Layer
                    |
                    v
             User Explanation
```

## Models

### Text Classification

- TF-IDF + Logistic Regression baseline
- DistilBERT fine-tuned for binary text classification

### URL Analysis

The URL layer uses reproducible lexical features such as URL length, domain length, character composition, subdomain count, query characters, special-character ratios, HTTPS, and IP-host indicators. No live DNS requests, webpage downloads, or browser execution are required by the current URL analysis layer.

## Current Results

### Text Classification: Held-Out Test Set

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Majority baseline | 74.90% | — | — | — | — |
| TF-IDF + Logistic Regression | 98.38% | 98.41% | 95.09% | 96.72% | 99.74% |
| DistilBERT | **98.54%** | **98.42%** | **95.71%** | **97.05%** | **99.92%** |
| DistilBERT + 1,500 benign augmentation | 98.92% | 97.27% | **98.47%** | **97.87%** | 99.91% |
| DistilBERT + benign + targeted malicious augmentation | 98.77% | 97.84% | 97.24% | 97.54% | 99.77% |
| DistilBERT + benign + targeted malicious + short-benign augmentation | 98.54% | 98.42% | 95.71% | 97.05% | 99.86% |

These are in-domain held-out results and are not presented as estimates of real-world phishing detection performance.

### OOD v2 Development Challenge Set

A 240-message development challenge set was constructed to test domain and register shift across benign and malicious business/personal messages in casual and formal registers. The threshold was fixed at 0.50 for the reported model evaluations.

Because OOD v2 was used to diagnose failure modes and guide the successive augmentation experiments, its Model D result is treated as development-set performance rather than an unbiased generalization estimate. A fresh OOD v3 holdout is used for the post-development generalization claim.

| Model | Accuracy | Precision | Malicious Recall | F1 | ROC-AUC | Benign FP | Malicious FN |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original DistilBERT | 62.08% | 57.14% | **96.67%** | 71.83% | 90.02% | 87/120 | 4/120 |
| + benign augmentation | 68.75% | **97.87%** | 38.33% | 55.09% | 88.54% | 1/120 | 74/120 |
| 50/50 score fusion | 80.42% | 93.98% | 65.00% | 76.85% | 90.43% | 5/120 | 42/120 |
| + targeted malicious augmentation | 82.92% | 76.87% | **94.17%** | 84.64% | 94.48% | 34/120 | 7/120 |
| **+ short-benign augmentation (Model D)** | **94.17%** | **100.00%** | 88.33% | **93.81%** | **99.65%** | **0/120** | 14/120 |

Model D was selected as the current final text model because it substantially reduced benign false positives while retaining high malicious recall on the frozen challenge set. The remaining 14 malicious false negatives are retained as a documented failure-analysis set rather than used for further tuning.

### OOD v3 Fresh Post-Development Holdout

OOD v3 is a fresh 400-message external holdout constructed after Model D development. It contains 200 ham and 200 spam messages from UCI SMS data, excluding messages reused during Model D benign augmentation. The dataset was frozen before evaluation and was not used for further model tuning.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | False Positive | False Negative |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original DistilBERT | 68.75% | 63.16% | 90.00% | 74.23% | 86.66% | 105/200 | 20/200 |
| **Model D** | **87.00%** | **100.00%** | 74.00% | **85.06%** | **98.05%** | **0/200** | 52/200 |

Model D changed 139 of 400 predictions relative to the original checkpoint. On this fresh holdout, it eliminated benign false positives but traded away malicious recall. OOD v3 is treated as the primary post-development generalization estimate; OOD v2 is retained as a development-set diagnostic result.

### Cross-Source Text Evaluation

| Train Source | Evaluation Source | Model | Accuracy | F1 | ROC-AUC |
|---|---|---|---:|---:|---:|
| Ling | SpamAssassin | DistilBERT | 43.56% | 49.95% | 85.70% |
| SpamAssassin | Ling | DistilBERT | 91.84% | 79.29% | 99.33% |

The cross-source results show that strong in-domain performance does not guarantee generalization across corpora.

### External Phishing Evaluation

Nazario is used as an external phishing-positive-only evaluation set.

| Model | Phishing Recall |
|---|---:|
| TF-IDF + Logistic Regression | 80.89% |
| DistilBERT | **83.83%** |

Because the external set is positive-only, ROC-AUC is not reported for this evaluation.

## URL Benchmark

Two complementary URL resources are used.

The original malicious-URL dataset is retained as a secondary reference for studying harder generalization conditions and domain leakage.

PhiUSIIL is used as the primary URL benchmark with a controlled URL-only feature subset and domain-aware train/validation/test splits.

### Domain-Aware PhiUSIIL Test Results

| Metric | Result |
|---|---:|
| Accuracy | 99.63% |
| Precision | 99.95% |
| Recall | 99.20% |
| F1 | 99.58% |
| ROC-AUC | 99.86% |

These results are dataset-specific benchmark results and should not be interpreted as production-level phishing detection performance.

### Live URL Deployment Verification

A separate balanced 100-URL sample (50 good, 50 bad; random_state=42) was passed through the same feature extraction, scaler, and logistic-regression artifacts used by the deployed URL scorer.

| Metric | Live 100-URL Result |
|---|---:|
| Accuracy | 72.00% |
| Precision | 76.19% |
| Recall | 64.00% |
| F1 | 69.57% |
| False positives | 10/50 good |
| False negatives | 18/50 bad |

The live scorer and offline raw-URL scorer produced numerically identical probabilities on all 100 URLs (maximum absolute difference approximately 1.11e-16). This verifies that the deployed URL inference path matches the offline implementation. Because this is a small balanced deployment check, it is reported separately from the full 41,904-row domain-aware benchmark.

## Why the Evaluation Goes Beyond Accuracy

A high held-out score can hide dataset-specific shortcuts. This project therefore evaluates:

- in-domain held-out performance
- cross-source generalization
- external phishing data
- domain-aware URL splits
- error distributions
- auxiliary manipulation indicators
- robustness to controlled linguistic transformations

The goal is to measure not only whether the model performs well, but also where that performance stops generalizing.

## Robustness Study

The project includes a controlled robustness experiment to test whether phishing detection remains effective after malicious messages are rewritten while preserving their underlying intent.

A fixed sample of 100 malicious messages is transformed in three ways:

1. Professional rewrite: improve grammar, readability, structure, and naturalness while preserving suspicious intent.
2. Cue-reduced rewrite: reduce obvious scam-style linguistic cues such as exaggerated urgency, awkward phrasing, excessive punctuation, or overt threats while preserving the underlying objective and requested action.
3. Combined rewrite: apply both transformations.

This produces 300 transformation attempts. Of these, 281 passed generation validation, 226 passed cosine-similarity screening, and 60 candidates were selected for manual semantic audit. One audited case was not assessable because the underlying message was materially corrupted, leaving 59 semantically verified transformation instances for the primary robustness analysis.

### Robustness Pipeline

```text
Original malicious message
          |
          v
Controlled LLM transformation
          |
          v
Placeholder integrity validation
          |
          v
Semantic similarity screening
          |
          v
Structured semantic preservation audit
          |
          v
Model evaluation
          |
          v
Before/after comparison
```

Cosine similarity is used only as a screening signal, not as proof of semantic equivalence. The predefined screening threshold is 0.85.

Transformations that fail the automatic screening are not rescued by lowering the threshold.

### Robustness Results

On the 59 semantically verified transformations, all 59 original malicious messages were classified as positive at the 0.50 threshold. After rewriting, 51/59 remained positive, producing 8/59 (13.56%) classification flips.

| Transformation | n | Mean Δ score | Mean absolute Δ | Wilcoxon p |
|---|---:|---:|---:|---:|
| Professional | 20 | -0.0656 | 0.1076 | 0.0136 |
| Cue-reduced | 20 | -0.0592 | 0.0729 | 0.0136 |
| Combined | 19 | -0.1579 | 0.1769 | 0.0053 |
| Overall | 59 | -0.0932 | 0.1182 | 0.000021 |

The 8 threshold-crossing flips came from 4 underlying base messages, so they are not 8 independent message-level failures. The combined transformation produced the largest mean confidence reduction.

All 8 threshold-crossing changes were in the same direction (malicious → benign). An exact two-sided McNemar test on the paired classifications gave p = 0.0078125, indicating a statistically significant change in binary predictions. The 8 flips came from 4 underlying base messages, so the transformations are not independent message-level observations.

At higher thresholds, the number of positive detections lost after rewriting was 6 at 0.75, 6 at 0.90, and 4 at 0.99, compared with 8 at the 0.50 threshold.

These results indicate statistically significant sensitivity to semantically preserved linguistic rewriting. This is a controlled robustness challenge-set evaluation, not a real-world prevalence estimate.


## Auxiliary Risk Indicators

The system also extracts transparent rule-based auxiliary context signals from messages, currently including:

- urgency or deadline language
- financial or payment requests
- authority or impersonation cues

These signals provide supporting context for the final explanation. They are not treated as a replacement for the primary text classifier or as model explainability.

## Error Analysis

Evaluation includes explicit analysis of false positives and false negatives rather than relying only on aggregate metrics.

The analysis examines:

- source-specific failures
- difficult phishing examples
- benign promotional or newsletter messages
- URL-related errors
- messages with manipulation indicators
- cross-source generalization failures

## Datasets

### Text

The text corpus combines multiple email datasets used for supervised training and evaluation, including Ling and SpamAssassin, with Nazario reserved for external phishing-positive evaluation.

Dataset preparation includes missing-value checks, exact and normalized duplicate analysis, cross-source overlap analysis, and source-aware splitting.

### URL

Two URL resources are used for complementary purposes:

- Malicious URL dataset: secondary reference for studying generalization conditions and domain leakage.
- PhiUSIIL: primary URL benchmark used with domain-aware splits and a controlled URL-only feature subset.

Raw and processed datasets are intentionally excluded from this repository. The data preparation and splitting scripts remain available for reproducibility.

## Repository Structure

```text
ai-scam-phishing-intelligence/
├── README.md
├── requirements.txt
├── .gitignore
│
├── src/
│   ├── data/           # dataset preparation and splitting
│   ├── features/       # URL and manipulation features
│   ├── models/         # model training scripts
│   └── evaluation/     # evaluation, error analysis, robustness
│
└── results/
    ├── dataset_investigation/
    ├── phiusiil/
    └── experiment_log.md
```

Large datasets, transformer weights, local virtual environments, secrets, and row-level experimental outputs are excluded from version control.

## Reproducibility

The project uses fixed random seeds where applicable and records major experimental decisions in `results/experiment_log.md`.

The evaluation emphasizes held-out performance, source-shift generalization, external evaluation, domain-aware URL splitting, error analysis, and robustness to controlled linguistic transformations.

## Limitations

Current limitations include dataset-specific language and URL distributions, possible corpus shortcuts, the use of controlled synthetic rewrites in the robustness study, and the absence of live threat-intelligence feeds or browser-side execution.

The system is intended as a defensive research and demonstration project, not as a guarantee that a message or URL is safe.

## Planned Final Components

- Complete the robustness experiment and semantic audit
- Produce final benchmark tables and plots
- Build a custom Streamlit interface
- Add architecture and methodology diagrams
- Write the full technical report
- Add tests and reproducibility instructions
- Deploy a public demonstration

## License

License to be selected before final publication.

## Author

**Simran**
