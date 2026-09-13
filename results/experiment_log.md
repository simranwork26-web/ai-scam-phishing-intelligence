# Experiment Log
## AI Scam & Phishing Intelligence Assistant

This file records established experimental decisions and quantitative results.
Major methodology decisions are documented in this log. Exploratory decisions
are explicitly marked where applicable.

---

## 1. Project / Task Definition

Primary task:
- Detect malicious/scam/spam email messages aimed at humans.
- Main supervised dataset combines Ling and SpamAssasin.
- The task is described as broader malicious/spam email classification, not
  pure phishing classification.
- Nazario is retained as an external phishing-focused positive-only evaluation set.

Evidence hierarchy:
1. DistilBERT = primary text classifier.
2. URL lexical model = secondary, noisy evidence.
3. Manipulation signals = Auxiliary Risk Indicators / Rule-Based Context.

URL and manipulation evidence must not be described as part of the DistilBERT
decision process.

---

## 2. Text Dataset Investigation

Source datasets:
- Ling.csv
- SpamAssasin.csv
- Nazario.csv

Files:
- data/raw/text/Ling.csv
- data/raw/text/SpamAssasin.csv
- data/raw/text/Nazario.csv

Dataset sizes:
- Ling: 2,859 rows
  - label 0: 2,401
  - label 1: 458
- SpamAssasin: 5,809 rows
  - label 0: 4,091
  - label 1: 1,718
- Nazario: 1,565 rows
  - all label 1

Combined Ling + SpamAssasin:
- Before deduplication: 8,668 rows
- Label 0: 6,492
- Label 1: 2,176
- Positive rate: approximately 25.1%

After authoritative normalized deduplication:
- 8,660 rows
- Label 0: 6,490
- Label 1: 2,170
- Removed: 8 rows

Cross-dataset normalized text overlap:
- Ling vs SpamAssasin: 0

Final source-aware text split:
- Train: 6,062
- Validation: 1,299
- Test: 1,299

Train:
- SpamAssasin: 4,061
- Ling: 2,001

Validation:
- SpamAssasin: 871
- Ling: 428

Test:
- SpamAssasin: 870
- Ling: 429

Normalized text overlap between train/validation/test:
- 0

---

## 3. Text TF-IDF Baseline

Implementation:
- src/models/train_tfidf_baseline.py

Features:
- lowercase
- strip_accents="unicode"
- word ngrams (1,2)
- min_df=2
- max_df=0.98
- sublinear_tf=True

Classifier:
- LogisticRegression
- max_iter=1000
- class_weight="balanced"
- random_state=42

Validation:
- ROC-AUC: 0.9978
- Accuracy: 0.9831
- Positive precision: 0.9690
- Positive recall: 0.9631
- Positive F1: 0.9660
- Macro F1: 0.9774
- Confusion matrix: [[964, 10], [12, 313]]

Test:
- ROC-AUC: 0.9974
- Accuracy: 0.9838
- Positive precision: 0.9841
- Positive recall: 0.9509
- Positive F1: 0.9672
- Macro F1: 0.9783
- Confusion matrix: [[968, 5], [16, 310]]

Majority-class test baseline:
- Accuracy: 74.90%

---

## 4. Text Error Analysis / Source Shift

TF-IDF test errors:
- False positives: 5
- False negatives: 16
- False positives were all SpamAssasin and mostly commercial/newsletter/promotional.
- False negatives included multilingual, encoded/corrupted, newsletter, and
  semantic/social-engineering examples.

Per-source test:
- Ling accuracy: 0.9953
- Ling positive precision: 1.0000
- Ling positive recall: 0.9710
- Ling positive F1: 0.9853
- Ling AUC: 1.0000

- SpamAssasin accuracy: 0.9782
- SpamAssasin positive precision: 0.9798
- SpamAssasin positive recall: 0.9455
- SpamAssasin positive F1: 0.9624
- SpamAssasin AUC: 0.9961

Cross-source TF-IDF:
- Ling-trained -> SpamAssasin:
  - Accuracy: 0.6989
  - Precision: 0.4946
  - Recall: 0.8988
  - F1: 0.6381
  - AUC: 0.8855

- SpamAssasin-trained -> Ling:
  - Accuracy: 0.9068
  - Precision: 0.6408
  - Recall: 0.9565
  - F1: 0.7674
  - AUC: 0.9794

Interpretation:
- Random-split performance is not sufficient evidence of generalization.
- Strong asymmetric source shift exists.

---

## 5. DistilBERT

Implementation:
- src/models/train_distilbert.py

Model:
- distilbert-base-uncased

Configuration:
- max sequence length: 256
- random seed: 42
- learning rate: 2e-5
- train batch size: 8
- eval batch size: 16
- epochs: 3
- weight decay: 0.01
- best checkpoint selected by validation F1
- Apple MPS

Training runtime:
- approximately 44 minutes 41 seconds

Validation:
- Epoch 1:
  - F1: 0.9753
  - AUC: 0.9989
- Epoch 2:
  - F1: 0.9829
  - AUC: 0.9992
- Epoch 3:
  - F1: 0.9814
  - AUC: 0.9995

Best checkpoint:
- Epoch 2 by validation F1

Clean test:
- Accuracy: 0.9854
- Precision: 0.9842
- Recall: 0.9571
- F1: 0.9705
- ROC-AUC: 0.9992
- Macro F1: 0.9804
- Confusion matrix: [[968, 5], [14, 312]]

DistilBERT vs TF-IDF on clean random/source-aware test:
- DistilBERT accuracy: 0.9854 vs 0.9838
- DistilBERT positive F1: 0.9705 vs 0.9672
- DistilBERT AUC: 0.9992 vs 0.9974

Conclusion:
- DistilBERT is the primary text model.
- Improvement over TF-IDF is real but modest on the controlled test split.

---

## 6. Cross-Source DistilBERT

Implementation:
- src/models/train_distilbert_cross_source.py
- src/evaluation/cross_source_distilbert.py

Ling-trained -> SpamAssasin:
- Accuracy: 0.4356
- Precision: 0.3384
- Recall: 0.9533
- F1: 0.4995
- AUC: 0.8570
- Confusion matrix: [[134, 479], [12, 245]]

SpamAssasin-trained -> Ling:
- Accuracy: 0.9184
- Precision: 0.6700
- Recall: 0.9710
- F1: 0.7929
- AUC: 0.9933
- Confusion matrix: [[327, 33], [2, 67]]

Important finding:
- Ling -> SpamAssasin DistilBERT F1 = 0.4995.
- This is substantially worse than the TF-IDF Ling -> SpamAssasin F1 = 0.6381.
- Source shift is therefore a major limitation and a core finding of the project.

---

## 7. Nazario External Evaluation

Nazario contains only positive/phishing examples.
AUC is therefore not reported.

TF-IDF:
- Predicted class 0: 299
- Predicted class 1: 1,266
- Phishing recall: 0.8089
- Mean probability: 0.642145

DistilBERT:
- Predicted class 0: 253
- Predicted class 1: 1,312
- Phishing recall: 0.8383
- Mean probability: 0.836178

Improvement:
- DistilBERT phishing recall improved by 2.94 percentage points.

Nazario false-negative analysis:
- 253 false negatives
- Saved to:
  results/dataset_investigation/nazario_distilbert_false_negatives.csv
- Many false negatives were ordinary organizational/security notifications.
- At least one explicit login URL received near-zero text probability.

Conclusion:
- Independent URL evidence was investigated as a complementary signal.
- It must not be treated as an automatic fallback classifier.

---

## 8. Auxiliary Manipulation Signals

Implementation:
- src/features/manipulation_signals.py

Frozen signals:
- urgency
- financial
- authority

Examples:
- Urgency: urgent, immediately, ASAP, action required, act now, deadlines.
- Financial: payment, pay, invoice, refund, bank, account number,
  credit card, debit card, wire transfer, transfer money.
- Authority: IT support, helpdesk, administrator/admin, security team,
  support team, HR department, bank, government, police.

Test-set prevalence:
- Urgency:
  - legitimate: 8.84%
  - malicious: 17.18%
- Financial:
  - legitimate: 6.37%
  - malicious: 28.83%
- Authority:
  - legitimate: 7.61%
  - malicious: 12.58%

Conditional malicious rates:
- Urgency present: 39.44%
- Urgency absent: 23.34%
- Financial present: 60.26%
- Financial absent: 20.30%
- Authority present: 35.65%
- Authority absent: 24.07%

Combinations:
- urgency + financial:
  - 47 messages
  - malicious rate: 0.617
- urgency + authority:
  - 25 messages
  - malicious rate: 0.640
- financial + authority:
  - 56 messages
  - malicious rate: 0.5714
- all three:
  - 20 messages
  - malicious rate: 0.650

Signal-count distribution:
- 0 signals: 789 legitimate / 204 malicious
- 1 signal: 156 legitimate / 62 malicious
- 2 signals: 23 legitimate / 42 malicious
- 3 signals: 6 legitimate / 17 malicious

DistilBERT error rate:
- Urgency present: 0.0141
- Urgency absent: 0.0147
- Financial present: 0.0192
- Financial absent: 0.0140
- Authority present: 0.0348
- Authority absent: 0.0127

False negatives:
- 14 total
- 4 had any signal
- 2 had a signal combination

Conclusion:
- Signals are useful contextual evidence but not a replacement classifier.
- Do not describe them as DistilBERT explanations.
- Terminology: "Auxiliary Risk Indicators" / "Rule-Based Context."

---

## 9. URL Dataset Investigation

Source:
- faizann24/Using-machine-learning-to-detect-malicious-URLs

Raw file:
- data/raw/url/malicious_urls.csv

Raw shape:
- 420,464 rows
- 2 columns: url, label

Labels:
- good: 344,821
- bad: 75,643

Duplicate audit:
- Exact duplicate rows: 9,216
- Duplicate URLs: 9,217
- URLs with conflicting labels: 1
- Unique raw URLs: 411,247

Conflicting URL:
- tommyhumphreys.com/
- appeared as both good and bad

Normalization:
- strip whitespace
- lowercase
- remove http/https scheme
- remove fragments
- strip trailing slash

After normalization:
- Unique normalized URLs: 406,955
- Normalized duplicate rows: 13,509
- Normalized conflicts: 1

Conflict was removed entirely.
Normalized duplicates were deduplicated.

Clean URL dataset:
- 406,954 rows
- good: 344,799
- bad: 62,155
- bad proportion: 15.27%

Processed file:
- data/processed/url_dataset.csv

---

## 10. URL Lexical Features

Implementation:
- src/features/url_features.py

22 lexical features:
- url_length
- hostname_length
- path_length
- query_length
- dot_count
- hyphen_count
- underscore_count
- slash_count
- question_mark_count
- equals_count
- ampersand_count
- percent_count
- at_count
- digit_count
- special_char_count
- subdomain_count
- has_ip_hostname
- has_port
- has_query
- has_fragment
- has_encoded_chars
- suspicious_token_count

Suspicious tokens include:
- login
- signin
- sign-in
- verify
- verification
- secure
- account
- update
- confirm
- password
- credential
- bank
- payment
- billing
- invoice
- admin
- webmail
- auth
- unlock

No live URL requests, DNS resolution, crawling, or browsing are used.

---

## 11. URL Leakage Audit

Initial random stratified split:
- Train: 284,867
- Validation: 61,043
- Test: 61,044

Random split hostname overlap:
- Train/Validation: 13,189
- Train/Test: 13,158
- Validation/Test: 6,713

Full clean dataset:
- Unique hostnames: 143,432
- Malformed/unparseable URLs: 759
- Hostnames with both labels: 38
- URLs from mixed-label hostnames: 3,446

Examples:
- twitter.com: 203 bad / 1,055 good
- itunes.apple.com: 1 bad / 477 good
- freepages...: 1 bad / 712 good
- github.com: 1 bad / 1 good
- pastebin.com: 48 bad / 9 good

Conclusion:
- Random URL splitting is not adequate for honest generalization testing.

---

## 12. Domain-Aware URL Split

Implementation:
- src/data/split_url_domain_aware.py

Method:
- GroupShuffleSplit by hostname.
- No non-empty hostname crosses train/validation/test.

Final domain-aware split:
- Train: 328,717
  - good: 278,955
  - bad: 49,762
  - bad rate: 15.14%
- Validation: 36,333
  - good: 29,893
  - bad: 6,440
  - bad rate: 17.72%
- Test: 41,904
  - good: 35,951
  - bad: 5,953
  - bad rate: 14.21%

Total:
- 406,954
- no data loss

Non-empty hostname overlap:
- Train/Validation: 0
- Train/Test: 0
- Validation/Test: 0

---

## 13. URL Logistic Regression

Majority baseline:
- Test accuracy: 84.73%

Random-split lexical model:
- Validation accuracy: 74.80%
- Precision: 0.3341
- Recall: 0.6547
- F1: 0.4424
- ROC-AUC: 0.8152
- Confusion: [[39554, 12166], [3219, 6104]]

Domain-aware lexical model:
Validation:
- Accuracy: 0.7300
- Precision: 0.3569
- Recall: 0.6522
- F1: 0.4613
- ROC-AUC: 0.8113
- Confusion: [[22324, 7569], [2240, 4200]]

Test:
- Accuracy: 0.7842
- Precision: 0.3536
- Recall: 0.6264
- F1: 0.4520
- ROC-AUC: 0.8374
- Confusion: [[29133, 6818], [2224, 3729]]

Conclusion:
- URL lexical features provide genuine but noisy signal.
- URL model is secondary evidence only.

---

## 14. Evidence / Fusion Experiments

DistilBERT test probabilities:
- 1,299 rows
- range: 0.000068 to 0.999635
- saved to:
  results/fusion/text_test_evidence.csv

Test URL extraction:
- messages with URLs: 685
- messages without URLs: 614
- URL occurrences: 2,520
- saved to:
  results/fusion/text_test_with_urls.csv

Test URL scoring:
- messages with scored URLs: 685
- messages without: 614
- maximum URL probability: 0.999954
- mean URL probability: 0.279598
- saved to:
  results/fusion/final_text_evidence.csv

Test evidence:
- text probability mean:
  - legitimate: 0.0047
  - malicious: 0.9540
- URL max probability mean:
  - legitimate: 0.3209
  - malicious: 0.2831
- URL presence:
  - legitimate: 54.88%
  - malicious: 46.32%

Validation evidence:
- DistilBERT validation F1: 0.9829
- Recall: 0.9723
- AUC: 0.9992

Naive fusion:
- 90% text + 10% URL:
  - F1: 0.9829
  - Recall: 0.9723
  - AUC: 0.9930
- 90% text + 10% signals:
  - F1: 0.9813
  - Recall: 0.9692
  - AUC: 0.9971
- 80% text + 10% URL + 10% signals:
  - F1: 0.9813
  - Recall: 0.9692
  - AUC: 0.9957

Decision:
- Keep DistilBERT as the primary classifier.
- Do not force weighted fusion.

---

## 15. Validation False-Negative / Escalation Analysis

Validation false negatives:
- 9 total
- 7 had URLs
- 2 had no URLs
- URL max probability mean: 0.4871
- URL max probability median: 0.4834
- URL max probability maximum: 0.9996
- urgency signals: 2
- financial signals: 2
- authority signals: 3
- any signal: 3

Hard URL escalation rule tested:
- text probability < 0.10
- URL probability >= 0.90

Result:
- 56 escalated
- 2 true positives
- 54 false positives
- precision: 0.0357
- malicious recall: 0.0062

Decision:
- Reject automatic URL escalation.
- URL evidence remains contextual/secondary.

---

## 16. Validation Risk Distribution

Validation probability bands:
- 0.00-0.10:
  - n=977
  - malicious rate=0.0082
- 0.25-0.40:
  - n=2
  - malicious rate=0
- 0.40-0.60:
  - n=3
  - malicious rate=0.6667
- 0.90-1.00:
  - n=317
  - malicious rate=0.9937

Other intermediate bands had no reported observations.

Conclusion:
- DistilBERT outputs are highly bimodal.
- Do not invent arbitrary smooth Low/Medium/High probability thresholds.
- Derive empirical risk bands before the final web application, or transparently document the bimodality.

---

## 17. Auxiliary Evidence Flag

Validation flag:
- strong URL >= 0.90 OR >=2 manipulation signals

Results:
- 177/1,299 flagged
- 13.63% of validation set
- malicious rate among flagged: 54.80%
- precision: 0.548
- recall: 0.2985
- strong URL: 97
- strong signals: 88
- both: 8

Decision:
- Useful as supporting evidence/context.
- Not strong enough to replace primary classifier.

---

## 18. Robustness Test Set

Robustness source:
- data/processed/text_splits/test.csv

Size:
- 1,299 rows
- legitimate: 973
- malicious: 326
- malicious rate: 25.10%

Normalized duplicate texts:
- 0

Malicious message length:
- character mean: 3,142.98
- character median: 1,483.5
- character maximum: 72,995
- word mean: 499.53
- word median: 206.5
- word maximum: 11,627

Length buckets:
- <50 chars: 1
- 50-99: 4
- 100-199: 5
- 200-499: 34
- 500-999: 62
- >=1000: 220

Eligible robustness messages:
- malicious messages with >=20 words: 318
- already detected by DistilBERT: 304
- already missed by DistilBERT: 14
- original recall on eligible messages: 95.60%

Primary robustness analysis:
- Focus on the 304 originally detected messages.
- Keep the 14 original false negatives as a separate hard-case analysis.

Detected pool:
- Ling: 66
- SpamAssasin: 238

Confidence bands among detected messages:
- 0.50-0.75: 2
- 0.75-0.90: 2
- 0.90-0.99: 5
- 0.99-1.00: 295

Fixed robustness sample:
- 100 messages
- random_state=42
- Ling: 25
- SpamAssasin: 75

Ling:
- 0.50-0.75: 1
- 0.75-0.90: 0
- 0.90-0.99: 4
- 0.99-1.00: 20

SpamAssasin:
- 0.50-0.75: 1
- 0.75-0.90: 2
- 0.90-0.99: 1
- 0.99-1.00: 71

Sample file:
- results/fusion/robustness_sample_100.csv

---

## 19. Robustness Transformation Conditions

Three conditions:

1. Professional rewrite
   - Preserve malicious intent and requested action.
   - Improve grammar, structure, and tone.
   - Introduce no new behavior.

2. Cue-reduced rewrite
   - Preserve malicious intent and requested action.
   - Remove exaggerated urgency, obvious scam phrasing, awkward grammar,
     and other superficial scam cues.
   - Preserve underlying malicious objective.

3. Combined rewrite
   - Professional rewrite + cue reduction.

Generated transformation records should contain:
- sample_id
- source
- original_text
- transformation_type
- transformed_text

Raw generated adversarial phishing/scam text:
- Keep local/gitignored.
- Do not commit raw operational phishing text to public GitHub.
- Public documentation should use aggregate metrics and neutralized examples.

---

## 20. Robustness Structural Protection

Generator:
- src/evaluation/generate_robustness_rewrite.py

Protected content types:
- legacy/spaced URLs
- normal URLs
- www URLs
- email addresses
- percentages

Current placeholders:
- __URL_
- __EMAIL_
- __PCT_

Previous issue:
- Regexes initially contained doubled backslashes and failed to detect
  protected structures.
- This was corrected.

Protection dry run:
- 100/100 samples passed
- 0 failures
- 154 URLs protected
- 62 email addresses protected
- 83 percentages protected
- Ling: 25/25
- SpamAssasin: 75/75

Dry-run file:
- results/fusion/robustness_protection_dry_run.csv

---

## 21. Placeholder Validation Policy

Validation checks:
- placeholder set equality
- each placeholder appears exactly once
- restoration of all protected values

Exact placeholder order:
- retained as a diagnostic
- NOT a rejection criterion

Rationale:
- Reordering complete placeholders does not compromise restoration correctness.
- Exact order is therefore diagnostic rather than structural validity.
- This rationale must remain documented so the methodology change does not look post-hoc.

---

## 22. Groq Generation

Provider:
- Groq
- OpenAI-compatible client
- no OpenAI billing

Model:
- openai/gpt-oss-120b

Environment:
- GROQ_API_KEY loaded from local .env
- actual key must never be exposed

Generation policy:
- maximum 2 attempts
- attempt 1 = normal controlled prompt
- attempt 2 = stricter retry if JSON parsing or placeholder validation fails
- if attempt 2 fails, discard and log
- never manually repair generated transformations

Single-sample test:
- Source: Ling
- Original probability: 0.729175
- Final status: pass
- Reason: validated

Attempt 1:
- finish_reason: length
- prompt tokens: 1,729
- completion tokens: 3,072
- raw characters: 5,139
- JSON valid: false

Attempt 2:
- finish_reason: stop
- prompt tokens: 1,779
- completion tokens: 2,572
- raw characters: 4,001
- JSON valid: true
- placeholder validation: passed
- sequence match: true

Final placeholder validation:
- Original placeholders: 8
- Rewritten placeholders: 8
- Exact sequence match: true
- Placeholder set match: true
- Each placeholder exactly once: true
- Restoration: successful

Important observation:
- Attempt 1 hitting the token limit demonstrates a potential long-message
  generation bias.
- Before scaling, retry/failure behavior must be analyzed by original message
  length.
- Do not silently discard long messages without measuring the pattern.

---

## 23. Generation Artifact Investigation Pending

The successful sample contained apparent merged-word artifacts:
- "latestgig"
- "copedfine"

These have NOT yet been established as model-generation artifacts or
post-processing/reconstruction artifacts.

Required investigation:
- Compare raw model JSON output against the final restored text for this
  exact sample.
- Determine whether merging occurred:
  1. in the model output itself,
  2. during JSON parsing,
  3. during placeholder restoration,
  4. or in another reconstruction step.

Do not run semantic/cosine screening at scale until this is understood.

---

## 24. Semantic Preservation Rubric

For each generated transformation:

- objective_retained: YES/NO/N/A
- requested_action_retained: YES/NO/N/A
- entities_claims_retained: YES/NO/N/A
- financial_intent_retained: YES/NO/N/A
- credential_intent_retained: YES/NO/N/A
- authority_impersonation_retained: YES/NO/N/A
- important_behavior_retained: YES/NO/N/A
- new_behavior_introduced: YES/NO/N/A

Rejection rule:
- Any applicable preservation criterion = NO -> reject.
- new_behavior_introduced = YES -> reject.
- N/A is permitted where genuinely not applicable.

---

## 25. Semantic Similarity Screening

Automatic screen:
- cosine similarity >= 0.85 -> pass
- cosine similarity < 0.85 -> obvious failure/reject

Important:
- This is a screening threshold only.
- It is NOT proof of semantic equivalence.

Manual audit:
- 60 transformations total
- 20 professional
- 20 cue-reduced
- 20 combined

Selection:
- After automatic screening.
- Fixed stratification across:
  - source
  - original confidence band
  - automatic similarity band

If fewer than 60 survive:
- Report the actual number.
- Do not lower the threshold.
- Do not manufacture samples.

---

## 26. Statistical Testing

Primary paired comparison:
- Original vs transformed predictions.

McNemar decision rule is documented:
- Discordant pairs >=20 -> standard chi-square McNemar
- Discordant pairs <20 -> exact McNemar

Do not choose the variant after seeing results.

Primary robustness interpretation:
- Focus on the 304 originally detected malicious messages.
- Keep the 14 original misses separate.

---

## 27. Groq Reproducibility

Before scaling:
- Run the exact same prompt twice under the final generator implementation.
- Diff outputs.

Do not assume temperature 0.0 guarantees deterministic hosted inference.

Log:
- model
- temperature
- top_p
- other generation parameters
- transformation type
- prompt/version identifier
- timestamp
- attempt number
- validation status
- finish reason
- token counts

An earlier reproducibility test showed different outputs at temperature 0.0,
including wording/formatting changes such as 9-11 vs 9/11, but that test occurred
before the final corrected URL/email/percentage protection implementation.
Therefore it must be rerun under the final code before being treated as the
official reproducibility result.

---

## 28. PhiUSIIL Dataset Comparison Gate

The PhiUSIIL comparison remains part of the documented evaluation plan.

Current URL benchmark:
- domain-aware test F1 = 0.4520

Quantitative switch threshold:
- Current F1 + 0.05 = 0.5020

Decision rule:
- Switch to PhiUSIIL ONLY if its domain-aware test F1 >= 0.5020.
- Do not override this rule qualitatively after seeing results.

Qualitative checklist must be scored BEFORE looking at comparison results:
- published/documented collection date
- documented collection/labeling methodology
- sample size
- label definition clarity
- provenance/source traceability
- license/redistribution terms
- whether derived features can be redistributed
- metadata sufficient to reproduce processing

Important execution decision:
- The PhiUSIIL comparison is a separate URL-dataset validation branch.
- It is independent of the text robustness generation branch.
- It must not block the robustness experiment indefinitely.
- However, it MUST be executed and documented before the final URL dataset choice
  is declared in the final project methodology.
- The robustness branch may continue after the comparison if the current URL
  dataset remains selected.
- The final README must explicitly state which URL dataset was selected and why.

---

## 29. Current Execution State

Completed:
- project setup
- environments
- dataset acquisition
- dataset investigation
- text leakage checks
- text split
- TF-IDF baseline
- text error analysis
- DistilBERT
- cross-source text evaluation
- Nazario external evaluation
- manipulation signal analysis
- URL dataset investigation
- URL lexical feature extraction
- random URL baseline
- hostname leakage audit
- domain-aware URL split
- domain-aware URL baseline
- evidence extraction
- fusion experiments
- fusion rejection
- robustness sample selection
- robustness protection dry run
- initial Groq generator
- controlled two-attempt generation policy
- single-sample structural generation test

Current stage:
- robustness generation validation before full 100 x 3 scaling

Required next stages:
1. PhiUSIIL comparison branch and dataset-choice documentation.
2. Complete generation-length/retry-bias audit before scaling.
3. Root-cause the latestgig/copedfine artifacts by comparing raw JSON to restored output.
4. Official Groq reproducibility test under final code.
5. Implement semantic rubric recording.
6. Implement cosine similarity screening at >=0.85.
7. Generate 100 x 3 transformations with retry/discard policy.
8. Perform fixed 60-transformation manual audit.
9. Evaluate original vs transformed predictions.
10. Calculate recall/F1 degradation and paired McNemar tests.
11. Finalize empirical risk bands.
12. Build Next.js web application.
13. Complete architecture diagram.
14. Finalize README with methodology, evaluation, robustness, limitations,
    ethics, and reproducibility.
15. Prepare GitHub-ready repository.

---

## 30. Non-Negotiable Workflow

- One terminal command at a time.
- Inspect output before giving the next command.
- Do not restart completed work.
- Do not silently alter documented thresholds or sampling rules.
- Do not invent metrics.
- Do not force fusion when experiments do not support it.
- Do not claim rule-based signals are model explanations.
- Do not claim random-split performance proves generalization.
- Do not lower robustness screening thresholds to obtain enough samples.
- Do not manually repair rejected generated transformations.
- Keep raw generated adversarial text local/gitignored.
- Preserve reproducibility metadata.
- Report failures and discarded samples honestly.

---

## 31. PhiUSIIL Pre-Result Dataset Assessment

Official source:
- UCI Machine Learning Repository, dataset ID 967
- Name: PhiUSIIL Phishing URL (Website)
- Dataset DOI: 10.1016/j.cose.2023.103545
- Authors: Arvind Prasad and Shalini Chandra
- Published in Computers & Security
- Dataset creation year: 2024
- UCI last updated: May 12, 2024

Dataset size:
- 235,795 instances
- 134,850 legitimate
- 100,945 phishing
- 54 documented predictive features
- No missing values

Qualitative checklist completed before benchmark results:

1. Published/documented collection date:
   PASS
   - UCI documents dataset creation in 2024 and last update in 2024.

2. Collection/labeling methodology documented:
   PASS
   - UCI documents that instances represent URLs and corresponding webpages.
   - Features are derived from URL and webpage/source-code information.

3. Sample size:
   PASS
   - 235,795 instances.

4. Label definition clarity:
   PASS, pending local confirmation of semantic mapping already observed:
   - local data contains labels 0 and 1.
   - observed data indicates label 0 = phishing and label 1 = legitimate.

5. Provenance/source traceability:
   PASS
   - Official UCI record, dataset ID 967, DOI, authors, and associated
     peer-reviewed publication are documented.

6. License/redistribution terms:
   PASS
   - UCI lists CC BY 4.0.
   - Attribution is required when sharing/adapting.

7. Derived-feature redistribution:
   PASS WITH CAUTION
   - Dataset is CC BY 4.0, but the project will not assume that every
     derived feature is independently reproducible from URL text alone.
   - Any redistribution will preserve attribution and dataset provenance.

8. Metadata sufficient for reproducible processing:
   PASS WITH LIMITATION
   - UCI provides feature names, dataset metadata, DOI, and dataset access.
   - Some PhiUSIIL features depend on webpage/source-code information and
     therefore cannot be reconstructed from a URL string alone.

Documented quantitative comparison gate:
- Current domain-aware URL test F1: 0.4520
- Required improvement: +0.05
- PhiUSIIL switch threshold: F1 >= 0.5020

Documented decision rule:
- Switch to PhiUSIIL only if the controlled domain-aware comparison reaches
  test F1 >= 0.5020.
- No qualitative override after observing performance.

Controlled comparison methodology:
- A URL-only PhiUSIIL feature subset will be evaluated against the current
  URL-only lexical feature approach for the primary dataset-choice comparison.
- Webpage/source-code-derived PhiUSIIL features will not be mixed into the
  primary apples-to-apples comparison because they require information that
  the current URL-only pipeline does not collect.
- If full PhiUSIIL features are evaluated separately, they will be reported
  as a distinct benchmark and not used to claim a directly comparable
  improvement over the current URL-only system.

Execution independence:
- The PhiUSIIL comparison is a separate URL-dataset branch.
- It is not a prerequisite for continuing the text robustness-generation
  branch.
- The final URL dataset choice must nevertheless be documented using the
  documented quantitative gate before the final project methodology is
  frozen.

## 32. PhiUSIIL Leakage / Shortcut Audit and Final URL Dataset Decision
- PhiUSIIL raw dataset: 235,795 rows; 235,370 unique URLs; 425 duplicate URL rows; 0 conflicting URL labels.
- 54 domains contained mixed labels, covering 112 rows (~0.047% of the dataset). Domain-aware splitting prevents these domains from crossing train/validation/test partitions.
- Domain-aware split: train 189,074; validation 23,044; test 23,677; zero domain overlap across all partition pairs.
- Explicit URL-overlap audit: train-validation 0; train-test 0; validation-test 0.
- URL-only logistic regression using the 15 reproducible URL-structural features achieved test F1 0.9958 and ROC-AUC 0.9986.
- Shortcut audit removed IsHTTPS and IsDomainIP. Validation and test metrics were unchanged (test F1 0.9958; ROC-AUC 0.9986), indicating that performance is not dependent on those two binary features.
- Strong class-conditional structural differences remain in PhiUSIIL, including URL length, letter/digit composition, and special-character statistics. These may limit external generalization.
- Final decision: use PhiUSIIL as the primary URL benchmark; retain the original malicious-URL dataset as a secondary harder/generalization reference.
- Reporting constraint: do not present PhiUSIIL performance as an estimate of real-world deployment performance. Clearly describe dataset-specific structural separation as a limitation.

## 33. Robustness Generation Reproducibility and Semantic Screening
- Final protected Groq generator was tested twice on the same Ling sample using the same transformation.
- Both generations passed JSON and placeholder validation, but the rewritten texts differed materially.
- Exact-output reproducibility: FAIL.
- Structural placeholder reproducibility: PASS.
- This non-determinism is accepted as an LLM-generation characteristic; identical paraphrases are not required for the robustness experiment.
- Added local semantic screening using sentence-transformers `all-MiniLM-L6-v2` with normalized 384-dimensional embeddings.
- Cosine similarity threshold was set at >=0.85 for screening.
- The threshold is used only as a screening signal, not as proof of semantic preservation.
- Final screening pipeline: Groq generation -> placeholder validation -> cosine similarity screening -> structured semantic rubric -> manual audit.
- Structured rubric fields: objective retained, requested action retained, entities/claims retained, financial intent retained, credential intent retained, authority/impersonation retained, important behavior retained, new behavior introduced.
- Any applicable preservation field marked NO rejects the transformation; new_behavior_introduced=YES rejects the transformation.
- Transformations are not regenerated solely to obtain a passing similarity score, avoiding selection bias.
- Toy semantic-screen test produced cosine similarity 0.8483 and correctly returned REVIEW_REQUIRED.

---

## 34. Robustness Batch Generation Checkpoint

The fixed robustness sample contains 100 malicious test messages, with three
predefined transformations per message: professional, cue_reduced, and
combined.

Batch generation began using the Groq-hosted openai/gpt-oss-120b model with
temperature=0.0 and MAX_ATTEMPTS=2.

At the first quota interruption:
- 51/300 transformation records were completed and persisted.
- Professional: 17 records.
- Cue-reduced: 17 records.
- Combined: 17 records.
- 49 generations passed mechanical generation validation.
- 2 generations were discarded.
- Of the 49 mechanically valid generations, 38 passed cosine screening
  (>=0.85) and 11 were rejected by the predefined similarity threshold.

Cosine distribution among the 49 mechanically valid generations:
- minimum: 0.2995
- 25th percentile: 0.8662
- median: 0.9140
- 75th percentile: 0.9473
- maximum: 0.9947
- <0.80: 9
- 0.80-0.85: 2
- 0.85-0.90: 8
- 0.90-0.95: 18
- >=0.95: 12

The batch was interrupted by Groq's daily token-per-day rate limit. The
resumable JSONL output retains completed records and the batch runner skips
existing sample/transformation pairs on subsequent runs.

The 0.85 cosine threshold was not changed in response to the observed
rejection rate.

## 35. Robustness Final Statistical Analysis
- Primary inference is performed at the base-message level because multiple verified transformations can originate from the same underlying message.
- Unique base messages: 42.
- Base messages with at least one malicious-to-benign threshold flip at the predefined 0.50 threshold: 4/42 (9.52%).
- 95% bootstrap CI for base-message flip rate: 2.38%–19.05% (10,000 resamples, random seed 42).
- Original mean malicious probability across base messages: 0.98328.
- Mean transformed malicious probability, using the mean score across each base message's available verified transformations: 0.92947.
- Mean paired score change: -0.05381.
- 95% bootstrap CI for mean paired score change: -0.12047 to -0.00185 (10,000 resamples, random seed 42).
- One-sided Wilcoxon signed-rank test on the 42 base-level paired scores: W=766, p=1.68e-05.
- The 59 verified transformation instances remain a secondary descriptive analysis and are not treated as 59 independent base-message observations.
