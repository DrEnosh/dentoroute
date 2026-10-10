## Purpose

The benchmark answers three practical questions about a decision layer used for triage-style routing:

1. How often do its answers match a fixed reference?
2. Do its probabilities mean what they say (calibration)?
3. Can confidence be used to decide which cases to automate?

It can also run the same cases through Claude with forced structured output, so the two approaches can be compared on the same inputs.

## Dataset

`data/synthetic_cases.json` holds 525 synthetic cases: 21 scenario categories with 25 cases each, generated with a fixed seed (42) by `backend/services/synthetic_case_service.py`. Regenerate with `python scripts/generate_dataset.py`. Within a scenario, details such as age, pain level, side of the mouth and medical history are randomised, and some scenarios include deliberate noise or missing information.

Categories are balanced by construction, so the label distribution is skewed toward rarer, more serious outcomes than a real clinic would see. Reference urgency counts: ROUTINE 220, PRIORITY 227, URGENT 50, EMERGENCY 28.

## Reference labels

`data/ground_truth_rules.yaml` is a deterministic rule engine, executed by `backend/services/ground_truth_service.py`. It labels each case on all seven questions from the structured case state alone.

- Urgency, referral priority: the highest matching level wins.
- Infection concern, same-day review, human review: true if any rule matches.
- Specialty: the first matching ordered rule wins.
- Complexity: matching rules add points, and thresholds map points to a level.

Neither Claude nor Jev produced these labels. The case generator's scenario category is not used to label cases, and it is not sent to either model.

## Questions

The same seven questions are asked of both providers.

| Question | Jev primitive | Claude output field |
|---|---|---|
| urgency (ROUTINE / PRIORITY / URGENT / EMERGENCY) | Score | enum + verbalized confidence |
| specialty (10 labels) | Choice | enum + verbalized confidence |
| infection_concern | Noul | boolean + probability |
| same_day_review | Noul | boolean + probability |
| human_review | Noul | boolean + probability |
| complexity (4 levels) | Score | enum |
| referral_priority (4 levels) | Score | enum |

For Jev, all seven are sent as atomic questions in a single request. For Claude, one request per case forces a `submit_decisions` tool call whose JSON schema restricts every label to the allowed set. The wording of each question comes from one place (`QUESTION_SPECS` in `jev_service.py`) and is reused in Claude's prompt.

For Score and Choice questions, the predicted label is the most probable level (argmax of Jev's probabilities). The label nearest to Jev's `score` is stored separately; when a distribution is split between two distant levels the two can disagree, and the argmax is the more honest answer.

## Metrics

- Categorical questions: accuracy, macro-F1, confusion matrix. Ordinal ones (urgency, complexity, referral priority) also report within-one-level accuracy and mean absolute error in levels. Urgency additionally reports under-triage rate (predicted lower than reference) and over-triage rate. Specialty reports top-3 accuracy where probabilities exist.
- Yes/no questions: accuracy, precision, recall, F1, Brier score, confusion matrix.
- Schema compliance: valid responses divided by responses received. Network failures are counted separately and are not schema failures. An invalid label, a missing field or probabilities that do not sum to one count as invalid output.
- Latency: measured only for real API calls. Cached responses and Demo Mode are excluded. If nothing was measured, the report says "not measured".
- Cost: computed from reported token usage only when price environment variables are set. Otherwise "not measured". Nothing is estimated from list prices.

## Calibration

All calibration numbers use the probability a provider attaches to its own answer.

- Reliability diagram: ten equal-width bins on the probability of the predicted answer, plotted against observed accuracy in each bin.
- ECE: the bin-size-weighted mean gap between stated probability and observed accuracy.
- Brier score: multiclass (sum over classes) for Score and Choice questions, binary for yes/no questions.
- Confidence histogram: stacked by correct and incorrect.
- Probability histogram: every probability emitted, to show how peaked the outputs are.
- Accuracy by confidence band: low (below 0.5), medium (0.5 to 0.8), high (0.8 and above), the same bands the routing policy uses.
- "Does higher confidence mean higher accuracy?": Spearman correlation between confidence and correctness, plus whether accuracy rises across occupied bands. The report refuses to answer below 30 samples.

Jev's "confidence" for Score and Choice questions is the one its documentation defines; for yes/no questions we use |2p - 1|. Claude has no probability distributions. For Claude, its self-reported confidence is used as the probability of its answer, and the benchmark labels it as verbalized.

## Confidence-gated routing analysis

For a grid of thresholds t, a case is "automated" if min(urgency confidence, specialty confidence) >= t. The risk-coverage curve plots the share of cases automated against the accuracy of those cases, where a case counts as correct only if both urgency and specialty match. The routing summary also counts cases the active policy automated even though the reference says URGENT, EMERGENCY or human review.

That summary is descriptive and is computed on every case. The safety-gate study is the separate, held-out evaluation: it fits and tunes on train and calibration splits, reports on a holdout split, and compares the old policy with the revised one using false automation, unnecessary human review, coverage and automation accuracy. See `docs/safety_gate.md`.

## Comparing Jev and Claude

The comparison uses only cases both providers answered. For each question it reports the accuracy difference with a 95% bootstrap confidence interval (2000 resamples, fixed seed). A winner is named only if the interval excludes zero; otherwise the verdict reads "no clear difference". The comparison table also shows schema compliance, invalid outputs, errors, latency and cost.

Demo Mode never produces a comparison. The simulator is not Jev, so comparing it with Claude would mean nothing.

## Running it

```
python scripts/run_benchmark.py --demo                     # simulated, no keys
python scripts/run_benchmark.py --provider both --live     # needs both keys
python scripts/run_benchmark.py --provider jev --limit 100 --live
```

Results are stored in SQLite, shown in the Benchmark Lab, and written as CSV and JSON under `results/`. Live Jev responses are cached by a hash of the case state, question-set version and model, so re-runs do not repeat requests.

## Threats to validity

- The rubric and the prompts have the same author. The questions explain what each level means, and the rubric encodes a similar idea in rules. Agreement is partly built in.
- Cases are generated from templates. They are cleaner and more prototypical than real presentations.
- The rubric is a function of the same fields the models see. A model that reads those fields carefully should do well; this measures instruction-following on structured input more than clinical judgement.
- No clinician reviewed anything.
- Balanced categories distort class frequencies, which affects precision and calibration relative to any real population.
- Single-run, single-seed results. Repeat with other seeds before drawing conclusions.
- The routing thresholds are research-demo values, not clinical thresholds. The shipped ones were tuned on a train/calibration split of simulated outputs and reported on a holdout, but the holdout is only 158 cases with 24 reference-urgent ones, so even zero misses leaves a wide confidence bound.
- Simulated results (Demo Mode) describe the simulator only. They are labelled SIMULATED everywhere.
