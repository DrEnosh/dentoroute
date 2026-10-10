## Why the gate was redesigned

The first routing policy (v1, kept as `config/routing_policy_legacy.yaml` for comparison) sent a case to urgent human review only when the urgency score reached 2.5, the same-day probability reached 0.80, or the infection probability reached 0.85. That left a gap. A confident "URGENT" call from the model has a score of about 2.0, so it passed straight through and was automated. Running v1 over the 525 synthetic cases on the demo simulator automated 61 cases that the reference rubric says are urgent or need human review. The old benchmark page counted these but nothing stopped them.

The v2 gate treats automation as the exception. A case is automated only if every check passes, and each check is set up so that doubt sends the case to a person.

## Thresholds are research-demo thresholds, not clinical thresholds

Every number in `config/routing_policy.yaml` is a research-demo threshold. None of them is a clinical threshold.

- They were chosen by a script on synthetic cases, scored against a rubric nobody clinical has reviewed.
- The shipped values were tuned on the demo simulator's output, which is not Jev. They show how the method works. They say nothing reliable about what Jev would do.
- Nothing here says how long a real patient can safely wait, or what miss rate is acceptable in a real service. Those are clinical and organisational decisions that need clinicians, real data and prospective validation.
- The policy file, the tuning script, the API response and the Benchmark Lab tab all carry this statement.

## The rules

The rules are checked in order and the first match decides the route. The code is in `backend/services/routing_service.py`.

| Rule | Fires when | Route |
|---|---|---|
| R0 | The model output is missing, invalid, or the provider failed | human review |
| R1 | P(human review needed) is at or above its threshold | escalate to human |
| R2 | P(infection concern) is at or above its threshold | urgent review |
| R3 | Any urgent-risk signal is high: urgency score, P(URGENT) + P(EMERGENCY), P(same-day review), or the learned reference-risk score | urgent review |
| R4 | A confidence floor is missed, or an infection / same-day / human-review answer is too uncertain to trust either way | human review |
| R5 | The chosen specialty is on the human-only list (emergency dental service, oral pathology) | urgent review or human review |
| R6 | Everything above passed | automated specialty route |

The requirements map onto the rules like this.

1. High infection concern triggers human review: R2. The trigger is also lower than before (v1 waited for 0.85).
2. Low-confidence decisions trigger human review: R4.
3. Emergency and urgent reference-risk patterns default toward human review: R3 and R5, plus the learned risk score described below.
4. Missed urgent cases come before automation: the thresholds are chosen under a missed-urgent budget (see "How thresholds are chosen"), and the hard limits below stop any config from loosening the gate.

Every result records the rule that fired and a plain-text reason with the actual numbers.

## Hard limits in code

`backend/services/policy_service.py` holds a ceiling for every trigger threshold and a floor for every confidence threshold. `validate_policy` runs on every routing call and raises `PolicyError` if a YAML file asks for something more permissive. The tuning grid is checked against the same limits, so a search can never wander outside them. Editing the YAML can make the gate stricter but not looser than the code allows. Loosening the limits means changing code and its tests.

Current ceilings: human-review probability 0.50, infection probability 0.50, same-day probability 0.50, urgency score 1.5, urgent-tail probability 0.40, risk score 0.50. Current floors: urgency confidence 0.40, specialty confidence 0.60, top-specialty probability 0.55, boolean certainty 0.20. These are also research-demo values.

## The learned reference-risk score

R3 includes a small logistic regression over ten features taken from the Jev output (urgency score and confidence, P(URGENT or EMERGENCY), specialty confidence and top probability, the three yes/no probabilities, complexity and referral scores). It predicts whether the reference rubric says the case needs a human. It is fitted on the train split only. Its coefficients are stored in the YAML, so routing needs no scikit-learn at runtime.

It exists because individual signals can each sit just under their threshold while the combination looks like cases that needed a person. It uses the model's outputs only and never reads the case text, so the Jev decision layer stays the only place a case is interpreted.

## How thresholds are chosen

The 525 synthetic cases are split 40/30/30 into train, calibration and holdout, stratified by reference urgency, with a fixed seed. The assignment is saved in `data/splits.json` and depends only on case IDs and reference labels, never on model outputs.

- Train: fits the risk score.
- Calibration: the only place thresholds are chosen. The search covers a grid (2,880 candidates). It keeps candidates that miss no more than the budget of urgent cases (zero), then picks the lowest expected cost, breaking ties toward the more cautious setting. Each chosen threshold is then moved one grid step further toward caution as a safety margin.
- Holdout: scored once, after the policy is frozen. It is never used to fit, select or compare candidates. `tune()` raises `HoldoutLeakError` if it is handed any row that is not from its own split, and a test confirms that changing every holdout row leaves the tuned thresholds unchanged.

Cost weights, used only for choosing thresholds and for the cost-per-case figure, are 100 for a missed urgent case, 20 for any other false automation, and 1 for an unnecessary human review. These are assumptions for the demo, not measured costs.

## Metrics

- Urgent: the reference urgency is URGENT or EMERGENCY.
- Needs human: urgent, or the reference marks infection concern, same-day review or human review.
- Missed urgent: automated and urgent.
- False automation: automated and needs human (so missed urgent is a subset).
- Unnecessary human review: sent to a person although the reference says no human is needed.
- Coverage: the share of all cases automated.
- Automation accuracy: specialty accuracy, urgency accuracy and joint accuracy among the automated cases only.

The study also reports a one-sided 95% Clopper-Pearson upper bound on the missed-urgent rate, a paired bootstrap (2,000 resamples) of the cost and coverage differences, and cost-per-case across missed-to-unnecessary cost ratios from 2:1 to 500:1.

## Results on the demo simulator (holdout, 158 cases, 24 reference-urgent)

These numbers come from the demo simulator, not from Jev.

| | Old policy | Revised policy |
|---|---|---|
| Missed urgent | 9 | 0 |
| Upper 95% bound on missed-urgent rate | 56.3% | 11.7% |
| False automation | 26 | 0 |
| Unnecessary human review | 26 | 71 |
| Coverage | 65.2% | 20.3% |
| Specialty accuracy when automated | 95.1% | 100.0% |
| Urgency accuracy when automated | 85.4% | 96.9% |
| Expected cost per case (100:1) | 8.01 | 0.45 |

Paired bootstrap, revised minus old: cost per case -7.56 (95% CI -11.3 to -4.2), coverage -44.9 points (95% CI -52.5 to -37.3).

## What these results do not show

- Zero missed cases in 24 urgent holdout cases is not zero risk. The true miss rate could still be around 12%.
- Safety is paid for with automation. Coverage fell from 65% to 20%, and the number of unnecessary reviews nearly tripled. If a missed urgent case were only about twice as costly as an unnecessary review, the old policy would have the lower cost per case. The revised policy wins from about 5:1 upward in this run.
- Across all 525 cases (which include the tuning data) the revised policy still automated one urgent case, in the train split. The calibration constraint does not carry over to every case.
- The simulator makes a quarter or so of reference-urgent cases look benign to the model. Those cases are exactly the kind a probability-based gate cannot catch. Real Jev outputs may differ in either direction.
- The rubric and the question wording share an author, so the reference labels are partly a mirror of the prompts.
- A guard that reads case findings directly (fever plus swelling, say) would catch some of those cases, but it would be checked against a rubric built from the same findings. The result would look better without being better evidence, so it is not included.
- One synthetic dataset, one split seed, one simulator. Nothing here is a clinical performance claim.

## Re-tuning on live data

```bash
python scripts/run_benchmark.py --provider jev --live      # all 525 cases
python scripts/tune_policy.py --live                       # study only, changes nothing
python scripts/tune_policy.py --live --write               # also writes config/routing_policy.yaml
python scripts/tune_policy.py --demo                       # the simulated version
```

`--write` rewrites the thresholds, the risk model and the provenance block, and keeps the header comment. The Benchmark Lab "Safety policy" tab and `POST /api/policy/study` run the same study but never change the active policy.

## Tests

`tests/test_safety_gate.py` covers each rule on its own, its boundary, its precedence over later rules, the hard limits, the disclaimer, and the original 2.5 gap (`test_r3_clean_urgent_call_is_not_automated_anymore`). `tests/test_policy_study.py` covers the split, leakage guards, tuning, hand-computed metrics, the Clopper-Pearson bound, the API and the study tab. `tests/test_routing_legacy.py` keeps the old policy under test as the comparison baseline.
