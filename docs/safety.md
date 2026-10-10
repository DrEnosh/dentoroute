## What this project is

DentoRoute is a research and education prototype. It explores one question: when a model gives you probabilities instead of a single answer, how should software use them to decide between acting automatically and asking a person?

Dental triage is the example domain because the stakes make the question concrete. The project is not a dental product, and it is not a medical device.

## What it is not

- It does not diagnose anything.
- It does not recommend treatment, medication or doses.
- It has not been clinically validated, reviewed by clinicians, or tested on real patients.
- It must not be used to decide how urgently a real person should be seen.
- It does not replace a dentist, a doctor or an emergency service. If someone has trouble breathing or swallowing, or rapidly spreading facial swelling, they need emergency care, not this tool.

## Data

Every case is synthetic. A seeded generator builds structured case states from scenario templates (routine check-up, facial swelling, ambiguous complaint and so on). No real patient data, photos, records or identifiers are used or accepted. Do not type real patient information into the Case Simulator.

## The reference labels are not clinical truth

Benchmark accuracy is measured against `data/ground_truth_rules.yaml`, a deterministic rule set written for this project. The rules are labelled "Educational synthetic benchmark rules" because that is what they are. They are not clinical guidelines, and no clinician checked them. A model that agrees with them has learned this rubric, not good dentistry. See `docs/benchmark_methodology.md` for the specific weaknesses of this setup.

## Thresholds are research-demo thresholds, not clinical thresholds

Every number in `config/routing_policy.yaml` is a research-demo threshold. None is a clinical threshold. They were selected by a script on synthetic cases scored against an unreviewed rubric, and the shipped values were tuned on the demo simulator's output, which is not Jev. They do not say how long a real patient can safely wait or what miss rate a real service could accept. That needs clinicians, real data and prospective validation.

The routing gate is deliberately conservative: it automates a case only when every check passes, and any doubt, high infection concern, low confidence or urgent-looking pattern sends it to a person. The first version of the gate let 61 reference-urgent or human-review cases through on the simulated run, which is why it was replaced. The full rules, how the thresholds were chosen on separate train, calibration and holdout splits, and what the results do and do not show are in `docs/safety_gate.md`.

Hard limits in `backend/services/policy_service.py` stop the YAML from being loosened past what the code allows. Zero missed urgent cases on a small holdout is not zero risk, and the study reports an upper confidence bound to say so.

## How the design limits harm

- Jev produces decisions. It never triggers an action. Plain Python applies the policy.
- Low confidence, high infection concern, urgent-looking patterns and a hard-coded list of specialties all lead to human review. Automation is the exception, not the default.
- Any invalid, missing or malformed model output leads to human review, never to a guess.
- Claude only explains. It receives the facts in the case and the engine's decisions, its list of "signals" is restricted to findings actually present, and its text is checked for clinical terms the case does not support. If the check fails, a deterministic template is shown instead.
- Every screen shows the safety notice. Demo Mode is labelled whenever simulated responses are in use.

## Demo Mode

With no `TYPESAFE_API_KEY`, the app runs a deterministic simulator that produces responses in Jev's format. It is not Jev, it is always labelled as such, and benchmark numbers produced from it are marked SIMULATED. It exists so the interface, routing and benchmark code can be explored without keys.

## Privacy and secrets

API keys are read from environment variables, are never logged, and are redacted if they appear in a log line. `.env` is git-ignored. The local SQLite database stores only synthetic cases and model outputs.

## Known limitations

- The synthetic cases and the rubric share an author, so the benchmark is easier and cleaner than reality.
- Categories are balanced by design, so class frequencies do not reflect any real population.
- Results from one run on one synthetic dataset say little about performance anywhere else.
- Claude's confidence values are self-reported. Jev's are computed from its probability distributions. They are different kinds of number and are not directly comparable.
