# Build Log

This file is the public record of ongoing work. Each meaningful coding, data, training, or evaluation session should add one short entry with evidence and the next question.

## 2026-08-22 - Teacher and recovery evaluation artifacts

- Retained deterministic teacher, recovery-gate, lifecycle, and five-minute admission reports for repeatable evaluation.
- Continued separating trainable observations from inconclusive or unsafe action evidence.
- Next: consolidate the strongest evaluation path into a reproducible command and record another independent episode.

## 2026-08-16 - Imitation-policy v4 experiment

- Trained `doom_policy_v4` on 87 sequences and evaluated against 18 time-block validation sequences.
- Best validation loss was 0.5767 at epoch 9.
- Recorded the central limitation: one source episode is not enough to establish episode-level generalization.
- Next: capture more independent demonstrations and compare episode-held-out performance.

## 2026-08-15 - Closed-loop observation milestone

- Added passive discovery and scouting modes.
- Added candidate tracking, crosshair-relative aim error, and structured replay events.
- Kept live action bounded and opt-in.
- Next: improve enemy evidence quality before treating firing results as conclusive.

## Entry template

```text
## YYYY-MM-DD - Short milestone

- Changed:
- Evidence:
- Limitation:
- Next:
```

