# DoomSol Agent

A local, closed-loop Doom agent project exploring visual perception, replay capture, imitation learning, evaluation, and conservative action control.

The project is deliberately honest about its current stage: it is an experimental learning system, not a finished autonomous player. The current source tree supports browser-to-agent observation, passive scouting, candidate tracking, replay logging, and bounded controller experiments.

## Why I am building it

This project is my practical route into AI training and data annotation. Doom provides a compact environment for learning the full loop:

1. Observe frames and identify useful visual evidence.
2. Label actions and outcomes consistently.
3. Train a policy from recorded demonstrations.
4. Evaluate the policy against held-out sequences.
5. Document failure states and improve the data.

That workflow connects directly to the work I want to do professionally: careful annotation, evidence-based evaluation, guideline application, and clear reporting of limitations.

## Current capabilities

- `discover`: inspects non-sensitive metadata without gameplay input.
- `scout`: captures frames passively for later review.
- `baseline`: runs a bounded benchmark controller and records actions.
- `closed-loop`: tracks conservative visual candidates and records structured replay events.
- Browser extension and local bridge for connecting the game tab to the Python agent.
- Perception tests covering the current candidate-detection layer.

## Verified experiment snapshot

The latest retained `doom_policy_v4` training report records:

- 87 training sequences
- 18 validation sequences
- sequence length of 32
- best validation loss of 0.5767 at epoch 9

Important limitation: the report uses a time-block holdout from one episode. Episode-level generalization cannot be claimed until additional independent demonstrations are recorded.

## Quick start

1. Start `run_bridge.bat`.
2. Reload the unpacked browser extension and attach the Doom tab.
3. Install the agent requirements in a Python environment.
4. Begin with non-playing discovery:

```powershell
python agent/agent.py --mode discover
```

5. Capture a passive sample:

```powershell
python agent/agent.py --mode scout --seconds 20
```

Only enable live action during short, bounded tests after reviewing the captured evidence.

## Repository map

- `agent/` - perception and controller code
- `bridge/` - local browser bridge
- `extension/` - browser extension files
- `tests/` - focused perception tests
- `WORKLOG.md` - chronological public build log
- `.github/workflows/verify.yml` - automatic verification after each push

## Public progress

Meaningful sessions are recorded through small commits, dated build-log entries, and automatic test runs. The exact routine is documented in `CONTRIBUTING.md` so the public history shows evidence of real progress rather than artificial contribution activity.

Large replay frames, trained model binaries, local environments, and temporary output are intentionally excluded from version control.

## Safety and privacy

No seed phrase or private key is needed or stored. Secrets, user profiles, local browser data, model binaries, and raw replay captures must not be committed.
