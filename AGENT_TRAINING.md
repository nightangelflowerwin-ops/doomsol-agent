# Dwasm deterministic teacher build

This fork adds a training-only WebAssembly surface. It does not replace the
DoomSol browser bridge and it does not expose privileged truth to the deployed
V4 policy.

## API

- `agent_reset(skill, episode, map, seed)` starts a seeded episode.
- `agent_set_action(forward, strafe, turn, buttons)` queues one Doom ticcmd.
- `agent_step(tics)` advances at Doom's fixed 35 Hz, bounded to 35 ticks/call.
- `agent_state_json()` exports player, health, ammo and progress truth.
- `agent_probe_move()` asks the real collision engine whether a move is valid.
- `agent_map_line_*()` exports authoritative linedefs.
- `agent_enemy_*()` exports living enemy state and engine line-of-sight.

`wasm/teacher_runner.js` uses that truth to create seeded visual/action samples.
Each row stores a rendered browser frame plus the teacher action. Truth is kept
for auditing only.

## Build prerequisite

Install Emscripten SDK, activate it, then use the repository's normal CMake
WebAssembly build. A legal Doom IWAD is still required; no game assets are
included here.

## Launch teacher mode

From this repository, run:

```powershell
.\start_teacher.ps1 -Seconds 300 -Seed 1701
```

The launcher verifies the WebAssembly build, starts the local collector,
waits for an HTTP readiness response, opens
the configured local teacher URL, and writes a timestamped
JSONL recording under `outputs/doomv9/models/dwasm_teacher`.

For a headless launch or a port conflict:

```powershell
.\start_teacher.ps1 -NoBrowser -Port 8151
```

Before using a recording for training, run the admission checker. The five-minute
gate requires synchronized ticks, changing positions and frames, firing, door use,
kills, and health changes:

```powershell
python .\wasm\validate_teacher.py <recording.jsonl> `
  --expected-seconds 300 --require-use --report <recording-admission.json>
```

An exit code of 2 means the recording is quarantined and must not enter training.

## Distill into V4

Write teacher samples as JSONL and fine-tune alongside the human run:

```powershell
python .\agent\train_imitation_v4.py `
  --run 20260816-054812-human-4b6de7 `
  --teacher-manifest .\models\dwasm_teacher\e1m1.jsonl `
  --teacher-weight 2 `
  --init-checkpoint .\models\doom_imitation_v4\doom_policy_v4.pt `
  --epochs 12
```

Validation remains human-only. This prevents a simulator-only improvement from
being mistaken for better DoomSol browser behavior.
