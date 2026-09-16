# ViZDoom ten-minute baseline — 2026-09-16

## Configuration

- Checkpoint: `examples/doom/checkpoints/deadly-dagger.pt`
- Policy: greedy seven-action released checkpoint
- Environment: ViZDoom `deadly_corridor`
- Resolution: 640 by 480 capture; 160 by 120 policy input
- Duration: 600 seconds of game time
- Respawn behavior: automatically start a new episode after death
- Device: CPU

## Results

- Episodes: 91
- Decisions: 5,250
- Mean reward: -102.8794
- Mean kills per episode: 0.4615
- Maximum observed running reward: 81.6772
- Mean online inference latency: 6.9393 ms
- Maximum visible objects in one decision: 12

## Action counts

| Action | Decisions |
| --- | ---: |
| Turn right | 2,070 |
| Turn left | 1,449 |
| Attack | 1,037 |
| Move forward | 694 |
| Move backward | 0 |
| Strafe left | 0 |
| Strafe right | 0 |

The released policy scored occasional two- and three-kill episodes, but died frequently and entered prolonged turn-only loops. It never selected backward or strafe actions during this greedy run. This is the pre-training baseline for the local `USE`-capable controller, not a promoted gameplay result.

## Local artifacts

The raw artifacts are intentionally excluded from Git because they exceed GitHub's normal per-file limit.

- `runs/agent-visible-10min.mp4`: 600.0 seconds, 201,353,505 bytes
  - SHA-256: `2F7E12BB213BE33D40CDAA086282F9D3BD334F252A2D389300BDC27937659003`
- `runs/agent-visible-10min-trace.json`: 626,374,223 bytes
  - SHA-256: `84815923D9614A7E80DA7ED75B1047E9C632683EACD57B8936F4C061EC2F5D9F`

