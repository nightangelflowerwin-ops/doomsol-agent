# Tactical oracle ten-minute run — 2026-09-16

## Purpose

This is an authoritative ViZDoom teacher, not a deployable visual policy. It uses all world objects, enemy coordinates, player position and facing, rendered line of sight, health changes and simultaneous button controls. The resulting trace is intended for later visual-policy distillation.

## Results

- Duration: 600 seconds
- Episodes: 43
- Decisions: 5,250
- Total player kills: 237
- Mean kills per episode: 5.5116 out of six enemies
- Attack decisions: 1,692
- Move-forward decisions: 1,971
- Turn-left decisions: 1,696
- Turn-right decisions: 1,714
- Strafe-left decisions: 1,377
- Strafe-right decisions: 1,564

Actions are multi-label, so a decision can fire, turn and move simultaneously. The controller prioritizes visible threats, turns toward authoritative bearings when targets are occluded, attacks only with line of sight and aim alignment, strafes under fire, backs away at close range and advances when the engagement is clear.

## Artifacts

- `runs/tactical-oracle-visible-10min.mp4`: 600 seconds, 421,673,076 bytes
  - SHA-256: `26BB643285B262625F129953AC9BA6783804BAE624A856CF6A6354032D50407A`
- `runs/tactical-oracle-visible-10min.jsonl`: 5,250 synchronized decisions, 2,344,543 bytes
  - SHA-256: `8EF8D890403E2156A9579FD1E864B52CCBAA490B017FC382B899F4A195E6C43F`

## Promotion boundary

These results prove the local teacher can fight. They do not prove that a pixel-only student has learned the same behavior. The next checkpoint must be trained from synchronized frames and the multi-label action trace, then evaluated without object truth.

