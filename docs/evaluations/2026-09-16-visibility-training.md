# Authoritative enemy-visibility training — 2026-09-16

## Teacher correction

ViZDoom labels are used only while generating teacher actions. The deployed policy receives RGB plus motion. Enemy geometry now records player-relative distance, screen-centre error, mutual line of sight and whether the enemy is facing the player.

The original teacher produced an attack-heavy dataset. A controlled firing burst followed by alternating lateral evasion reduced that shortcut while retaining strong teacher performance.

### Balanced teacher probe

- Episodes: 20
- Mean kills: 5.4
- Attack labels: 952
- Strafe-left labels: 950
- Strafe-right labels: 916
- Forward labels: 894
- Turn-right labels: 397
- Turn-left labels: 394

## Candidate admission tests

| Candidate | Decode | Episodes | Mean kills | Mean reward | Decision |
| --- | --- | ---: | ---: | ---: | --- |
| Released `deadly-dagger.pt` | Greedy, 600 s | 91 | 0.4615 | -102.8794 | Baseline |
| Attack-heavy visibility imitation | Greedy, 60 s | 23 | 0.0 | -111.5440 | Rejected |
| Attack-heavy visibility DAgger | Greedy, 60 s | 24 | 0.0 | -111.7051 | Rejected |
| Balanced visibility imitation | Greedy, 60 s | 29 | 0.9310 | 218.1720 | Current candidate |
| Balanced visibility DAgger | Greedy, 60 s | 25 | 0.0 | -80.1685 | Rejected |
| Balanced visibility imitation | Sampled, 60 s | 25 | 0.2 | -63.7613 | Rejected decode |

The balanced imitation checkpoint is the only candidate that improved both kills and reward over the released baseline. It still over-selects forward movement under greedy decoding and therefore remains an experimental candidate, not a generally capable Doom policy.

