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

## Ten-minute visible candidate run

- Game time: 600 seconds
- Episodes: 277
- Decisions: 5,250
- Mean reward: 234.3164
- Mean reported kills: 1.1697
- Maximum running reward: 858.6039
- Mean inference latency: 6.8124 ms
- Move-forward decisions: 4,765
- Strafe-right decisions: 485
- Attack decisions: 0
- All other decisions: 0

This candidate is **not promoted**. Its higher reported kills cannot demonstrate learned combat because it never selected attack. Enemy infighting or other scenario mechanics can raise outcome counters while the agent merely moves. Future combat admission must require actual attack decisions, hits attributable to the player, and enemy-conditioned changes under shuffled-frame controls—not kills or reward alone.

Local artifacts:

- `runs/visibility-agent-visible-10min.mp4`: 600 seconds, 441,998,747 bytes, SHA-256 `94C83B6235E124FB988BC03BD1E0082BCAD9088DC936CFEF646982EF2888CD06`
- `runs/visibility-agent-visible-10min-trace.json`: 649,948,240 bytes, SHA-256 `6BF98F49E99F108295D35B8EB7F1B43099323B9EEBE56F4B6842D3D31B95DB47`
