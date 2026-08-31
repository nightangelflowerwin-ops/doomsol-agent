# Agent layer

`agent.py --mode discover` inspects only non-sensitive page metadata through the already-attached tab and does not press gameplay keys.

`agent.py --mode scout --seconds 20` captures frames without gameplay input.

`agent.py --mode baseline --seconds 20` runs a crude benchmark controller and records frames/actions. It is deliberately not presented as a learned policy; it gives us a baseline for later improvement.

## Local enemy-recognition pretraining

`extract_wad_enemy_sprites.py` extracts living and attacking enemy rotations
from a Doom-format WAD while excluding death animations. The checked-in dataset
contains 349 sprite views across ten enemy families.

`train_threat_detector.py` combines those sprites with the reviewed DoomSol
enemy crops, gameplay backgrounds, and mined hard negatives. The compact
checkpoint and its report are stored in `models/doom_threat_detector_v1`.

The local WAD imagery is pretraining data. Promotion still requires a bounded
DoomSol evaluation measuring confirmed targets, false positives, aim turns,
firing, kills, and survival.
