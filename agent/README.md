# Agent layer

`agent.py --mode discover` inspects only non-sensitive page metadata through the already-attached tab and does not press gameplay keys.

`agent.py --mode scout --seconds 20` captures frames without gameplay input.

`agent.py --mode baseline --seconds 20` runs a crude benchmark controller and records frames/actions. It is deliberately not presented as a learned policy; it gives us a baseline for later improvement.
