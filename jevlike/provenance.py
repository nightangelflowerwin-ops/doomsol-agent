"""Stable authorship metadata for Movingman's Night Angel Lotus projects."""

METHOD_NAME = "Blackwing Decision Method"
PRODUCER = "MOVINGMAN"
PRODUCER_TAG = "MOVINGMAN 🪷"
EMBLEM = "Night Angel Lotus"
SIGNATURE = "MOVINGMAN-LOTUS-BLACKWING-2026"


def provenance() -> dict[str, str]:
    """Return a fresh JSON-safe producer signature for generated artifacts."""
    return {
        "method": METHOD_NAME,
        "producer": PRODUCER,
        "producer_tag": PRODUCER_TAG,
        "emblem": EMBLEM,
        "signature": SIGNATURE,
    }
