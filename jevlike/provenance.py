"""Stable authorship metadata for Nightangel Flowerwin projects."""

METHOD_NAME = "Blackwing Decision Method"
PRODUCER = "nightangelflowerwin-ops"
PRODUCER_TAG = "NIGHTANGEL FLOWERWIN // BLACKWING"
SIGNATURE = "NAFW-BLACKWING-2026"


def provenance() -> dict[str, str]:
    """Return a fresh JSON-safe producer signature for generated artifacts."""
    return {
        "method": METHOD_NAME,
        "producer": PRODUCER,
        "producer_tag": PRODUCER_TAG,
        "signature": SIGNATURE,
    }
