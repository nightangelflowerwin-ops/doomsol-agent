from jevlike.provenance import EMBLEM, METHOD_NAME, PRODUCER_TAG, SIGNATURE, provenance


def test_blackwing_provenance_is_stable_and_fresh():
    first = provenance()
    second = provenance()
    assert first == {
        "method": METHOD_NAME,
        "producer": "MOVINGMAN",
        "producer_tag": PRODUCER_TAG,
        "emblem": EMBLEM,
        "signature": SIGNATURE,
    }
    assert first is not second
