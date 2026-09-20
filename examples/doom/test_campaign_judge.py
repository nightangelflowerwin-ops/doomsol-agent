import json

from judge_campaign_trace import cross_attempt_evidence


def write_trace(path, sectors):
    path.write_text("".join(
        json.dumps({"current_sector": sector}) + "\n" for sector in sectors
    ), encoding="utf-8")


def test_cross_attempt_judge_detects_identical_failed_route(tmp_path):
    prior = tmp_path / "prior.jsonl"
    write_trace(prior, [1, 1, 2, 2, 3])
    rows = [{"current_sector": sector} for sector in [1, 1, 2, 2, 3]]

    evidence = cross_attempt_evidence(rows, [prior])

    assert evidence["closest_prior_attempt"]["directed_edge_jaccard"] == 1.0
    assert not evidence["new_meaningful_sectors"]
    assert not evidence["new_directed_portals"]
