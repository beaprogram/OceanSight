from oceansight.statistics import bootstrap_counts


def test_cluster_bootstrap_is_reproducible_and_bounded():
    rows = [
        {"video": "a", "tp": 5, "fp": 1, "fn": 2},
        {"video": "a", "tp": 2, "fp": 1, "fn": 0},
        {"video": "b", "tp": 3, "fp": 0, "fn": 1},
    ]
    result = bootstrap_counts(rows, repetitions=100)
    assert result == bootstrap_counts(rows, repetitions=100)
    assert result["video_groups"] == 2
    for key in ["precision_95_percentile_interval", "recall_95_percentile_interval"]:
        low, high = result[key]
        assert 0 <= low <= high <= 1
