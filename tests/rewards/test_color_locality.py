"""A failed protected region cannot be averaged away by correct regions."""

from PIL import Image

from vrl.rewards.models.color_locality import color_locality, combine_locality


def test_worst_region_and_calibrated_quality_bound_prevent_compensation():
    config = {
        "color_hue_degrees": {"blue": 240},
        "hue_sigma_degrees": 25,
        "saturation_start": 0.15,
        "saturation_full": 0.5,
    }
    rule = {
        "color": "blue",
        "target": [[0, 0, 1 / 3, 1]],
        "protected": {"back": [1 / 3, 0, 2 / 3, 1], "rack": [2 / 3, 0, 1, 1]},
    }
    source = Image.new("RGB", (30, 10), "grey")
    correct = source.copy()
    correct.paste("blue", (0, 0, 10, 10))
    wrong_back, wrong_rack = correct.copy(), correct.copy()
    wrong_back.paste("blue", (10, 0, 20, 10))
    wrong_rack.paste("blue", (20, 0, 30, 10))
    good = color_locality(source, correct, rule, config)
    bad = [color_locality(source, image, rule, config) for image in (wrong_back, wrong_rack)]
    assert good["locality"] > 0.99
    assert bad[0]["preservation_rack"] == bad[1]["preservation_back"] == 1
    assert all(row["locality"] < 0.01 for row in bad)
    weight = (good["locality"] - max(row["locality"] for row in bad)) / 4
    lowest_good = combine_locality(good, -100, quality_weight=weight)["editreward_locality"]
    assert all(
        lowest_good > combine_locality(row, 100, quality_weight=weight)["editreward_locality"]
        for row in bad
    )


def test_existing_target_color_in_protected_source_is_not_new_leakage():
    config = {
        "color_hue_degrees": {"blue": 240},
        "hue_sigma_degrees": 25,
        "saturation_start": 0.15,
        "saturation_full": 0.5,
    }
    rule = {
        "color": "blue",
        "target": [[0, 0, 0.5, 1]],
        "protected": {"existing_blue_object": [0.5, 0, 1, 1]},
    }
    source = Image.new("RGB", (20, 10), "grey")
    source.paste("blue", (10, 0, 20, 10))
    edited = Image.new("RGB", (20, 10), "blue")
    scores = color_locality(source, edited, rule, config)
    assert scores["worst_preservation"] == 1
    assert scores["locality"] > 0.99
