"""The learned regional term and the benchmark's scribbles and scoring."""

import numpy as np
import pytest

from learned.pets import (
    LABEL_BACKGROUND,
    LABEL_FOREGROUND,
    LABEL_IGNORE,
    foreground_iou,
    pixel_accuracy,
    scribble_rng,
    simulate_scribbles,
)
from learned.unary import build_learned_capacities
from maxflow.graph_builder import BACKGROUND, FOREGROUND, UNMARKED
from maxflow.grid_maxflow import solve_grid_min_cut


def _label(size=64):
    label = np.full((size, size), LABEL_BACKGROUND, dtype=np.uint8)
    label[20:44, 20:44] = LABEL_FOREGROUND
    label[18:20, 18:46] = LABEL_IGNORE
    return label


def _flat_image(size=24):
    return np.full((size, size, 3), 128, dtype=np.uint8)


def test_scribbles_stay_inside_their_true_regions():
    label = _label()
    scribbles = simulate_scribbles(label, scribble_rng("cat_1"))
    assert np.all(label[scribbles == FOREGROUND] == LABEL_FOREGROUND)
    assert np.all(label[scribbles == BACKGROUND] == LABEL_BACKGROUND)


def test_scribbles_mark_both_labels():
    scribbles = simulate_scribbles(_label(), scribble_rng("dog_7"))
    assert (scribbles == FOREGROUND).any() and (scribbles == BACKGROUND).any()


def test_scribbles_never_touch_the_border_band():
    label = _label()
    scribbles = simulate_scribbles(label, scribble_rng("dog_9"))
    assert np.all(scribbles[label == LABEL_IGNORE] == UNMARKED)


def test_scribbles_are_the_same_for_the_same_image():
    label = _label()
    first = simulate_scribbles(label, scribble_rng("cat_2"))
    second = simulate_scribbles(label, scribble_rng("cat_2"))
    assert np.array_equal(first, second)


def test_scribbles_differ_between_images():
    label = _label()
    assert not np.array_equal(
        simulate_scribbles(label, scribble_rng("cat_2")),
        simulate_scribbles(label, scribble_rng("cat_3")),
    )


def test_small_subject_still_gets_a_scribble():
    label = np.full((64, 64), LABEL_BACKGROUND, dtype=np.uint8)
    label[30:33, 30:33] = LABEL_FOREGROUND  # gone after erosion
    assert (simulate_scribbles(label, scribble_rng("tiny")) == FOREGROUND).any()


def test_iou_is_one_for_a_perfect_mask():
    label = _label()
    assert foreground_iou(label == LABEL_FOREGROUND, label) == 1.0


def test_iou_is_zero_for_a_disjoint_mask():
    label = _label()
    prediction = np.zeros(label.shape, dtype=bool)
    prediction[:5, :5] = True
    assert foreground_iou(prediction, label) == 0.0


def test_iou_ignores_the_border_band():
    label = _label()
    prediction = label == LABEL_FOREGROUND
    prediction[label == LABEL_IGNORE] = True
    assert foreground_iou(prediction, label) == 1.0


def test_accuracy_ignores_the_border_band():
    label = _label()
    prediction = label == LABEL_FOREGROUND
    prediction[label == LABEL_IGNORE] = True
    assert pixel_accuracy(prediction, label) == 1.0


def test_confident_foreground_costs_more_to_cut_from_the_source():
    probability = np.full((8, 8), 0.99)
    source, sink, _, _ = build_learned_capacities(
        _flat_image(8), np.zeros((8, 8), dtype=np.uint8), probability
    )
    assert np.all(source > sink)


def test_certain_probabilities_stay_finite():
    probability = np.zeros((8, 8))
    probability[:4] = 1.0
    source, sink, _, _ = build_learned_capacities(
        _flat_image(8), np.zeros((8, 8), dtype=np.uint8), probability
    )
    assert np.isfinite(source).all() and np.isfinite(sink).all()


def test_input_probability_is_not_modified():
    probability = np.full((8, 8), 0.3)
    scribbles = np.zeros((8, 8), dtype=np.uint8)
    scribbles[0, 0] = FOREGROUND
    build_learned_capacities(_flat_image(8), scribbles, probability)
    assert np.all(probability == 0.3)


def test_the_cut_follows_the_model():
    probability = np.full((24, 24), 0.05)
    probability[6:18, 6:18] = 0.95
    scribbles = np.zeros((24, 24), dtype=np.uint8)
    capacities = build_learned_capacities(_flat_image(), scribbles, probability, n_link_weight=0.1)
    _, mask = solve_grid_min_cut(*capacities)
    assert mask[12, 12] and not mask[1, 1]


def test_a_scribble_beats_a_confident_model():
    probability = np.full((24, 24), 0.999)
    scribbles = np.zeros((24, 24), dtype=np.uint8)
    scribbles[0, 0] = BACKGROUND
    capacities = build_learned_capacities(_flat_image(), scribbles, probability, n_link_weight=1.0)
    _, mask = solve_grid_min_cut(*capacities)
    assert not mask[0, 0]


def test_from_scratch_and_fast_solvers_agree_on_learned_costs():
    from maxflow.fast_maxflow import is_available, solve_grid_min_cut_fast

    if not is_available():
        pytest.skip("PyMaxflow is not installed")

    rng = np.random.default_rng(5)
    image = rng.integers(0, 255, (20, 20, 3), dtype=np.uint8)
    probability = rng.random((20, 20))
    scribbles = np.zeros((20, 20), dtype=np.uint8)
    scribbles[0, 0] = BACKGROUND
    scribbles[10, 10] = FOREGROUND
    capacities = build_learned_capacities(image, scribbles, probability, n_link_weight=2.0)
    assert solve_grid_min_cut(*capacities)[0] == pytest.approx(solve_grid_min_cut_fast(*capacities)[0], rel=1e-4)


def test_unet_returns_one_logit_per_pixel():
    torch = pytest.importorskip("torch")
    from learned.unet import SmallUNet

    output = SmallUNet(base=4)(torch.zeros(2, 3, 32, 32))
    assert output.shape == (2, 1, 32, 32)
