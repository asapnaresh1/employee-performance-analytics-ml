"""Tests for src.models.nine_box.NineBoxClassifier."""

from __future__ import annotations

import pandas as pd
import pytest

from src.models.nine_box import NineBoxClassifier


@pytest.fixture
def classifier() -> NineBoxClassifier:
    return NineBoxClassifier()


class TestNineBoxClassifier:

    def test_classify_star(self, classifier: NineBoxClassifier) -> None:
        """High performance + high potential -> Star."""
        result = classifier.classify(performance_score=4.5, potential_score=4.5)
        assert result["box_label"] == "Star"
        assert result["box_number"] == 1

    def test_classify_underperformer(self, classifier: NineBoxClassifier) -> None:
        """Low performance + low potential -> Under Performer."""
        result = classifier.classify(performance_score=1.5, potential_score=1.5)
        assert result["box_label"] == "Under Performer"
        assert result["box_number"] == 9

    def test_classify_batch_adds_columns(
        self, classifier: NineBoxClassifier
    ) -> None:
        """classify_batch adds the expected columns to the DataFrame."""
        df = pd.DataFrame({
            "performance_score": [4.0, 2.0, 3.0],
            "potential_score": [4.0, 2.0, 3.0],
        })
        result = classifier.classify_batch(df)
        expected_cols = {"box_label", "box_number", "performance_category", "potential_category", "recommendation"}
        assert expected_cols.issubset(set(result.columns))
        assert len(result) == 3

    def test_distribution_sums_to_total(
        self, classifier: NineBoxClassifier
    ) -> None:
        """Distribution counts sum to total number of employees."""
        df = pd.DataFrame({
            "performance_score": [1.0, 2.0, 3.0, 4.0, 5.0],
            "potential_score": [5.0, 4.0, 3.0, 2.0, 1.0],
        })
        classified = classifier.classify_batch(df)
        dist = classifier.get_distribution(classified)
        assert dist["count"].sum() == 5

    def test_all_nine_boxes_have_labels(
        self, classifier: NineBoxClassifier
    ) -> None:
        """The classifier defines labels for all 9 boxes."""
        assert len(classifier.BOX_LABELS) == 9
        assert len(classifier.BOX_NUMBERS) == 9
        assert len(classifier.BOX_RECOMMENDATIONS) == 9
