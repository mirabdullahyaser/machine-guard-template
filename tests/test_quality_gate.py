"""Unit tests for the quality gate: a good model passes, a useless one is stopped."""
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier

from machine_guard.quality_gate import RECALL_GATE, evaluate, passes_gate


def test_gate_is_the_contract_value():
    assert RECALL_GATE == 0.75


@pytest.mark.parametrize("recall, expected", [(0.84, True), (0.75, True), (0.7499, False), (0.0, False)])
def test_passes_gate_compares_recall_with_the_gate(recall, expected):
    assert passes_gate({"recall": recall}) is expected


def test_always_no_failure_model_fails_the_gate(sample):
    X, y = sample.drop(columns="machine_failure"), sample["machine_failure"]
    lazy = DummyClassifier(strategy="most_frequent").fit(X, y)
    metrics = evaluate(lazy, X, y)
    # TODO (Lab 1, step 4): three asserts. Recall is 0, accuracy is above 0.9 (the accuracy trap),
    # and the model does not pass the gate.
    pytest.skip("TODO: write the asserts")


def test_cost_uses_the_20_to_1_ratio():
    class Fixed:
        def predict_proba(self, X):
            return pd.DataFrame({"no": 1 - X["p"], "yes": X["p"]}).to_numpy()

    X = pd.DataFrame({"p": [0.9, 0.1, 0.9, 0.1]})
    y = pd.Series([1, 1, 0, 0])            # one caught, one missed, one false alarm, one quiet
    m = evaluate(Fixed(), X, y)
    assert (m["caught_failures"], m["missed_failures"], m["false_alarms"]) == (1, 1, 1)
    assert m["cost"] == 1 * 20 + 1 * 1
