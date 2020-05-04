import io
import json
from typing import List

import pytest
from gluonts.dataset.common import DataEntry
from gluonts.model.forecast import Config
from gluonts.model.predictor import Predictor

import entrypoint


@pytest.fixture
def predictor():
    predictor = entrypoint.model_fn("test/refdata/model")
    assert isinstance(predictor, Predictor)
    return predictor


@pytest.fixture
def input_ts():
    request_body = """{"start": "2019-09-29", "target": [128, 57, 0, 29, 10, 64], "feat_static_cat": [0], "item_id": "cat:ts1|name:AB"}
{"start": "2019-10-01", "target": [256, 125, 150, 127, 20, 205], "feat_static_cat": [1], "item_id": "cat:ts2|name:EF"}
"""
    ts = entrypoint.input_fn(request_body, "request_content_type_ignored_by_input_fn")
    expected = [json.loads(line) for line in io.StringIO(request_body)]
    assert ts == expected
    return ts


@pytest.mark.parametrize("num_samples,quantiles", [(5, ["0.4", "0.6"]), (25, ["0.2", "0.5", "0.8"])])
def test_inference(predictor: Predictor, input_ts: List[DataEntry], num_samples: int, quantiles: List[str]):
    # Somewhat equivalent to predictor.predict(input_ts).
    results = entrypoint.predict_fn(input_ts, predictor, num_samples=num_samples)

    # Verify forecast paths.
    for result in results:
        assert result.samples.shape == (num_samples, predictor.prediction_length)

    # Serialize forecast results to JSON lines
    results_str = entrypoint.output_fn(results, Config(quantiles=quantiles))

    # Make sure each JSON line is somewhat correct.
    for line in io.StringIO(results_str):
        d = json.loads(line)
        for quantile in quantiles:
            assert quantile in d["quantiles"]
            assert len(d["quantiles"][quantile]) == predictor.prediction_length
        assert ("mean" in d) and (len(d["mean"]) == predictor.prediction_length)

    # Print results; need to run pytest -v -rA --tb=short ...
    print(results_str)
    return results_str
