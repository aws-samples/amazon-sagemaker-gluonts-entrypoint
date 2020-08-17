import io
import json
from typing import List

import pytest
from gluonts.model.predictor import Predictor


@pytest.fixture(autouse=True)
def gluonts_inference(root_dir, monkeypatch, helpers):
    """Import inference script as a module.

    See https://github.com/verdimrc/python-project-skeleton/tree/master/test for
    the reason behind importing this way (as opposed import statement.
    """
    monkeypatch.syspath_prepend(root_dir / "src")
    inference = helpers.import_from_file("gluonts_inference", root_dir / "src" / "inference.py")
    return inference


@pytest.fixture
def predictor(gluonts_inference):
    predictor = gluonts_inference.model_fn("test/refdata/model")
    assert isinstance(predictor, Predictor)
    return predictor


@pytest.fixture
def request_body():
    return b"""{"start": "2019-09-29", "target": [128, 57, 0, 29, 10, 64], "feat_static_cat": [0], "item_id": "cat:ts1|name:AB"}
{"start": "2019-10-01", "target": [256, 125, 150, 127, 20, 205], "feat_static_cat": [1], "item_id": "cat:ts2|name:EF"}
"""


@pytest.mark.parametrize("num_samples,quantiles", [(5, ["0.4", "0.6"]), (25, ["0.2", "0.5", "0.8"])])
def test_transform_fn(
    gluonts_inference, predictor: Predictor, request_body: bytes, num_samples: int, quantiles: List[str]
):
    results_bytes, accept_type = gluonts_inference.transform_fn(
        predictor, request_body, "application/json", "application/json", 3
    )

    # Make sure each JSON line is somewhat correct.
    results_str = results_bytes.decode("utf-8")
    for line in io.StringIO(results_str):
        d = json.loads(line)
        for quantile in quantiles:
            assert quantile in d["quantiles"]
            assert len(d["quantiles"][quantile]) == predictor.prediction_length
        assert ("mean" in d) and (len(d["mean"]) == predictor.prediction_length)

    # Print results; need to run pytest -v -rA --tb=short ...
    print(results_str)
    return results_str


@pytest.mark.parametrize("num_samples,quantiles", [(5, ["0.4", "0.6"]), (25, ["0.2", "0.5", "0.8"])])
def test_predict_fn(
    gluonts_inference, predictor: Predictor, request_body: bytes, num_samples: int, quantiles: List[str]
):
    input_ts = gluonts_inference._input_fn(request_body, "application/json")
    results = gluonts_inference._predict_fn(input_ts, predictor, num_samples=num_samples)

    # Verify forecast paths.
    for result in results:
        assert result.samples.shape == (num_samples, predictor.prediction_length)
        print(result.samples.shape)
