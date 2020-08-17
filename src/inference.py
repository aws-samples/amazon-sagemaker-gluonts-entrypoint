import smepu

import io
import json
import os
import warnings
from pathlib import Path
from typing import List, Tuple, Union

import matplotlib.cbook
import numpy as np
from gluonts.dataset.common import DataEntry, ListDataset
from gluonts.model.forecast import Config, Forecast
from gluonts.model.predictor import Predictor

from gluonts_example.util import clip_to_zero, expm1_and_clip_to_zero, log1p

warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)

# Setup logger must be done in the entrypoint script.
logger = smepu.setup_opinionated_logger(__name__)


def model_fn(model_dir: Union[str, Path]) -> Predictor:
    """Load a glounts model from a directory.

    Args:
        model_dir (Union[str, Path]): a directory where model is saved.

    Returns:
        Predictor: A gluonts predictor.
    """
    predictor = Predictor.deserialize(Path(model_dir))

    # If model was trained on log-space, then forecast must be inverted before metrics etc.
    with open(os.path.join(model_dir, "y_transform.json"), "r") as f:
        y_transform = json.load(f)
        logger.info("model_fn: custom transformations = %s", y_transform)

        if y_transform["inverse_transform"] == "expm1":
            predictor.output_transform = expm1_and_clip_to_zero
        else:
            predictor.output_transform = clip_to_zero

        # Custom field
        predictor.pre_input_transform = log1p if y_transform["transform"] == "log1p" else None

    logger.info("predictor.pre_input_transform: %s", predictor.pre_input_transform)
    logger.info("predictor.output_transform: %s", predictor.output_transform)
    logger.info("model_fn() done; loaded predictor %s", predictor)

    return predictor


def transform_fn(
    model: Predictor,
    request_body: Union[str, bytes],
    content_type: str = "application/json",
    accept_type: str = "application/json",
    num_samples: int = 1000,
) -> Union[bytes, Tuple[bytes, str]]:
    # See https://sagemaker.readthedocs.io/en/stable/using_mxnet.html#use-transform-fn
    #
    # [As of this writing on 20200506]
    # Looking at sagemaker_mxnet_serving_container/handler_service.py [1], it turns out I must use transform_fn()
    # because my gluonts predictor is neither mx.module.BaseModule nor mx.gluon.block.Block.
    #
    # I suppose the model_fn documentation [2] can be updated also, to make it clear that if entrypoint does not use
    # transform_fn(), then model_fn() must returns object with similar type to what the default implementation does.
    #
    # [1] https://github.com/aws/sagemaker-mxnet-serving-container/blob/406c1f387d9800ed264b538bdbf9a30de68b6977/src/sagemaker_mxnet_serving_container/handler_service.py
    # [2] https://sagemaker.readthedocs.io/en/stable/using_mxnet.html#load-a-model
    deser_input: List[DataEntry] = _input_fn(request_body, content_type)
    fcast: List[Forecast] = _predict_fn(deser_input, model, num_samples=num_samples)
    ser_output: Union[bytes, Tuple[bytes, str]] = _output_fn(fcast, accept_type)
    return ser_output


# Because we use transform_fn(), make sure this entrypoint does not contain input_fn() during inference.
def _input_fn(request_body: Union[str, bytes], request_content_type: str = "application/json") -> List[DataEntry]:
    """Deserialize JSON-lines into Python objects.

    Args:
        request_body (str): Incoming payload.
        request_content_type (str, optional): Ignored. Defaults to "".

    Returns:
        List[DataEntry]: List of gluonts timeseries.
    """

    # [20200508] I swear: two days ago request_body was bytes, today's string!!!
    if isinstance(request_body, bytes):
        request_body = request_body.decode("utf-8")
    return [json.loads(line) for line in io.StringIO(request_body)]


# Because we use transform_fn(), make sure this entrypoint does not contain predict_fn() during inference.
def _predict_fn(input_object: List[DataEntry], model: Predictor, num_samples=1000) -> List[Forecast]:
    """Take the deserialized JSON-lines, then perform inference against the loaded model.

    Args:
        input_object (List[DataEntry]): List of gluonts timeseries.
        model (Predictor): A gluonts predictor.
        num_samples (int, optional): Number of forecast paths for each timeseries. Defaults to 1000.

    Returns:
        List[Forecast]: List of forecast results.
    """
    # Create ListDataset here, because we need to match their freq with model's freq.
    X = ListDataset(input_object, freq=model.freq)

    # Apply forward transformation to input data, before injecting it to the predictor.
    if model.pre_input_transform is not None:
        logger.debug("Before model.pre_input_transform: %s", X.list_data)
        model.pre_input_transform(X)
        logger.debug("After model.pre_input_transform: %s", X.list_data)

    it = model.predict(X, num_samples=num_samples)
    return list(it)


# Because we use transform_fn(), make sure this entrypoint does not contain output_fn() during inference.
def _output_fn(
    forecasts: List[Forecast],
    content_type: str = "application/json",
    config: Config = Config(quantiles=["0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9"]),
) -> Union[bytes, Tuple[bytes, str]]:
    """Take the prediction result and serializes it according to the response content type.

    Args:
        prediction (List[Forecast]): List of forecast results.
        content_type (str, optional): Ignored. Defaults to "".

    Returns:
        List[str]: List of JSON-lines, each denotes forecast results in quantiles.
    """

    # jsonify_floats is taken from gluonts/shell/serve/util.py
    #
    # The module depends on flask, and we may not want to import when testing in our own dev env.
    def jsonify_floats(json_object):
        """Traverse through the JSON object and converts non JSON-spec compliant floats(nan, -inf, inf) to string.

        Parameters
        ----------
        json_object
            JSON object
        """
        if isinstance(json_object, dict):
            return {k: jsonify_floats(v) for k, v in json_object.items()}
        elif isinstance(json_object, list):
            return [jsonify_floats(item) for item in json_object]
        elif isinstance(json_object, float):
            if np.isnan(json_object):
                return "NaN"
            elif np.isposinf(json_object):
                return "Infinity"
            elif np.isneginf(json_object):
                return "-Infinity"
            return json_object
        return json_object

    str_results = "\n".join((json.dumps(jsonify_floats(forecast.as_json_dict(config))) for forecast in forecasts))
    bytes_results = str.encode(str_results)
    return bytes_results, content_type
