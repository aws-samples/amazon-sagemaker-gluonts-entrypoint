# python entrypoint.py --freq M --prediction_length 5 --distr_output gluonts.distribution.gaussian.GaussianOutput --use_feat_static_cat True --cardinality '[5]'
# python entrypoint.py --algo gluonts.model.deepar.DeepAREstimator --freq M --prediction_length 5 --distr_output gluonts.distribution.gaussian.GaussianOutput --use_feat_static_cat True --cardinality '[5]'
# python entrypoint.py --algo gluonts.model.deepstate.DeepStateEstimator --freq M --prediction_length 5 --use_feat_static_cat True --cardinality '[5]' --noise_std_bounds gluonts.distribution.lds.ParameterBounds --noise_std_bounds.lower 1e-5 --noise_std_bounds.upper 1e-1

# Based on glounts/nursery/sagemaker_sdk/entrypoint_scripts/train_entry_point.py


import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict

from gluonts.core import serde
from gluonts.dataset import common
from gluonts.dataset.repository import datasets
from gluonts.evaluation import Evaluator, backtest
from gluonts.model.deepar import DeepAREstimator
from gluonts.model.deepstate import DeepStateEstimator

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s %(message)s", datefmt="[%Y-%m-%d %H:%M:%S]",
)
logger = logging.getLogger(__name__)

# TODO: implement model_fn, input_fn, predict_fn, and output_fn !!
# TODO: segment script for readability


def klass_dict(klass: str, args=[], kwargs={}):
    return {"__kind__": "instance", "class": klass, "args": args.copy(), "kwargs": kwargs.copy()}


def get_kwargs(hp: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    prefix += "."
    prefix_len = len(prefix)
    kwargs = {k[prefix_len:]: v for k, v in hp.items() if k.startswith(prefix)}
    return kwargs


def deepar(hp: Dict[str, Any]):
    kwargs = {k: v for k, v in hp.items() if "." not in k}
    for argname in ["trainer", "distr_output"]:
        if argname in kwargs:
            klass_name = kwargs[argname]
            kwargs[argname] = serde.decode(klass_dict(klass_name, [], get_kwargs(hp, argname)))
    return DeepAREstimator(**kwargs)


def deepstate(hp: Dict[str, Any]):
    kwargs = {k: v for k, v in hp.items() if "." not in k}
    for argname in ["trainer", "issm", "noise_std_bounds", "prior_cov_bounds", "innovation_bounds"]:
        if argname in kwargs:
            klass_name = kwargs[argname]
            kwargs[argname] = serde.decode(klass_dict(klass_name, [], get_kwargs(hp, argname)))
    return DeepStateEstimator(**kwargs)


def deser_algo_args(hp: Dict[str, Any], deser_list=[]):
    kwargs = {k: v for k, v in hp.items() if "." not in k}
    for argname in deser_list:
        if argname in kwargs:
            klass_name = kwargs[argname]
            # kwargs[argname] = serde.decode(klass_dict(klass_name, [], get_kwargs(hp, argname)))
            kwargs[argname] = klass_dict(klass_name, [], get_kwargs(hp, argname))
    return kwargs


deser_args = {
    "gluonts.model.deepar.DeepAREstimator": ["trainer", "distr_output"],
    "gluonts.model.deepstate.DeepStateEstimator": [
        "trainer",
        "issm",
        "noise_std_bounds",
        "prior_cov_bounds",
        "innovation_bounds",
    ],
}


def train(args, algo_args):
    """
    Generic train method that trains a specified estimator on a specified dataset.
    """
    estimator_config = klass_dict(args.algo, [], deser_algo_args(algo_args, deser_args[args.algo]))
    estimator = serde.decode(estimator_config)
    print(estimator)
    import sys

    sys.exit(0)
    logger.info("Downloading dataset.")
    if args.s3_dataset is None:
        # load built in dataset
        dataset = datasets.get_dataset(args.dataset)
    else:
        # load custom dataset
        s3_dataset_dir = Path(args.s3_dataset)
        dataset = common.load_datasets(
            metadata=s3_dataset_dir, train=s3_dataset_dir / "train", test=s3_dataset_dir / "test",
        )

    logger.info("Starting model training.")
    predictor = estimator.train(dataset.train)
    forecast_it, ts_it = backtest.make_evaluation_predictions(
        dataset=dataset.test, predictor=predictor, num_samples=int(args.num_samples),
    )

    logger.info("Starting model evaluation.")
    evaluator = Evaluator(quantiles=eval(args.quantiles))

    agg_metrics, item_metrics = evaluator(ts_it, forecast_it, num_series=len(dataset.test))

    # required for metric tracking.
    for name, value in agg_metrics.items():
        logger.info(f"gluonts[metric-{name}]: {value}")

    # save the evaluation results
    metrics_output_dir = Path(args.output_data_dir)
    with open(metrics_output_dir / "agg_metrics.json", "w") as f:
        json.dump(agg_metrics, f)
    with open(metrics_output_dir / "item_metrics.csv", "w") as f:
        item_metrics.to_csv(f, index=False)

    # save the model
    model_output_dir = Path(args.model_dir)
    predictor.serialize(model_output_dir)


def parse_hyperparameters(hm) -> Dict[str, Any]:
    """Convert list of ['--name', 'value', ...] to { 'name': value}, where 'value' is converted to the nearest data type.

    Conversion follows the principle: "if it looks like a duck and quacks like a duck, then it must be a duck".
    """
    d = {}
    it = iter(hm)
    try:
        while True:
            key = next(it)[2:]
            value = next(it)
            d[key] = value
    except StopIteration:
        pass

    # Infer data types.
    dd = {k: infer_dtype(v) for k, v in d.items()}
    return dd


def infer_dtype(s):
    """Auto-cast string values to nearest matching datatype.

    Conversion follows the principle: "if it looks like a duck and quacks like a duck, then it must be a duck".
    Note that python 3.6 implements PEP-515 which allows '_' as thousand separators. Hence, on Python 3.6,
    '1_000' is a valid number and will be converted accordingly.
    """
    if s == "None":
        return None
    if s == "True":
        return True
    if s == "False":
        return False

    try:
        i = float(s)
        if ("." in s) or ("e" in s.lower()):
            return i
        else:
            return int(s)
    except:  # noqa:E722
        pass

    try:
        # If string is json, deser it.
        return json.loads(s)
    except:  # noqa:E722
        return s


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # an alternative way to load hyperparameters via SM_HPS environment variable.
    parser.add_argument("--sm-hps", type=json.loads, default=os.environ.get("SM_HPS", {}))

    # input data, output dir and model directories
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR", "model"))
    parser.add_argument("--output-data_dir", type=str, default=os.environ.get("SM_OUTPUT_DATA_DIR", "output"))
    # parser.add_argument("--input-dir", type=str, default=os.environ.get("SM_INPUT_DIR", "input"))
    parser.add_argument("--s3-dataset", type=str, default=os.environ.get("SM_CHANNEL_S3-DATASET", None))
    parser.add_argument("--dataset", type=str, default=os.environ.get("SM_HP_DATASET", ""))
    parser.add_argument("--num-samples", type=int, default=os.environ.get("SM_HP_NUM_SAMPLES", 100))
    parser.add_argument(
        "--quantiles",
        type=str,
        default=os.environ.get("SM_HP_QUANTILES", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]),
    )
    parser.add_argument(
        "--algo", type=str, default=os.environ.get("SM_HP_ALGO", "gluonts.model.deepar.DeepAREstimator"),
    )

    args, train_args = parser.parse_known_args()
    algo_args = parse_hyperparameters(train_args)
    train(args, algo_args)
