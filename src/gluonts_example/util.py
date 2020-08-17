import logging
import os
import warnings
from pathlib import Path
from typing import Any, Dict, Union

import numpy as np
from gluonts.dataset.common import ListDataset, MetaData, TrainDatasets


def mkdir(path: Union[str, os.PathLike]):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def hp2estimator(algo: str, algo_args: Dict[str, Any], metadata: MetaData) -> Any:
    algo_args = merge_metadata_hp(algo_args, metadata)
    # estimator_config = klass_dict(algo, [], deser_algo_args(algo_args, deser_args[algo]))
    # estimator = serde.decode(estimator_config)
    # return estimator
    raise NotImplementedError


def merge_metadata_hp(hp: Dict[str, Any], metadata: MetaData) -> Dict[str, Any]:
    """Resolve values to inject to the estimator: is it the hp or the one from metadata.

    This function:
    - mitigates errors made by callers when inadvertantly specifies hyperparameters that shouldn't be done, e.g.,
      the frequency should follow how the data prepared.
    - uses some metadata values as defaults, unless stated otherwise by the hyperparameters.
    """
    hp = hp.copy()

    # Always use freq from dataset.
    if "freq" in hp and hp["freq"] != metadata.freq:
        freq_hp = hp["freq"]
        logging.info(f"freq: set freq='{metadata.freq}' from metadata; ignore '{freq_hp}' from hyperparam.")
    hp["freq"] = metadata.freq

    # Use prediction_length hyperparameters, but if not specified then fallbacks/defaults to the one from metadata.
    if "prediction_length" not in hp:
        hp["prediction_length"] = metadata.prediction_length
        logging.info(
            "prediction_length: no hyperparam, so set " f"prediction_length={metadata.prediction_length} from metadata"
        )

    # TODO: autoprobe cardinalities.
    warnings.warn("This implementation still ignores cardinality and static features in the metadata", RuntimeWarning)

    return hp


################################################################
# Data transformations
################################


def log1p_tds(dataset: TrainDatasets) -> TrainDatasets:
    """Create a new train datasets with targets log-transformed."""
    # Implementation note: currently, the only way is to eagerly load all timeseries in memory, and do the transform.
    train = ListDataset(dataset.train, freq=dataset.metadata.freq)
    log1p(train)

    if dataset.test is not None:
        test = ListDataset(dataset.test, freq=dataset.metadata.freq)
        log1p(test)
    else:
        test = None

    # fmt: off
    return TrainDatasets(
        dataset.metadata.copy(),  # Note: pydantic's deep copy.
        train=train,
        test=test
    )
    # fmt: on


def log1p(ds: ListDataset):
    """In-place log transformation."""
    for data_entry in ds:
        data_entry["target"] = np.log1p(data_entry["target"])


def expm1_and_clip_to_zero(_, yhat: np.ndarray):
    """Expm1, followed by clip at 0.0."""
    # logger.debug("Before expm1: %s %s", yhat.shape, yhat)
    # logger.debug("After expm1: %s %s", yhat.shape, np.expm1(yhat))

    return np.clip(np.expm1(yhat), a_min=0.0, a_max=None)


def clip_to_zero(_, yhat: np.ndarray):
    return np.clip(yhat, a_min=0.0, a_max=None)
