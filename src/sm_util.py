import logging
import os
import warnings
from pathlib import Path
from typing import Any, Dict, Union

from gluonts.core import serde
from gluonts.dataset.common import MetaData

################################################################################
# To deal with estimator instantiation
################################################################################


def hp2estimator(algo: str, algo_args: Dict[str, Any], metadata: MetaData) -> Any:
    algo_args = merge_metadata_hp(algo_args, metadata)
    estimator_config = klass_dict(algo, [], deser_algo_args(algo_args, deser_args[algo]))
    estimator = serde.decode(estimator_config)
    return estimator


def klass_dict(klass: str, args=[], kwargs={}):
    return {"__kind__": "instance", "class": klass, "args": args.copy(), "kwargs": kwargs.copy()}


def get_kwargs(hp: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    prefix += "."
    prefix_len = len(prefix)
    kwargs = {k[prefix_len:]: v for k, v in hp.items() if k.startswith(prefix)}
    return kwargs


def deser_algo_args(hp: Dict[str, Any], deser_list=[]):
    kwargs = {k: v for k, v in hp.items() if "." not in k}
    for argname in deser_list:
        if argname in kwargs:
            klass_name = kwargs[argname]
            kwargs[argname] = klass_dict(klass_name, [], get_kwargs(hp, argname))
    return kwargs


# TODO: time feature: Optional[List[TimeFeature]]
deser_args = {
    "gluonts.model.deepar.DeepAREstimator": ["trainer", "distr_output"],
    "gluonts.model.deepstate.DeepStateEstimator": [
        "trainer",
        "issm",
        "noise_std_bounds",
        "prior_cov_bounds",
        "innovation_bounds",
        # Optional[List[TimeFeature]],
    ],
    "gluonts.model.deep_factor.DeepFactorEstimator": ["trainer", "distr_output"],
    "gluonts.model.transformer.TransformerEstimator": ["trainer", "distr_output"],  # Optional[List[TimeFeature]]
    "gluonts.model.gp_forecaster.GaussianProcessEstimator": [
        "trainer",
        "kernel_output",
    ],  # Optional[List[TimeFeature]], also cardinality should autodetect from train data?
    # NPTS: see predictor for the args & kwargs.
    # NOTE:
    # - npts doesn't need training, and just straight away predict. However, we still need to use its estimator to fake
    #   the training to let this script treat this algo exactly the same way as the others, rather than having an edge
    #   case aka if-else (ugh so yucky).
    # - kernel_type accepts strings: 'exponential' or 'uniform'. See the enumeration KernelType, so we actually don't
    #   need to do anything with this kwarg.
    "gluonts.model.npts.NPTSEstimator": [],  # ["kernel_type"],
}


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


################################################################################
# Misc. utilities
################################################################################


def mkdir(path: Union[str, os.PathLike]):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
