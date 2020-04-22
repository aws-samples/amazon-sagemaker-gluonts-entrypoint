import os
import warnings
from pathlib import Path
from typing import Any, Dict, Tuple, Union

import matplotlib.pyplot as plt
import pandas as pd
from gluonts.core import serde
from gluonts.dataset.common import MetaData
from gluonts.evaluation import Evaluator
from gluonts.model.forecast import Forecast

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
            # kwargs[argname] = serde.decode(klass_dict(klass_name, [], get_kwargs(hp, argname)))
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
        print(f"freq: set freq='{metadata.freq}' from metadata; ignore '{freq_hp}' from hyperparam.")
    hp["freq"] = metadata.freq

    # Use prediction_length hyperparameters, but if not specified then fallbacks/defaults to the one from metadata.
    if "prediction_length" not in hp:
        hp["prediction_length"] = metadata.prediction_length
        print(
            "prediction_length: no hyperparam, so set " f"prediction_length={metadata.prediction_length} from metadata"
        )

    # TODO: autoprobe cardinalities.
    warnings.warn("This implementation still ignores cardinality and static features in the metadata", RuntimeWarning)

    return hp


################################################################################
# Misc. utilities
################################################################################

# This is a snapshot from mlsl's mlmax library.
class SimpleMatrixPlotter(object):
    """A simple helper class to fill-in subplot one after another.

    Sample usage using add():

    >>> import pandas as pd
    >>> df = pd.DataFrame({'a': [1,1,1,2,2,2,3,3,3,4,4]})
    >>> gb = df.groupby(by=['a'])
    >>>
    >>> smp_1 = SimpleMatrixPlotter(gb.ngroups)
    >>> for group_name, df_group in gb:
    >>>     ax, _ = smp_1.add(df_group.plot)
    >>>     assert ax == _
    >>>     ax.set_title(f"Item={group_name}")
    >>> smp_1.trim()
    >>> plt.savefig('/tmp/testfigure.png')   # or: plt.show()
    >>>

    Alternative usage using pop():

    >>> smp_2 = SimpleMatrixPlotter(gb.ngroups)
    >>> for group_name, df_group in gb:
    >>>     ax = smp_2.pop()
    >>>     df_group.plot(ax=ax, title=f"Item={group_name}")
    >>> smp_2.trim()
    >>> plt.savefig('/tmp/testfigure.png')   # or: plt.show()

    Attributes:
        i (int): Index of the currently free subplot
    """

    def __init__(self, ncols: int = 3, init_figcount: int = 5, figscale=(6, 4)):
        """Initialize a ``SimpleMatrixPlotter`` instance.

        Args:
            ncols (int, optional): Number of columns. Defaults to 3.
            init_figcount (int, optional): Total number of subplots. Defaults to 5.
        """
        # Initialize subplots
        nrows = init_figcount // ncols + (init_figcount % ncols > 0)
        self.fig, self.axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(figscale[0] * ncols, figscale[1] * nrows))
        self.flatten_axes = self.axes.flatten()
        plt.subplots_adjust(hspace=0.35)

        self._i = 0  # Index of the current free subplot

    @property
    def i(self):
        """:int: Index of the current free subplot."""
        return self._i

    def add(self, plot_fun, *args, **kwargs) -> Tuple[plt.Axes, Any]:
        """Fill the current free subplot using `plot_fun()`.

        Args:
            plot_fun (callable): A function that must accept `ax` keyword argument.

        Returns:
            (plt.Axes, Any): a tuple of (axes, return value of plot_fun).
        """
        # TODO: extend with new subplots:
        # http://matplotlib.1069221.n5.nabble.com/dynamically-add-subplots-to-figure-td23571.html#a23572
        ax = self.flatten_axes[self._i]
        retval = plot_fun(*args, ax=ax, **kwargs)

        self._i += 1
        return ax, retval

    def pop(self) -> plt.Axes:
        """Get the next axes in this subplot, and set the it as the current axes.

        Returns:
            plt.Axes: the next axes
        """
        ax = self.flatten_axes[self._i]
        plt.sca(ax)
        self._i += 1
        return ax

    def trim(self):
        for ax in self.flatten_axes[self._i :]:
            self.fig.delaxes(ax)


class MyEvaluator(Evaluator):
    def __init__(self, plot_dir: os.PathLike, ts_count: int, *args, plot_transparent: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.plot_dir = plot_dir
        self.plot_single_dir = mkdir(self.plot_dir / "single")
        self.ts_count = ts_count  # FIXME: workaround until SimpleMatrixPlotter can dynamically adds axes
        self.plot_ci = [50.0, 90.0]
        self.plot_transparent = plot_transparent
        self.figure, self.ax = plt.subplots(figsize=(4, 3), dpi=300)
        self.smp = SimpleMatrixPlotter(ncols=5)
        self.i = 0

    def get_metrics_per_ts(
        self, time_series: Union[pd.Series, pd.DataFrame], forecast: Forecast
    ) -> Dict[str, Union[float, str, None]]:
        # Compute the built-in metrics
        metrics = super().get_metrics_per_ts(time_series, forecast)

        # logger.info(f"Plot {forecast.item_id}")   # FIXME: this intermingles with tqdm

        # As a subplot in the grid plotter
        plt.figure(self.smp.fig.number)
        self.smp.pop()
        self.plot_prob_forecasts(plt.gca(), time_series, forecast, self.plot_ci)

        # Plot & save as a single image
        plt.figure(self.figure.number)
        plt.sca(self.ax)
        self.ax.clear()
        self.plot_prob_forecasts(self.ax, time_series, forecast, self.plot_ci)
        plt.tight_layout()
        self.figure.savefig(self.plot_single_dir / f"{self.i:03d}.png", transparent=self.plot_transparent)

        # TODO: should we reset this in __call__()?
        self.i += 1

        return metrics

    def get_aggregate_metrics(self, metric_per_ts: pd.DataFrame) -> Tuple[Dict[str, float], pd.DataFrame]:
        totals, metrics_per_ts = super().get_aggregate_metrics(metric_per_ts)

        # Save the montage
        self.smp.trim()
        self.smp.fig.tight_layout()
        self.smp.fig.savefig(self.plot_dir / "plots.png", transparency=self.plot_transparent)

        return totals, metrics_per_ts

    @staticmethod
    def plot_prob_forecasts(ax, time_series, forecast, intervals, past_length=8):
        plot_length = past_length + forecast.prediction_length
        legend = ["observations", "median prediction"] + [f"{k}% prediction interval" for k in intervals[::-1]]
        time_series[-plot_length:].plot(ax=ax)  # plot the ground truth (incl. truncated historical)
        forecast.plot(prediction_intervals=intervals, color="g")
        plt.grid(which="both")
        plt.legend(legend, loc="upper left")
        plt.gca().set_title(forecast.item_id)


def mkdir(path: Union[str, os.PathLike]):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
