import json
import math
import os
import warnings
from typing import Any, Dict, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gluonts.evaluation import Evaluator
from gluonts.model.forecast import Config, Forecast

from metrics import wmape
from sm_util import mkdir

output_configuration = Config(quantiles=["0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9"])
# fmt: off
plt.rcParams.update({
    "legend.fontsize": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8
})
# fmt: on


class MyEvaluator(Evaluator):
    # NOTE: we must not write anything to stdout or stderr, otherwise will intermingle with tqdm progress bar!!!
    def __init__(self, out_dir: os.PathLike, ts_count: int, *args, plot_transparent: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.out_dir = mkdir(out_dir)
        self.out_fname = self.out_dir / "results.jsonl"
        self.plot_dir = mkdir(self.out_dir / "plots")
        self.plot_single_dir = mkdir(self.plot_dir / "single")

        # Plot configurations
        self.ts_count = ts_count  # FIXME: workaround until SimpleMatrixPlotter can dynamically adds axes
        self.plot_ci = [50.0, 90.0]
        self.plot_transparent = plot_transparent
        self.figure, self.ax = plt.subplots(figsize=(6.4, 4.8), dpi=100, tight_layout=True)
        self.smp = SimpleMatrixPlotter(init_figcount=self.ts_count)

        # A running counter
        self.i = 0

        self.out_f = self.out_fname.open("w")

    def get_metrics_per_ts(
        self, time_series: Union[pd.Series, pd.DataFrame], forecast: Forecast
    ) -> Dict[str, Union[float, str, None]]:
        # Compute the built-in metrics
        metrics = super().get_metrics_per_ts(time_series, forecast)

        result: Dict[str, Any] = {"item_id": str(forecast.item_id), **forecast.as_json_dict(output_configuration)}
        json.dump(result, self.out_f)
        self.out_f.write("\n")

        # region: custom metrics.
        # Follow gluonts.evaluation.Evaluator who uses median
        pred_target = np.array(self.extract_pred_target(time_series, forecast))
        pred_target = np.ma.masked_invalid(pred_target)
        median_fcst = forecast.quantile(0.5)

        # And here comes custom metrics...
        metrics["wMAPE"] = wmape(pred_target, median_fcst, version=2)
        # endregion: custom metrics

        # region: plottings
        # Add to montage
        self.plot_prob_forecasts(self.smp.pop(), time_series, forecast, self.plot_ci)

        # Plot & save as a single image
        plt.figure(self.figure.number)
        plt.sca(self.ax)
        self.ax.clear()
        self.plot_prob_forecasts(self.ax, time_series, forecast, self.plot_ci)
        plt.tight_layout()
        self.figure.savefig(self.plot_single_dir / f"{self.i:03d}.png", transparent=self.plot_transparent)

        # TODO: should we reset this in __call__()?
        self.i += 1
        # endregion: plotting

        return metrics

    def get_aggregate_metrics(self, metric_per_ts: pd.DataFrame) -> Tuple[Dict[str, float], pd.DataFrame]:
        totals, metrics_per_ts = super().get_aggregate_metrics(metric_per_ts)

        # region: for each metric, aggregate across timeseries.
        # Aggregation step
        agg_funs = {
            "wMAPE": "mean",
        }
        assert set(metric_per_ts.columns) >= agg_funs.keys(), "The some of the requested item metrics are missing."
        my_totals = {key: metric_per_ts[key].agg(agg) for key, agg in agg_funs.items()}

        # Update base metrics with our custom metrics.
        totals.update(my_totals)
        # endregion

        # Save montage
        self.smp.savefig(self.plot_dir / "plots.png", transparency=self.plot_transparent)

        return totals, metrics_per_ts

    @staticmethod
    def plot_prob_forecasts(ax, time_series, forecast, intervals, past_length=8):
        plot_length = past_length + forecast.prediction_length
        legend = ["observations", "median prediction"] + [f"{k}% prediction interval" for k in intervals[::-1]]
        time_series[-plot_length:].plot(ax=ax)  # plot the ground truth (incl. truncated historical)
        forecast.plot(prediction_intervals=intervals, color="g")
        plt.grid(which="both")
        plt.legend(legend, loc="upper left")
        plt.gca().set_title(forecast.item_id.replace("|", "\n"))


# This is a snapshot from mlsl's mlmax library.
class SimpleMatrixPlotter(object):
    """A simple helper class to fill-in subplot one after another.

    Sample usage using add():

    >>> import pandas as pd
    >>> df = pd.DataFrame({'a': [1,1,1,2,2,2,3,3,3,4,4]})
    >>> gb = df.groupby(by=['a'])
    >>>
    >>> smp = SimpleMatrixPlotter(gb.ngroups)
    >>> for group_name, df_group in gb:
    >>>     ax, _ = smp_1.add(df_group.plot)
    >>>     assert ax == _
    >>>     ax.set_title(f"Item={group_name}")
    >>> # smp.trim(); plt.show()
    >>> smp.savefig("/tmp/testfigure.png")  # After this, figure & axes are gone.

    Alternative usage using pop():

    >>> smp = SimpleMatrixPlotter(gb.ngroups)
    >>> for group_name, df_group in gb:
    >>>     ax = smp.pop()
    >>>     df_group.plot(ax=ax, title=f"Item={group_name}")
    >>> # smp.trim(); plt.show()
    >>> smp.savefig("/tmp/testfigure.png")  # After this, figure & axes are gone.

    Attributes:
        i (int): Index of the currently free subplot
    """

    def __init__(
        self, ncols: Union[str, int] = "square", init_figcount: int = 5, figsize=(6.4, 4.8), dpi=100, **kwargs
    ):
        """Initialize a ``SimpleMatrixPlotter`` instance.

        Args:
            ncols (int, optional): Number of columns. Passing "square" means to set to sqrt(init_figcount) clipped at
                5 and 20. Defaults to "square".
            init_figcount (int, optional): Total number of subplots. Defaults to 5.
            figsize: size per subplot, see figsize for matplotlib. Defaults to (6.4, 4.8).
            dpi: dot per inch, see matplotlib. Defaults to 100.
        """
        # Initialize subplots
        if ncols == "square":
            ncols = min(max(5, int(math.sqrt(init_figcount))), 20)
        nrows = init_figcount // ncols + (init_figcount % ncols > 0)
        self.fig, self.axes = plt.subplots(
            nrows=nrows, ncols=ncols, figsize=(figsize[0] * ncols, figsize[1] * nrows), dpi=100
        )
        self.flatten_axes = self.axes.flatten()
        self.fig.subplots_adjust(hspace=0.35)

        self._i = 0  # Index of the current free subplot

        # Warn if initial pixels exceed matplotlib limit.
        pixels = np.ceil(self.fig.get_size_inches() * self.fig.dpi).astype("int")
        if (pixels > 2 ** 16).any():
            warnings.warn(f"Initial figure is {pixels} pixels, and at least one dimension exceeds 65536 pixels.")

    @property
    def i(self):
        """:int: Index of the current free subplot."""
        return self._i

    def add(self, plot_fun, *args, **kwargs) -> Tuple[plt.Axes, Any]:
        """Fill the current free subplot using `plot_fun()`, and set the axes and figure as the current ones.

        Args:
            plot_fun (callable): A function that must accept `ax` keyword argument.

        Returns:
            (plt.Axes, Any): a tuple of (axes, return value of plot_fun).
        """
        ax = self.pop()
        retval = plot_fun(*args, ax=ax, **kwargs)
        return ax, retval

    def pop(self) -> plt.Axes:
        """Get the next axes in this subplot, and set the it and its figure as the current axes and figure,
        respectively.

        Returns:
            plt.Axes: the next axes
        """
        # TODO: extend with new subplots:
        # http://matplotlib.1069221.n5.nabble.com/dynamically-add-subplots-to-figure-td23571.html#a23572
        ax = self.flatten_axes[self._i]
        plt.sca(ax)
        plt.figure(self.fig.number)
        self._i += 1
        return ax

    def trim(self):
        for ax in self.flatten_axes[self._i :]:
            self.fig.delaxes(ax)

    def savefig(self, *args, **kwargs):
        self.trim()
        kwargs["bbox_inches"] = "tight"
        self.fig.savefig(*args, **kwargs)
        plt.close(self.fig)
