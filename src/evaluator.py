import os
from typing import Any, Dict, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gluonts.evaluation import Evaluator
from gluonts.model.forecast import Forecast

from metrics import wmape
from sm_util import mkdir


class MyEvaluator(Evaluator):
    # NOTE: we must not write anything to stdout or stderr, otherwise will intermingle with tqdm progress bar!!!
    def __init__(self, plot_dir: os.PathLike, ts_count: int, *args, plot_transparent: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.plot_dir = plot_dir
        self.plot_single_dir = mkdir(self.plot_dir / "single")
        self.ts_count = ts_count  # FIXME: workaround until SimpleMatrixPlotter can dynamically adds axes
        self.plot_ci = [50.0, 90.0]
        self.plot_transparent = plot_transparent
        self.figure, self.ax = plt.subplots(figsize=(8, 4.5), dpi=300)
        self.smp = SimpleMatrixPlotter(ncols=5, init_figcount=self.ts_count)
        self.i = 0

    def get_metrics_per_ts(
        self, time_series: Union[pd.Series, pd.DataFrame], forecast: Forecast
    ) -> Dict[str, Union[float, str, None]]:
        # Compute the built-in metrics
        metrics = super().get_metrics_per_ts(time_series, forecast)

        # region: custom metrics.
        # Follow gluonts.evaluation.Evaluator who uses median
        pred_target = np.array(self.extract_pred_target(time_series, forecast))
        pred_target = np.ma.masked_invalid(pred_target)
        median_fcst = forecast.quantile(0.5)

        # And here comes custom metrics...
        metrics["wMAPE"] = wmape(pred_target, median_fcst, version=2)
        # endregion: custom metrics

        # region: plottings
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

        # region: save montage
        self.smp.trim()
        self.smp.fig.tight_layout()
        self.smp.fig.savefig(self.plot_dir / "plots.png", transparency=self.plot_transparent)
        # endregion: montage

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
