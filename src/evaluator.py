import json
import math
import os
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, Union

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
    "ytick.labelsize": 8,
    "legend.borderpad": 0.8,
})
# fmt: on


class MyEvaluator(Evaluator):
    # NOTE: we must not write anything to stdout or stderr, otherwise will intermingle with tqdm progress bar!!!
    def __init__(
        self,
        out_dir: os.PathLike,
        *args,
        plot_transparent: bool = False,
        gt_inverse_transform: Optional[Callable] = None,
        clip_at_zero: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.out_dir = mkdir(out_dir)
        self.out_fname = self.out_dir / "results.jsonl"
        self.plot_dir = mkdir(self.out_dir / "plots")
        self.plot_single_dir = mkdir(self.plot_dir / "single")

        self.gt_inverse_transform = gt_inverse_transform
        self.clip_at_zero = clip_at_zero

        # Plot configurations
        self.plot_ci = [50.0, 90.0]
        self.plot_transparent = plot_transparent
        self.figure, self.ax = plt.subplots(figsize=(6.4, 4.8), dpi=100, tight_layout=True)
        self.mp = MontagePager(
            self.out_dir / "plots", page_size=100, savefig_kwargs={"transparent": self.plot_transparent}
        )

        # A running counter for individual image files.
        self.i = 0

        self.out_f = self.out_fname.open("w")

    def get_metrics_per_ts(
        self, time_series: Union[pd.Series, pd.DataFrame], forecast: Forecast
    ) -> Dict[str, Union[float, str, None]]:
        # Inverse tranformation (if any), then clip to 0.
        if self.gt_inverse_transform is not None:
            time_series = self.gt_inverse_transform(time_series)
        if self.clip_at_zero:
            time_series = time_series.clip(lower=0.0)

        # Compute the built-in metrics
        metrics = super().get_metrics_per_ts(time_series, forecast)

        # Write forecast results to output file.
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
        self.plot_prob_forecasts(self.mp.pop(), time_series, forecast, self.plot_ci)

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
        self.mp.savefig()

        # Make sure to flush buffered results to the disk.
        self.out_f.close()

        return totals, metrics_per_ts

    @staticmethod
    def plot_prob_forecasts(ax, time_series, forecast, intervals, past_length=8):
        plot_length = past_length + forecast.prediction_length
        time_series[0][-plot_length:].plot(ax=ax, label="actual")
        MyEvaluator.plot2(forecast, prediction_intervals=intervals, show_mean=False)
        plt.grid(which="both")
        plt.legend(loc="upper left")
        plt.gca().set_title(str(forecast.item_id).replace("|", "\n"))

    @staticmethod
    def plot2(
        forecast,
        prediction_intervals=(50.0, 90.0),
        show_mean=False,
        color="g",  # This is for alpha (CI range)
        label=None,
        *args,
        **kwargs,
    ):
        """Customize gluonts.model.forecast.Forecast.

        Notable changes: change median and mean to other than green, increase transparencyi of interval filling.

        The rest are exactly the same as the original.
        """
        label_prefix = "" if label is None else label + "-"

        for c in prediction_intervals:
            assert 0.0 <= c <= 100.0

        ps = [50.0] + [50.0 + f * c / 2.0 for c in prediction_intervals for f in [-1.0, +1.0]]
        percentiles_sorted = sorted(set(ps))

        def alpha_for_percentile(p):
            return (p / 100.0) ** 0.5  # marcverd: increase transparency

        ps_data = [forecast.quantile(p / 100.0) for p in percentiles_sorted]
        i_p50 = len(percentiles_sorted) // 2

        p50_data = ps_data[i_p50]
        p50_series = pd.Series(data=p50_data, index=forecast.index)
        p50_series.plot(color="xkcd:maroon", ls="-", label=f"{label_prefix}median")

        if show_mean:
            mean_data = np.mean(forecast._sorted_samples, axis=0)
            pd.Series(data=mean_data, index=forecast.index).plot(
                color="xkcd:crimson", ls=":", label=f"{label_prefix}mean", *args, **kwargs,
            )

        for i in range(len(percentiles_sorted) // 2):
            ptile = percentiles_sorted[i]
            alpha = alpha_for_percentile(ptile)
            plt.fill_between(
                forecast.index,
                ps_data[i],
                ps_data[-i - 1],
                facecolor=color,
                alpha=alpha,
                interpolate=True,
                *args,
                **kwargs,
            )
            # Hack to create labels for the error intervals.
            # Doesn't actually plot anything, because we only pass a single data point
            pd.Series(data=p50_data[:1], index=forecast.index[:1]).plot(
                color=color, alpha=alpha, linewidth=8, label=f"{label_prefix}{100 - ptile * 2}%", *args, **kwargs,
            )


################################################################################
# Visualization utilities -- these are a snapshot from mlsl's mlmax library.
################################################################################
class SimpleMatrixPlotter(object):
    """A simple helper class to fill-in subplot one after another.

    Sample usage using add():

    >>> import pandas as pd
    >>> df = pd.DataFrame({'a': [1,1,1,2,2,2,3,3,3,4,4]})
    >>> gb = df.groupby(by=['a'])
    >>>
    >>> smp = SimpleMatrixPlotter(gb.ngroups)
    >>> for group_name, df_group in gb:
    >>>     ax, _ = smp.add(df_group.plot)
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

    def __init__(self, ncols: Optional[int] = None, init_figcount: int = 5, figsize=(6.4, 4.8), dpi=100, **kwargs):
        """Initialize a ``SimpleMatrixPlotter`` instance.

        Args:
            ncols (int, optional): Number of columns. Passing None means to set to sqrt(init_figcount) clipped at
                5 and 20. Defaults to None.
            init_figcount (int, optional): Total number of subplots. Defaults to 5.
            figsize (tuple, optional): size per subplot, see ``figsize`` for matplotlib. Defaults to (6.4, 4.8).
            dpi (int, optional): dot per inch, see ``dpi`` in matplotlib. Defaults to 100.
            kwargs (optional): Keyword argumetns for plt.subplots, but these are ignored and will be overriden:
                ``ncols``, ``nrows``, ``figsize``, ``dpi``.
        """
        # Initialize subplots
        if ncols is None:
            ncols = min(max(5, int(math.sqrt(init_figcount))), 20)
        nrows = init_figcount // ncols + (init_figcount % ncols > 0)

        kwargs = {k: v for k, v in kwargs.items() if k not in {"nrows", "ncols", "figsize", "dpi"}}
        self.fig, _ = plt.subplots(
            nrows=nrows, ncols=ncols, figsize=(figsize[0] * ncols, figsize[1] * nrows), dpi=100, **kwargs
        )
        self.axes = self.fig.axes  # Cache list of axes returned by self.fig.axes
        self.fig.subplots_adjust(hspace=0.35)
        self._i = 0  # Index of the current free subplot

        # Warn if initial pixels exceed matplotlib limit.
        pixels = np.ceil(self.fig.get_size_inches() * self.fig.dpi).astype("int")
        if (pixels > 2 ** 16).any():
            warnings.warn(f"Initial figure is {pixels} pixels, and at least one dimension exceeds 65536 pixels.")

    @property
    def i(self):
        """:int: Index of the earliest unused subplot."""
        return self._i

    @property
    def ncols(self):
        return self.axes[0].get_geometry()[1] if len(self.axes) > 0 else 0

    @property
    def nrows(self):
        return self.axes[0].get_geometry()[0] if len(self.axes) > 0 else 0

    @property
    def shape(self):
        return (self.nrows, self.ncols)

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
        """Get the next axes in this subplot, and set it and its figure as the current axes and figure, respectively.

        Returns:
            plt.Axes: the next axes
        """
        # TODO: extend with new subplots:
        # http://matplotlib.1069221.n5.nabble.com/dynamically-add-subplots-to-figure-td23571.html#a23572
        ax = self.axes[self._i]
        plt.sca(ax)
        plt.figure(self.fig.number)
        self._i += 1
        return ax

    def trim(self):
        for ax in self.axes[self._i :]:
            self.fig.delaxes(ax)
        self.axes = self.axes[: self._i]

    def savefig(self, *args, **kwargs):
        self.trim()
        kwargs["bbox_inches"] = "tight"
        self.fig.savefig(*args, **kwargs)
        # Whatever possible ways to release figure
        self.fig.clf()
        plt.close(self.fig)
        del self.fig
        self.fig = None


# NOTE: to simplify the pager implementation, just destroy-old-and-create-new SimpleMatrixPlotter instances.
#       Should it be clear that the overhead of this approach is not acceptable, then reset-and-reuse shall be
#       considered.
#
# As of now, using pager does caps memory usage (in addition to making sure not to hit matplotlib limit of 2^16 pixels
# per figure dimension). The following benchmark to render 10 montages at 100 subplots/montage tops at 392MB RSS, when
# measured on MBP early 2015 model, Mojave 10.14.6, python-3.7.6.
#
# from mlmax.visualization.visualize import MontagePager
# import pandas as pd
# mp = MontagePager()
# ser = pd.Series([0,1,2,1,3,2], name='haha')
# for i in range(1000):
#     ser.plot(ax=mp.pop())
# mp.savefig()
class MontagePager(object):
    def __init__(
        self,
        path: os.PathLike = Path("."),
        prefix: str = "montage",
        page_size: int = 100,
        savefig_kwargs: Dict[str, Any] = {},
        **kwargs,
    ):
        """Render plots to one or more montages.

        Each montage has at most ``page_size`` subplots. This pager automatically saves an existing montage when the
        montage is full and an attempt was made to add a new subplot to it. After the exsting montage is saved, a new
        blank montage is created, and the new subplot will be added to it. Callers are expected to explicitly save the
        last montage.

        Args:
            prefix (str, optional): Prefix of output filenames. Defaults to "montage".
            page_size (int, optional): Number of subplots per montage. Defaults to 100.
            savefig_kwargs (dict, optional): Keyword arguments to SimpleMatrixPlotter.savefig(), but ``fname`` will be
                overriden by MontagePager.
            kwargs: Keyword arguments to instantiate each montage (i.e., SimpleMatrixPlotter.__init__()).
        """
        self.path = path
        self.prefix = prefix
        self.page_size = page_size
        self.smp_kwargs = kwargs
        self.smp_kwargs["init_figcount"] = page_size
        self.savefig_kwargs = savefig_kwargs
        self.smp = SimpleMatrixPlotter(**self.smp_kwargs)
        self._i = 0

    @property
    def i(self):
        """:int: Sequence number of the current montage (zero-based)."""
        return self._i

    def pop(self, **kwargs):
        if self.smp.i >= self.page_size:
            self.savefig()
            self.smp = SimpleMatrixPlotter(**self.smp_kwargs)
            self._i += 1
        return self.smp.pop()

    def savefig(self):
        # No need to check for empty montage, because Figure.savefig() won't generate output file in such cases.
        self.smp.savefig(self.path / f"{self.prefix}-{self._i:04d}.png", **self.savefig_kwargs)
