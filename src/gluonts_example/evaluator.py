import json
import os
from typing import Any, Callable, Dict, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gluonts.evaluation import Evaluator
from gluonts.model.forecast import Config, Forecast
from smallmatter.ds import MontagePager

from .metrics import wmape
from .util import mkdir

output_configuration = Config(quantiles=["0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9"])

plt.rcParams.update(
    {
        "legend.fontsize": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.borderpad": 0.8,
    }
)


class MyEvaluator(Evaluator):
    # NOTE: do not write anything to stdout or stderr, otherwise mix-up with tqdm progress bar.
    def __init__(
        self,
        out_dir: os.PathLike,
        *args,
        plot_transparent: bool = False,
        gt_inverse_transform: Optional[Callable] = None,
        clip_at_zero: bool = True,
        **kwargs,
    ):
        super().__init__(*args, num_workers=0, **kwargs)
        self.out_dir = mkdir(out_dir)
        self.out_fname = self.out_dir / "results.jsonl"
        self.plot_dir = mkdir(self.out_dir / "plots")
        mkdir(self.plot_dir / "montages")
        mkdir(self.plot_dir / "individuals")

        self.gt_inverse_transform = gt_inverse_transform
        self.clip_at_zero = clip_at_zero

        # Plot configurations
        self.plot_ci = [50.0, 90.0]
        self.plot_transparent = plot_transparent
        self.figure, self.ax = plt.subplots(figsize=(6.4, 4.8), dpi=100, tight_layout=True)
        self.mp = MontagePager(
            self.out_dir / "plots", page_size=100, savefig_kwargs={"transparent": self.plot_transparent}
        )

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

        # Add to montage
        self.plot_prob_forecasts(self.mp.pop(forecast.item_id), time_series, forecast, self.plot_ci)

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
