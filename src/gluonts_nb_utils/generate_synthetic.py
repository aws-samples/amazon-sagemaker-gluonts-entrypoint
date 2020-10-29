import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, TextIO, Union

import pandas as pd
from gluonts.dataset.artificial._base import ArtificialDataset
from gluonts.dataset.field_names import FieldName
from smallmatter.pathlib import Path2, S3Path


# Adapted from gluonts.dataset.artificial.generate_synthetic.generate_sf2s_and_csv()
# See also: https://gluon-ts.mxnet.io/api/gluonts/gluonts.dataset.artificial.generate_synthetic.html
def generate_daily_csv(
    file_name: Union[str, Path2],
    artificial_dataset: ArtificialDataset,
    is_missing: bool = False,
    num_missing: int = 4,
    colnames: Sequence[str] = ("ts_id", "x", "y"),
    ts_prefix: str = "",
) -> None:
    """Generate daily data in csv format, where the `artifical_dataset` must have "D" frequency."""
    # Checks.
    freq = artificial_dataset.metadata.freq
    assert freq == "D", 'artificial_dataset must have "D" frequency'

    # Create directory if not exist.
    if not isinstance(file_name, Path):
        file_name = Path2(file_name)
    _try_mkdir_parent(file_name)

    with file_name.open("w") as csv_file:
        # Test set has training set with the additional values to predict.
        _write_csv(artificial_dataset.test, freq, csv_file, is_missing, num_missing, colnames, ts_prefix)


def _try_mkdir_parent(path: Path2) -> None:
    if isinstance(path, S3Path):
        return
    path.parent.mkdir(parents=True, exist_ok=True)


# Adapted from gluonts.dataset.artificial.generate_synthetic.write_csv_row()
# See also: https://gluon-ts.mxnet.io/api/gluonts/gluonts.dataset.artificial.html
def _write_csv(
    time_serieses: List[Dict[str, Any]],
    freq: str,
    csv_file: TextIO,
    is_missing: bool,
    num_missing: int,
    usecols: Optional[Sequence[str]] = None,
    ts_prefix: str = "",
) -> None:
    csv_writer = csv.writer(csv_file)
    if usecols:
        csv_writer.writerow(usecols)

    time_delta = 1 * pd.Timedelta(1, unit=freq)
    zfill = len(str(len(time_serieses)))
    for timeseries in time_serieses:
        _write_timeseries(
            csv_writer,
            timeseries,
            freq,
            time_delta,
            ts_prefix,
            is_missing,
            num_missing,
            zfill=zfill,
        )


# convert to right date where MON == 0, ..., SUN == 6
_week_dict = {
    0: "MON",
    1: "TUE",
    2: "WED",
    3: "THU",
    4: "FRI",
    5: "SAT",
    6: "SUN",
}


def _write_timeseries(
    csv_writer,
    timeseries: Dict[str, Any],
    freq: str,
    time_delta: pd.Timedelta,
    ts_prefix: str,
    is_missing: bool,
    num_missing: int,
    zfill: int = 0,
) -> None:
    # Prefix item_id
    item_id = timeseries[FieldName.ITEM_ID]
    item_id = str(item_id).zfill(zfill)
    item_id = f"{ts_prefix}{item_id}"

    timestamp = pd.Timestamp(timeseries[FieldName.START])
    if freq == "W":
        freq = f"W-{_week_dict[timestamp.weekday()]}"
        timestamp = pd.Timestamp(timeseries[FieldName.START], freq=freq)

    for row_idx, target in enumerate(timeseries[FieldName.TARGET]):
        # ComplexSeasonalTimeSeries may produce non-numbers
        if (target is None) or (target == "NaN"):
            continue
        else:
            timestamp_row = timestamp.date() if freq in ["W", "D", "M"] else timestamp
            row = [item_id, timestamp_row, target]

            # Check if related time series is present
            if FieldName.FEAT_DYNAMIC_REAL in timeseries.keys():
                for feat_dynamic_real in timeseries[FieldName.FEAT_DYNAMIC_REAL]:
                    row.append(feat_dynamic_real[row])

            csv_writer.writerow(row)

        timestamp += time_delta
