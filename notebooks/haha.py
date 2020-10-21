# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

# Standard library imports
import csv
import json
import os
from typing import List, TextIO, Optional, Sequence
import random
import math

# Third party imports
import pandas as pd

# First-party imports
from gluonts.dataset.artificial._base import ArtificialDataset, ConstantDataset
from gluonts.dataset.field_names import FieldName

class MyConstantDataset(ConstantDataset):
    def generate_ts(self, *args, **kwargs):
        """Left-clip and scale-up y by a fix constant."""
        res = super().generate_ts(*args, **kwargs)
        for ts in res:
            ts["target"] = [max(1, math.ceil(y)*2) for y in ts["target"]]
        return res

def write_csv_row(
    time_series: List,
    freq: str,
    csv_file: TextIO,
    is_missing: bool,
    num_missing: int,
    usecols: Optional[Sequence[str]]=None,
    ts_prefix: str="",
) -> None:
    csv_writer = csv.writer(csv_file)
    if usecols:
        csv_writer.writerow(usecols)
    # convert to right date where MON == 0, ..., SUN == 6
    week_dict = {
        0: "MON",
        1: "TUE",
        2: "WED",
        3: "THU",
        4: "FRI",
        5: "SAT",
        6: "SUN",
    }
    time_delta = 1 * pd.Timedelta(1, unit=freq)
    for i in range(len(time_series)):
        data = time_series[i]
        timestamp = pd.Timestamp(data[FieldName.START])
        freq_week_start = freq
        if freq_week_start == "W":
            freq_week_start = f"W-{week_dict[timestamp.weekday()]}"
        timestamp = pd.Timestamp(data[FieldName.START], freq=freq_week_start)
        item_id = int(data[FieldName.ITEM_ID])
        for j, target in enumerate(data[FieldName.TARGET]):
            # Using convention that there are no missing values before the start date
            if is_missing and j != 0 and j % num_missing == 0:
                timestamp += time_delta      # Fix pandas error
                continue  # Skip every 4th entry
            else:
                timestamp_row = timestamp
                if freq in ["W", "D", "M"]:
                    timestamp_row = timestamp.date()
                try:
                    int(target)
                except:
                    # marcverd: complex seasional time can produce None or NaN, these must be skip.
                    continue
                else:
                    row = [f"{ts_prefix}{item_id}", timestamp_row, target]  # marcverd: left-clip
                    # Check if related time series is present
                    if FieldName.FEAT_DYNAMIC_REAL in data.keys():
                        for feat_dynamic_real in data[FieldName.FEAT_DYNAMIC_REAL]:
                            row.append(feat_dynamic_real[j])
                    csv_writer.writerow(row)
                finally:
                    timestamp += time_delta  # Fix pandas error


def generate_daily_data(
    file_path: str,
    folder_name: str,
    artificial_dataset: ArtificialDataset,
    is_missing: bool = False,
    num_missing: int = 4,
    colnames=['ts_id', 'x', 'y'],
    ts_prefix=""
) -> None:
    file_path += f"/{folder_name}"
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))
    freq = artificial_dataset.metadata.freq
    test_set = artificial_dataset.test
    with open(file_path + "/input_to_forecast.csv", "w") as csv_file:
        # Test set has training set with the additional values to predict
        write_csv_row(test_set, freq, csv_file, is_missing, num_missing, colnames, ts_prefix)
