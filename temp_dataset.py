import numpy as np
import pandas as pd

from darts import TimeSeries
from darts.datasets import ElectricityDataset

df = ElectricityDataset().load().pd_dataframe()

ts_list = []
for label in df:
    srs = df[label]

    # filter column down to the period of recording
    srs = srs.replace(0.0, np.nan)
    start_date = min(srs.fillna(method="ffill").dropna().index)
    end_date = max(srs.fillna(method="bfill").dropna().index)
    active_range = (srs.index >= start_date) & (srs.index <= end_date)
    srs = srs[active_range].fillna(0.0)

    tmp = pd.DataFrame({"power_usage": srs})
    tmp["date"] = tmp.index
    date = tmp.index
    ts = TimeSeries.from_dataframe(tmp, "date", ["power_usage"])
    ts_list.append(ts)

asd = 0
