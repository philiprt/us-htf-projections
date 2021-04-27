# ---------------------------------------------------------------------------

import numpy as np
import pandas as pd
import pickle
import xarray as xr
import glob
import os
import json
import requests

from support_functions import station_string

# ---------------------------------------------------------------------------

fname = "../data/stations.pickle"
stations = pd.read_pickle(fname)

# -----------------------------------------------------------------------

# # only do select stations
# select = [8726520, 8443970, 9410230, 1612340]
# keep = [n for n in stations.index if stations.loc[n, "noaa id"] in select]
# stations = stations.loc[keep]

# ---------------------------------------------------------------------------

n = 0
Nsta = stations.index.size
for idx, sta in stations.iterrows():

    n += 1

    print(
        "Station "
        + str(sta["noaa id"])
        + ": "
        + sta["name"]
        + " ("
        + str(n)
        + " of "
        + str(Nsta)
        + ")"
    )

    url = (
        "https://api.tidesandcurrents.noaa.gov/mdapi/prod/webapi/stations/"
        + str(sta["noaa id"])
        + "/datums.json"
    )

    # get datums
    datums = json.loads(requests.get(url).text)["datums"]

    # convert mhhw and mllw from feet to cm
    mhhw = [d["value"] for d in datums if d["name"] == "MHHW"][0] * 30.48
    mllw = [d["value"] for d in datums if d["name"] == "MLLW"][0] * 30.48
    gt = mhhw - mllw  # great diurnal range

    # -----------------------------------------------------------------------
    # get flooding thresholds in cm above MHHW based on Sweet et al. (2018)

    flood = {
        "minor": np.round(0.04 * gt + 50),
        "moderate": np.round(0.03 * gt + 80),
        "major": np.round(0.04 * gt + 117),
    }

    # -----------------------------------------------------------------------

    sta_str = station_string(sta)

    sta_path = "./output/" + sta_str + "/"
    os.makedirs(sta_path, exist_ok=True)
    fname = sta_path + "noaa_thresholds.pickle"
    with open(fname, "wb") as f:
        pickle.dump(flood, f)

    # -----------------------------------------------------------------------
