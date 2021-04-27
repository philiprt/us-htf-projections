# ---------------------------------------------------------------------------

import glob
import os

import numpy as np
import pandas as pd

import pickle
import xarray as xr

import datetime
import matplotlib.dates as mdates
import pendulum
from timezonefinder import TimezoneFinder

import utide

from support_functions import station_string

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def tide_gauge_data(sta):

    # -----------------------------------------------------------------------
    # load station data

    fname = "../data/tide_gauge/" + str(sta["noaa id"]) + ".nc"

    fname = glob.glob(fname)
    if len(fname) == 0:
        tg = None
        return tg
    else:
        fname = fname[0]

    tg = xr.open_dataset(fname).to_dataframe()
    tg.columns = ["sl", "td"]

    beg_end = tg.sl.loc[~tg.sl.isna()].iloc[[0, -1]].index
    tg = tg.loc[beg_end[0] : beg_end[1]]

    tg *= 100  # cm above MHHW

    # -----------------------------------------------------------------------

    if sta["noaa id"] == 8413320:
        tg = tg.drop(tg.loc["1975-08":"1977"].index)

    if sta["noaa id"] == 8774770:
        tg = tg.loc["1960":]

    if sta["noaa id"] == 1770000:
        tg = tg.loc[:"2009-09-28"]

    if sta["noaa id"] == 8729108:
        tg = tg.loc["1979":]

    # -----------------------------------------------------------------------

    sta_str = station_string(sta)
    fname = "../tides/output/" + sta_str + "/tide_prediction_mn_std.pickle"

    if os.path.exists(fname):

        with open(fname, "rb") as f:
            ptd = pickle.load(f)

        tg = tg.loc[: ptd["obs"].index[-1]]
        tg.td = ptd["obs"].reindex(tg.index)
        tg.td -= tg.td.loc["1983":"2001"].mean()
        tg.td += tg.sl.loc["1983":"2001"].mean()

    # -----------------------------------------------------------------------
    # calculate residuals

    tg["res"] = tg.sl - tg.td

    # -----------------------------------------------------------------------
    # calculate annual cycle and trend in residuals

    jdt = tg.index.to_julian_date()
    jdt -= jdt[0]

    # calculate annual cycle and trend in residuals
    phs = jdt * (2 * np.pi) / 365.25
    harm = np.vstack(
        [np.sin(phs), np.cos(phs), np.sin(2 * phs), np.cos(2 * phs)]
    )

    z = ~tg.res.isna()
    A = np.vstack([np.ones(jdt.size), jdt, harm]).T
    c = np.linalg.lstsq(A[z, :], tg.res[z], rcond=None)[0]

    tg["acyc"] = np.sum(A[:, 2:] * c[2:], axis=1)
    tg["trnd"] = np.sum(A[:, 0:2] * c[0:2], axis=1)

    # --------------------------------------------------------------------

    return tg

    # -----------------------------------------------------------------------
