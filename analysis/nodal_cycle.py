# ---------------------------------------------------------------------------

import json
import os
import shutil
import pickle
import warnings
from multiprocessing import Pool

import hdf5storage as hdf5
import numpy as np
import pandas as pd
import scipy as sp
import scipy.stats as stats
from scipy.interpolate import interp1d, splev, splrep

from support_functions import station_string
from tide_gauge_data import *
from utilities import *

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def main():

    # -----------------------------------------------------------------------

    fname = "../data/stations.pickle"
    stations = pd.read_pickle(fname)

    # # restart
    # stations = stations.loc[72:, :]

    # -----------------------------------------------------------------------

    # remove stations to skip
    skip = [8771013, 8557380, 8467150, 1619910]
    keep = [
        n for n in stations.index if stations.loc[n, "noaa id"] not in skip
    ]
    stations = stations.loc[keep]

    # priortize certain stations in the order of completion
    first = [1612340, 8443970, 8658120, 8545240, 9414290]
    stations["sort"] = [
        first.index(stations.loc[n, "noaa id"])
        if stations.loc[n, "noaa id"] in first
        else n + len(first)
        for n in stations.index
    ]
    stations = stations.sort_values(by="sort")
    stations.index = range(stations.index.size)

    # # only do select stations
    # select = [8726520]  # 8723214, 1612340, 9410230, 8443970, 8726520
    # keep = [n for n in stations.index if stations.loc[n, "noaa id"] in select]
    # stations = stations.loc[keep]

    # -----------------------------------------------------------------------

    Nsta = stations.index.size
    for idx, sta in stations.iterrows():

        print(
            "\nStation "
            + str(sta["noaa id"])
            + ": "
            + sta["name"]
            + " ("
            + str(idx)
            + " of "
            + str(Nsta)
            + ")"
        )

        nodal_cycle_vs_slr(sta)


# ---------------------------------------------------------------------------


def nodal_cycle_vs_slr(sta):

    sta_str = station_string(sta)

    # -----------------------------------------------------------------------
    # Directories for I/O
    # -----------------------------------------------------------------------

    # station directory
    sta_path = "./nodal_cycle/" + sta_str + "/"
    os.makedirs(sta_path, exist_ok=True)

    # -----------------------------------------------------------------------
    # Load resources and setup
    # -----------------------------------------------------------------------

    tg = tide_gauge_data(sta)
    if tg is None:
        return

    # -----------------------------------------------------------------------
    # calculate nodal cycle amplitude

    amn = tg.sl.groupby(pd.Grouper(freq="A")).mean()
    a99 = tg.td.groupby(pd.Grouper(freq="A")).quantile(0.99)
    aN = tg.td.groupby(pd.Grouper(freq="A")).apply(
        lambda x: x.dropna().count()
    )
    a99 = a99.loc[aN > 0.95 * 8760]
    amn = amn.loc[a99.index]
    a99.index = a99.index.year
    amn.index = amn.index.year

    y = a99.index + 0.5

    A = np.stack(
        [
            np.ones(y.size),
            y - y.values.mean(),
            np.sin(2 * np.pi * y / 18.61),
            np.cos(2 * np.pi * y / 18.61),
        ]
    ).T

    ncyc_c = np.linalg.lstsq(A, a99, rcond=None)[0]
    ncyc = pd.Series(A @ ncyc_c, index=y)
    ncyc_mn_ntde = ncyc.loc[1983.5:2001.5].mean()

    ncyc_mod = ncyc.max() - ncyc.min()

    # -----------------------------------------------------------------------
    # recent slr

    # amn = amn.loc[2000:].dropna()
    amn = amn.dropna()
    y = amn.index - amn.index[0]

    A = np.stack(
        [
            np.ones(y.size),
            y,
        ]
    ).T

    c = np.linalg.lstsq(A, amn, rcond=None)[0]
    slr = c[1] * 10

    # -----------------------------------------------------------------------
    # scenario-based future sea level rise

    pid = sta["psmsl_id"]

    scn_nm = "int"

    fname = "../data/noaa_scenarios/NOAA_SLR_scenarios.pickle"
    with open(fname, "rb") as f:
        scn = pickle.load(f)
    scn["proj"].index += 0.5

    scn = scn["proj"].loc[:, (pid, scn_nm, 50)]

    idx = np.arange(2010, 2101, 1 / 12) + 1 / 24
    noaa_prjn = pd.Series(index=idx)

    spl = splrep(scn.index, scn.values)
    noaa_prjn.loc[:] = splev(idx, spl)
    noaa_prjn += tg.sl.loc["1991":"2009"].mean()

    i1 = (idx > 2030).argmax()
    i2 = (idx < 2040).argmin() - 1
    slr_prjn = noaa_prjn.iloc[i2] - noaa_prjn.iloc[i1]

    # A21 = np.stack(
    #     [np.sin(2 * np.pi * idx / 18.61), np.cos(2 * np.pi * idx / 18.61),]
    # ).T
    # ncyc_21 = (
    #     pd.Series(A21 @ ncyc_c[-2:], index=idx)
    #     - tg.td.loc["1983":"2001"].mean()
    # )

    # import matplotlib.pyplot as plt
    #
    # plt.clf()
    # (noaa_prjn + 0).plot()
    # (noaa_prjn + ncyc_21).plot()
    # plt.xlim([2020, 2050])
    # plt.ylim([-15, 45])
    # plt.show()
    #
    raise

    # -----------------------------------------------------------------------

    fname = sta_path + "/nodal_cycle_vs_slr.pickle"
    with open(fname, "wb") as f:
        pickle.dump({"ncyc": ncyc_mod, "slr": slr, "slr_prjn": slr_prjn}, f)


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
