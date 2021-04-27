# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def main():

    # mode = "test"
    mode = "multiprocess"

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

    # only do select stations
    select = [1612340]
    keep = [n for n in stations.index if stations.loc[n, "noaa id"] in select]
    stations = stations.loc[keep]

    # -----------------------------------------------------------------------

    mo_options = range(1, 13)
    # mo_options = [1]

    tod_options = [[0, 23]]
    # tod_options = [[6, 18]]
    # tod_options = [[0, 23], [6, 18]]
    # tod_options = [[0, 23], [8, 18], [6, 9], [16, 19]]

    # -----------------------------------------------------------------------

    # dirs = glob.glob("./Progress/*")
    # for d in dirs:
    #     files = glob.glob(d + "/*")
    #     prg = [pd.read_table(f) for f in files]
    #     fin = [p.iloc[0][0] for p in prg if p.iloc[-1][-1][-1] != "."]
    #     if len(fin) != 0:
    #         incmplt[d] = fin

    # -----------------------------------------------------------------------

    if mode == "test":

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

            for mo in mo_options:
                for tod in tod_options:
                    print(
                        "\nCASE: iso_mo = " + str(mo) + "; tod = " + str(tod)
                    )
                    # try:
                    projections([sta, mo, tod])
                    # except:
                    #     print("FAILED")

    # ----------------------------------------------------------------------

    elif mode == "multiprocess":

        try:
            shutil.rmtree("./progress/")
        except:
            pass
        os.makedirs("./progress/")

        sta_list = []
        for n in stations.index:
            for mo in mo_options:
                for tod in tod_options:
                    sta_list.append([stations.loc[n], mo, tod])

        agents = 3
        chunksize = 1
        with Pool(processes=agents) as pool:
            result = pool.map(projections, sta_list, chunksize)


# ---------------------------------------------------------------------------


def projections(input):

    sta, iso_mo, tod = input[0], input[1], input[2]

    sta_str = station_string(sta)

    # -----------------------------------------------------------------------
    # Setup log file for tracking process
    # ----------------------------------------------------------------------

    mo_str = "m" + "{:02}".format(iso_mo)
    tod_str = "_t" + "{:02}".format(tod[0]) + "{:02}".format(tod[1])
    log_dir = "./progress/" + sta_str + "/"
    os.makedirs(log_dir, exist_ok=True)
    log_file = log_dir + mo_str + tod_str + ".txt"

    with open(log_file, "w") as lf:
        print(sta["name"] + " (NOAA ID: " + str(sta["noaa id"]) + ")", file=lf)
        print("CASE: iso_mo = " + str(iso_mo) + "; tod = " + str(tod), file=lf)

    # -----------------------------------------------------------------------
    # Directories for I/O
    # -----------------------------------------------------------------------

    # json_path = "./output/json/" + sta_str + "/"
    # os.makedirs(json_path, exist_ok=True)

    # station output directory
    sta_path = "./output/" + sta_str + "/"
    os.makedirs(sta_path, exist_ok=True)

    # station ensemble directory
    ens_path = "./ensemble/" + sta_str + "/"
    os.makedirs(sta_path, exist_ok=True)

    # -----------------------------------------------------------------------
    # Load resources and setup
    # -----------------------------------------------------------------------

    tg = tide_gauge_data(sta)
    if tg is None:
        return

    msl2000 = tg.res.loc["1991":"2009"].mean()

    # -----------------------------------------------------------------------
    # calculate mean annual cycle

    # drop months which have more than 10 days missing data
    tg = tg.groupby(pd.Grouper(freq="M")).apply(
        lambda x: None if x.res.isna().sum() > 24 * 10 else x
    )
    try:
        tg.index = tg.index.droplevel(0)
    except:
        pass

    mac = tg.sl.groupby(tg.index.month).mean()
    mac.sort_index(inplace=True)
    mac -= mac.mean()
    mac = mac.values

    # -----------------------------------------------------------------------

    # percentiles = [5, 17, 50, 83, 95]

    with open(sta_path + "noaa_thresholds.pickle", "rb") as f:
        noaa_thrsh = pickle.load(f)
    thresholds = [
        int(noaa_thrsh[z]) for z in noaa_thrsh if noaa_thrsh[z] <= 100
    ]

    # -----------------------------------------------------------------------

    if iso_mo in [1, 3, 5, 7, 8, 10, 12]:
        Ndy = 31
    elif iso_mo in [4, 6, 9, 11]:
        Ndy = 30
    else:
        Ndy = 28

    # if iso_mo is None:
    #     counts_first = [5, 10, 20, 50, 100]
    #     occN = counts_first[1]
    #     chcN = counts_first[3]
    # else:
    #     counts_first = [5, 10, 15, 25]
    #     occN = counts_first[1]
    #     chcN = counts_first[2]

    # -----------------------------------------------------------------------

    pid = sta["psmsl_id"]
    msl_prjn = {}

    scn_nm = ("int_low", "int", "int_high")

    fname = "../data/noaa_scenarios/NOAA_SLR_scenarios.pickle"
    with open(fname, "rb") as f:
        scn = pd.read_pickle(f)
    scn["proj"].index += 0.5

    scn = scn["proj"].loc[:, (pid, scn_nm, 50)] + msl2000
    scn.columns = scn.columns.get_level_values("Scenario")

    idx = np.arange(2010, 2101, 1 / 12) + 1 / 24
    noaa_prjn = pd.DataFrame(index=idx, columns=scn.columns)

    for s in scn_nm:
        spl = splrep(scn.index, scn[s].values)
        msl_prjn[s] = splev(noaa_prjn.index, spl)

    # noaa_prjn.plot(ax=plt.gca())
    # plt.show()

    # kopp_pth = "../../Data/Projections/Kopp/projectionsPT/"
    # fname = kopp_pth + "kopp14_rcp85_" + "{:0>4}".format(pid) + "_smpls.mat"
    # if os.path.exists(fname):
    #     rcp = hdf5.loadmat(fname)
    #     rcp["yr"] = rcp["yr"].flatten() + 0.5
    #     rcp["lmsl"] /= 10  # cm
    #     if iso_mo is None:
    #         kopp_prjn = pd.DataFrame(rcp["lmsl"], index=rcp["yr"])
    #         kopp_prjn = kopp_prjn.loc[idx]
    #     else:
    #         lmsl = np.reshape(
    #             np.tile(rcp["lmsl"], [12, 1, 1]),
    #             [rcp["lmsl"].shape[0] * 12, rcp["lmsl"].shape[1]],
    #             order="F",
    #         )
    #         kopp_prjn = pd.DataFrame(lmsl, index=idx)
    #     kopp_prjn += msl2000
    #     msl_prjn["kopp"] = kopp_prjn

    # -----------------------------------------------------------------------

    fname = sta_path + "gp_monthly.pickle"
    with open(fname, "rb") as f:
        gp_smpls = pickle.load(f)

    gp_idx = np.arange(2000, 2101, 1 / 12) + 1 / 24
    gp = pd.DataFrame(gp_smpls, index=gp_idx)
    gp = gp.loc[gp.index > 2010]
    gp = np.tile(gp.values, [1, int(1e4 / gp.values.shape[1])])

    # -----------------------------------------------------------------------

    td_path = "../tides/output/" + station_string(sta) + "/"
    fname = td_path + "tide_99th_percentile_monthly.pickle"

    with open(fname, "rb") as f:
        zeta99 = pickle.load(f)

    for k in zeta99.keys():
        if k != "obs":
            zeta99[k] = zeta99[k].loc["2010":"2100"]
            zeta99[k] = pd.DataFrame(
                np.tile(zeta99[k].values, [1, 10]), index=idx
            )

    # -----------------------------------------------------------------------

    mac = np.tile(mac, int(gp.shape[0] / 12))

    e99 = {
        s: zeta99[s + "_oc"]
        .add(gp, axis=0)
        .add(msl_prjn[s], axis=0)
        .add(mac, axis=0)
        for s in msl_prjn
    }

    # -----------------------------------------------------------------------

    mo_idx = np.arange(iso_mo - 1, e99["int"].shape[0], 12)
    e99 = {s: e99[s].iloc[mo_idx, :] for s in msl_prjn}
    for s in e99.keys():
        e99[s].index = e99[s].index.astype(int)

    # -----------------------------------------------------------------------

    mo_str = "mo" + "{:0>2}".format(iso_mo)
    tod_str = "h" + "{:0>2}".format(tod[0]) + "{:0>2}".format(tod[1])

    exp_str = mo_str + "_" + tod_str
    fname = sta_path + exp_str + "/bbmodel.pickle"

    with open(fname, "rb") as f:
        mod = pickle.load(f)

    trace, x_nrm = mod["trace"], mod["x_nrm"]

    # -----------------------------------------------------------------------
    # define rate of transition between GCDFs
    # -----------------------------------------------------------------------

    # should be same as defined in thrsh_exprmnts_bbmodel_2gcdf.py
    rt = 1 / 0.1

    # -----------------------------------------------------------------------
    # Get e99 corresponding to occasional and chronic probabilities
    # -----------------------------------------------------------------------

    # representative flat matrix of e99 (flat meaning constant rows)
    e99_flat = pd.DataFrame(
        np.tile(np.arange(-50, 101), [10000, 1]).T, index=np.arange(-50, 101)
    )

    # calculate exceedance days
    xd = xd_from_e99(e99_flat, trace, x_nrm, rt, Ndy)

    # -----------------------------------------------------------------------
    # Loop over each threshold and calculate projection statistics
    # -----------------------------------------------------------------------

    # calculate projections of exceedance days above each threshold
    for thrsh in thresholds:

        with open(log_file, "a") as lf:
            print(" ", file=lf)
            print("Threshold: " + str(thrsh), file=lf)

        for s in e99.keys():

            with open(log_file, "a") as lf:
                print("  Scenario: " + s, file=lf)

            # calculate threshold exceedance days
            rltv_e99 = pd.DataFrame(e99[s].values - thrsh, index=e99[s].index)
            xd = xd_from_e99(rltv_e99, trace, x_nrm, rt, Ndy)

        # -------------------------------------------------------------------

        if thrsh in thresholds:

            thrsh_str = [z for z in noaa_thrsh if noaa_thrsh[z] == thrsh][0]

            fname = (
                ens_path + exp_str + "/xdys_ensemble_" + thrsh_str + ".pickle"
            )
            os.makedirs(ens_path + exp_str, exist_ok=True)
            with open(fname, "wb") as f:
                # pickle.dump(p21["xd_ensemble"], f)
                pickle.dump(xd, f)

        # -------------------------------------------------------------------

    with open(log_file, "a") as lf:
        print(" ", file=lf)
        print("CASE complete.", file=lf)


# -----------------------------------------------------------------------
# Define functions describing mean (mu) of beta distributions
# -----------------------------------------------------------------------

# gaussian CDF (GCDF)
def gcdf(x0, loc0, sc0):
    return 0.5 * (1 + sp.special.erf((x0 - loc0) / (sc0 * np.sqrt(2))))


def logistic(x0, rt0, z0):
    # rt is the slope of transition, x0 is the location
    return 1 / (1 + np.exp(-rt0 * (x0 - z0)))


# -----------------------------------------------------------------------
# Function for calculating exceedance days from e99
# -----------------------------------------------------------------------


def xd_from_e99(e99, trace, x_nrm, rt, Ndy):

    # get height relative to threshold and normalize scale using x_nrm
    xp = (e99.values - x_nrm[0]) / (x_nrm[1] - x_nrm[0])

    # calculate mean (mu) of beta distributions
    mu_bta = np.zeros([xp.shape[0], trace["mu_loc1"].size])
    N_xd = np.zeros([xp.shape[0], trace["mu_loc1"].size])
    for k in range(trace["mu_loc1"].size):
        mu1 = logistic(xp[:, k], -rt, trace["z"][k]) * gcdf(
            xp[:, k], trace["mu_loc1"][k], trace["mu_sc1"][k]
        )
        mu2 = logistic(xp[:, k], rt, trace["z"][k]) * gcdf(
            xp[:, k], trace["mu_loc2"][k], trace["mu_sc2"][k]
        )
        mu_bta[:, k] = mu1 + mu2

    # calculate alpha/beta parameters of beta distributions
    v_bta = trace["nu"][None, :] * mu_bta * (1 - mu_bta)
    kappa = mu_bta * (1 - mu_bta) / v_bta - 1
    z = v_bta > 1e-15  # don't divide by zero
    z_min = ~z & (mu_bta < 0.5)
    z_max = ~z & (mu_bta > 0.5)
    mu_bta = mu_bta[z]
    v_bta = v_bta[z]
    kappa = mu_bta * (1 - mu_bta) / v_bta - 1
    a_bta = kappa * mu_bta
    b_bta = kappa * (1 - mu_bta)

    # get beta-distributed binomial probabilities
    p = np.zeros(xp.shape)
    p[z] = stats.beta.rvs(a_bta, b_bta)

    # get threshold exceedance days
    xd0 = stats.binom.rvs(Ndy, p).astype(float)
    xd0[z_min] = 0
    xd0[z_max] = Ndy
    xd = pd.DataFrame(xd0, index=e99.index).astype(int)

    return xd


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
