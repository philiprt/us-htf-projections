# ---------------------------------------------------------------------------

import sys
import glob
import os
import shutil
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm
import theano.tensor as tt
from pymc3.gp.util import plot_gp_dist

from multiprocessing import Pool
import logging

from support_functions import station_string
from thrsh_exprmnts_figs_2gcdf import thrsh_exprmnts_figs
from tide_gauge_data import *

# suppress some 'future warnings' internal to arviz (pymc3 plotting)
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

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
    skip = [8771013]
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
    # select = [8747437, 8656483, 8652587, 9440910, 8631044, 8636580]
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
                    # manage_experiments([sta, mo, tod, mode])
                    thrsh_exprmnts_bbmodel([sta, mo, tod, mode])

    # -----------------------------------------------------------------------

    elif mode == "multiprocess":

        try:
            shutil.rmtree("./progress/")
        except:
            pass
        os.makedirs("./progress/")

        exp_list = []
        for n in stations.index:
            for mo in mo_options:
                for tod in tod_options:
                    exp_list.append([stations.loc[n], mo, tod, mode])

        agents = 24
        chunksize = 1
        with Pool(processes=agents) as pool:
            result = pool.map(manage_experiments, exp_list, chunksize)


# ---------------------------------------------------------------------------


def manage_experiments(experiments):

    sta, iso_mo, tod, mode = (
        experiments[0],
        experiments[1],
        experiments[2],
        experiments[3],
    )

    # -----------------------------------------------------------------------
    # check if analysis has already been completed, and skip if so

    sta_str = station_string(sta)
    mo_str = "mo" + "{:0>2}".format(iso_mo)
    tod_str = "h" + "{:0>2}".format(tod[0]) + "{:0>2}".format(tod[1])
    exp_str = mo_str + "_" + tod_str
    sta_path = "./output/" + sta_str + "/"
    output_fname = fname = sta_path + exp_str + "/bbmodel.pickle"
    if os.path.exists(output_fname):
        return

    # -----------------------------------------------------------------------
    # run model up to ten times to get a satisfactory fit

    tries = 0
    success = False

    while (tries < 10) & (success is False):

        try:
            thrsh_exprmnts_bbmodel(experiments)
            success = True

        except:
            tries += 1

    if success is False:
        log_dir = "./progress/"
        log_file = log_dir + "_FAILED_" + sta_str + mo_str + tod_str + ".txt"
        lf = open(log_file, "w")
        print("Model fitting failed 10 consecutive times.", file=lf)
        lf.close()


# ---------------------------------------------------------------------------


def thrsh_exprmnts_bbmodel(experiments):

    # -----------------------------------------------------------------------

    sta, iso_mo, tod, mode = (
        experiments[0],
        experiments[1],
        experiments[2],
        experiments[3],
    )
    tg = tide_gauge_data(sta)
    if tg is None:
        return

    sta_str = station_string(sta)

    # -----------------------------------------------------------------------
    # setup for logging

    if mode == "multiprocess":

        mo_str = "m" + "{:02}".format(iso_mo)
        tod_str = "_t" + "{:02}".format(tod[0]) + "{:02}".format(tod[1])
        log_dir = "./progress/" + sta_str + "/"
        os.makedirs(log_dir, exist_ok=True)
        log_file = log_dir + mo_str + tod_str + ".txt"
        lf = open(log_file, "w")

        pm_logger = logging.getLogger("pymc3")
        pm_logger.removeHandler(pm_logger.handlers[0])
        pm_handler = logging.StreamHandler(lf)
        pm_logger.addHandler(pm_handler)

        th_log_file = log_dir + "th_" + mo_str + tod_str + ".txt"
        thlf = open(th_log_file, "w")
        th_logger = logging.getLogger("theano")
        th_logger.removeHandler(th_logger.handlers[0])
        th_handler = logging.StreamHandler(thlf)
        th_logger.addHandler(th_handler)

        print(sta["name"] + " (NOAA ID: " + str(sta["noaa id"]) + ")", file=lf)
        print("CASE: iso_mo = " + str(iso_mo) + "; tod = " + str(tod), file=lf)

    # -----------------------------------------------------------------------
    # station output directory

    sta_path = "./output/" + sta_str + "/"
    os.makedirs(sta_path, exist_ok=True)

    # -----------------------------------------------------------------------
    # detrend residuals

    tg.res -= tg.trnd

    # -----------------------------------------------------------------------

    freq = "MS"  # if iso_mo is not None else "A"
    min_req = 26 * 24  # if iso_mo is not None else np.ceil(365 * 0.95)

    # drop months which have more than 5 days missing data
    tg = tg.groupby(pd.Grouper(freq=freq)).apply(
        lambda x: None if x.sl.dropna().count() < min_req else x
    )
    try:
        tg.index = tg.index.droplevel(0)
    except:
        pass

    # -----------------------------------------------------------------------

    eta_bar = tg.res.groupby(pd.Grouper(freq=freq)).mean()

    sta_str = station_string(sta)

    fname = (
        "../tides/output/" + sta_str + "/tide_99th_percentile_monthly.pickle"
    )
    with open(fname, "rb") as f:
        pzeta_99 = pickle.load(f)

    zeta_99 = pzeta_99["obs"].quantile(0.5, axis=1)
    zeta_99.index -= pd.Timedelta(days=14)

    eta_bar = eta_bar.loc[: zeta_99.index[-1]]
    zeta_99 = zeta_99.reindex(eta_bar.index)

    eta_99 = eta_bar + zeta_99
    eta_99.dropna(inplace=True)

    # -----------------------------------------------------------------------

    thrsh = np.arange(-100, 51)

    Nmo = eta_99.size
    exp_delta_99 = np.zeros((Nmo, thrsh.size))
    exp_N_xd = np.zeros((Nmo, thrsh.size))
    exp_mo = np.zeros((Nmo, thrsh.size))
    exp_dycnt = np.zeros((Nmo, thrsh.size))

    tg.loc[tg.res.isna(), "res"] = 0
    hrly = tg.res + tg.td

    itod = (hrly.index.hour >= tod[0]) & (hrly.index.hour <= tod[1])
    hrly = hrly.loc[itod]

    dymx = hrly.groupby(pd.Grouper(freq="D")).max()

    for k, z in enumerate(thrsh):
        N_xd = (
            dymx.groupby(pd.Grouper(freq=freq))
            .apply(lambda x: np.sum(x > z))
            .loc[eta_99.index]
        )
        exp_N_xd[:, k] = N_xd
        exp_delta_99[:, k] = eta_99 - z
        exp_mo[:, k] = N_xd.index.month
        exp_dycnt[:, k] = (
            dymx.groupby(pd.Grouper(freq=freq)).count().loc[eta_99.index]
        )

    exp = pd.DataFrame(
        {
            "delta_99": exp_delta_99.flatten(),
            "N_xd": exp_N_xd.flatten(),
            "mo": exp_mo.flatten(),
            "dycnt": exp_dycnt.flatten(),
        }
    )
    exp.sort_values("delta_99", inplace=True)
    exp.reset_index(drop=True, inplace=True)

    # -----------------------------------------------------------------------

    # plt.figure(num='test')
    # plt.clf()
    # for m in [3]:#[1, 4, 7, 10]:
    #     z = exp['mo'] == m
    #     plt.plot(exp.loc[z, 'delta_99'], exp.loc[z, 'N_xd'], '.')
    # plt.show()
    # import sys; sys.exit()

    # -----------------------------------------------------------------------

    if iso_mo in [1, 3, 5, 7, 8, 10, 12]:
        Ndy = 31
    elif iso_mo in [4, 6, 9, 11]:
        Ndy = 30
    else:
        Ndy = 28
        exp["N_xd"].loc[exp["N_xd"] > 28] = 28

    exp = exp.loc[(exp.mo == iso_mo) & (exp.dycnt == Ndy)]

    # -----------------------------------------------------------------------
    # normalization extents for delta_99 dimension

    fname = sta_path + "bbmodel_xnrm.pickle"

    upper = Ndy  # /2 if (tod[1] - tod[0] + 1) <= 6 else Ndy/3
    # upper = Ndy/3

    mn = np.floor(exp.loc[exp.N_xd > 1, "delta_99"].iloc[0])
    mx = np.ceil(exp.loc[exp.N_xd < (upper - 1), "delta_99"].iloc[-1])
    mn -= 0.25 * (mx - mn)
    mx += 0.25 * (mx - mn)
    x_nrm = [mn, mx]

    # -----------------------------------------------------------------------

    exp = exp.loc[(exp.delta_99 > x_nrm[0]) & (exp.delta_99 < x_nrm[1])]
    exp["x"] = (exp.delta_99 - x_nrm[0]) / (x_nrm[1] - x_nrm[0])

    bins = np.arange(0.01, 1.0, 0.01)
    mu_tru = np.zeros(bins.size)
    # v_tru = np.zeros(bins.size)
    for bi, b in enumerate(bins):
        bz = (exp.x > b - 0.01) & (exp.x < b + 0.01)
        mu_tru[bi] = exp.N_xd.loc[bz].mean()
        # v_tru[bi] = exp.N_xd.loc[bz].mean()
    tru = {"x": bins, "mu": mu_tru}

    # thin so each year used approximately once every 10 cm of eta_99
    frac = (
        (x_nrm[1] - x_nrm[0])
        * tg.index.year.unique().size
        / (10 * exp.index.size)
    )
    exp_thin = exp.sample(frac=frac)
    exp_thin.sort_values("x", inplace=True)

    x = exp_thin.x.values
    y = exp_thin.N_xd.values

    # plt.figure(num="test")
    # plt.clf()
    # plt.plot(x, y, ".r")
    # # # test = -0.05 + 0.4*x
    # # # plt.plot(x, Ndy*test, 'k')
    # plt.show()
    # import ipdb
    #
    # ipdb.set_trace()

    # -----------------------------------------------------------------------

    # def ngpdcdf(x0, loc0, sc0, zta0):
    #     z = -(x0 - loc0)/sc0
    #     return (1 + zta0*z)**(-1/zta0)
    #
    # test = ngpdcdf(x, 1.05, 0.02, 0.5)
    # plt.figure(num='gpd_test')
    # plt.clf()
    # plt.plot(x, y, '.')
    # plt.plot(x, 365*test)
    # plt.show()
    #
    # import sys; sys.exit()

    # -----------------------------------------------------------------------

    with pm.Model() as model:

        # -------------------------------------------------------------------

        # gaussian CDF (GCDF)
        def gcdf(x0, loc0, sc0):
            return 0.5 * (1 + tt.erf((x0 - loc0) / (sc0 * np.sqrt(2))))

        # -------------------------------------------------------------------
        # function for smoothing transition across changepoint

        def logistic(x0, rt0, z0):
            # rt is the slope of transition, x0 is the location
            return pm.math.invlogit(rt0 * (x0 - z0))

        # -------------------------------------------------------------------

        # location of switch point between GCDFs
        z = pm.Uniform("z", lower=0.0, upper=1.0, testval=0.5)

        # distance of GCDF location parameters from switch point
        dloc = pm.HalfNormal("dloc", sd=0.1, testval=0.01)

        # GCDF1 location prior
        # mu_loc1 = pm.Uniform("mu_loc1", lower=0.0, upper=1.0, testval=0.45)
        mu_loc1 = pm.Deterministic("mu_loc1", z - dloc)

        # GCDF1 scale prior
        mu_sc1 = pm.Uniform("mu_sc1", lower=0.0, upper=0.5)

        # GCDF2 location prior
        # mu_loc2 = pm.Uniform("mu_loc2", lower=0.0, upper=1.0, testval=0.55)
        mu_loc2 = pm.Deterministic("mu_loc2", z + dloc)

        # GCDF2 scale prior
        mu_sc2 = pm.Uniform("mu_sc2", lower=0.0, upper=0.5)

        # -------------------------------------------------------------------

        # rate of transition between GCDFs (1/lengthscale in normalized units)
        rt = 1 / 0.1

        mu1 = logistic(x, -rt, z) * gcdf(x, mu_loc1, mu_sc1)
        mu2 = logistic(x, rt, z) * gcdf(x, mu_loc2, mu_sc2)

        mu = mu1 + mu2

    # -----------------------------------------------------------------------

    with model:

        # -------------------------------------------------------------------
        # variance: related to mu by parameter nu

        # nu prior
        nu = pm.Uniform("nu", lower=0.0, upper=1.0)

        # variance of the beta distribution
        v = nu * mu * (1 - mu)

        # -------------------------------------------------------------------
        # distribution of observations conditional on alpha(x) and beta(x)

        # reformulate beta dist parameters in terms of alpha/beta
        kappa = mu * (1 - mu) / v - 1
        a = mu * kappa
        b = (1 - mu) * kappa

        y_ = pm.BetaBinomial("y", alpha=a, beta=b, n=Ndy, observed=y)

        # -------------------------------------------------------------------

        if mode == "multiprocess":

            trace = pm.sample(
                2500,
                chains=4,
                cores=1,
                tune=5000,
                target_accept=0.9,
                progressbar=False,
            )

        elif mode == "test":

            trace = pm.sample(
                2500, chains=4, cores=4, tune=5000, target_accept=0.9
            )

        # -------------------------------------------------------------------

        # do not accept this fit if there are many divergences
        if np.sum(trace.get_sampler_stats("diverging")) > 500:
            raise Exception("Too many divergences.")

    # -----------------------------------------------------------------------

    mo_str = "mo" + "{:0>2}".format(iso_mo)

    tod_str = "h" + "{:0>2}".format(tod[0]) + "{:0>2}".format(tod[1])

    exp_str = mo_str + "_" + tod_str

    os.makedirs(sta_path + exp_str, exist_ok=True)
    fname = sta_path + exp_str + "/bbmodel.pickle"
    with open(fname, "wb") as f:
        pickle.dump(
            {
                "trace": trace,
                "x_nrm": x_nrm,
                "true": tru,
                "xy": {"x": x, "y": y},
                "freq": freq,
                "Ndy": Ndy,
                "rt": rt,
            },
            f,
        )

    # -----------------------------------------------------------------------

    sos = "save" if mode == "multiprocess" else "show"

    msmtch = thrsh_exprmnts_figs(
        sta, Ndy, trace, rt, x_nrm, tru, x, y, exp_str, show_or_save="save"
    )
    # print('Mismatch: ' + str(msmtch))

    # -----------------------------------------------------------------------

    # return trace, bins

    # -----------------------------------------------------------------------

    # close log file
    if mode == "multiprocess":
        lf.close()
        thlf.close()


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
