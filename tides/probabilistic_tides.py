# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

import numpy as np
import pandas as pd

import matplotlib.dates as mdates
import matplotlib.pyplot as plt

from scipy.interpolate import splrep, splev

import utide

import pickle
import os
import time

from multiprocessing import Pool
from functools import partial

from support_functions import station_data, station_string, ProgressBar

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def main():

    # -----------------------------------------------------------------------

    fname = "../data/stations.pickle"
    with open(fname, "rb") as f:
        stations = pickle.load(f)

    # # restart
    # stations = stations.loc[72:, :]

    # -----------------------------------------------------------------------

    # remove stations to skip
    skip = [8771013]  # , 9410230, 9410660, 9410840, 9412110, 9413450, 9414750]
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

    lp = LoopInfo(stations.index.size)
    for idx, sta in stations.iterrows():
        lp.begin_iteration(sta)
        probabilistic_tides(sta)
        lp.end_iteration()


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


class LoopInfo:
    def __init__(self, N):
        self.N = N
        self.count = 0
        self.time_elapsed = 0
        self.t0 = 0

    def begin_iteration(self, sta):
        self.count += 1
        print(
            "\n["
            + str(self.count)
            + " of "
            + str(self.N)
            + "] "
            + sta["name"]
            + " (NOAA ID: "
            + str(sta["noaa id"])
            + ")"
        )
        self.t0 = time.time()

    def end_iteration(self):
        dt = time.time() - self.t0  # seconds
        self.time_elapsed += dt
        tm_sta_str = "{:0.1f}".format(dt / 60)
        tm_avg = self.time_elapsed / (60 * self.count)
        tm_avg_str = "{:0.1f}".format(tm_avg)
        tm_rem_mns = tm_avg * (self.N - self.count)
        tm_rem_hrs = int(np.floor(tm_rem_mns / 60))
        tm_rem_mns -= tm_rem_hrs * 60
        tm_rem_hrs_str = str(tm_rem_hrs)
        tm_rem_mns_str = "{:0.1f}".format(tm_rem_mns)
        print("Time this station: " + tm_sta_str + " mins")
        print("Average time/station so far: " + tm_avg_str + " mins")
        print(
            "Estimated time remaining: "
            + tm_rem_hrs_str
            + " hrs, "
            + tm_rem_mns_str
            + " mins"
        )


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# convert posterior amplitudes and phases into tidal variations;
# use utide to deal with greenwich phase lags;
# function below handles a single realization/sample from the posterior


def constituent_realizations(loop_args):

    global jyr_pred, jhr_pred, Ep, Em

    pstr_amp, pstr_phs = loop_args[0], loop_args[1]

    amp_rep = splrep(jyr_pred, pstr_amp)
    amp_ev = splev(jhr_pred, amp_rep)

    phs_rep = splrep(jyr_pred, pstr_phs)
    phs_ev = splev(jhr_pred, phs_rep)

    ap = 0.5 * amp_ev * np.exp(-1j * phs_ev * np.pi / 180)
    am = np.conj(ap)

    mod_td_cnst = np.real(Ep * ap + Em * am)

    return mod_td_cnst


# function for initializing parallel workers to avoid passing large arrays
# for every iteration of above function
def pool_initializer(base_args):

    global jyr_pred, jhr_pred, Ep, Em
    jyr_pred, jhr_pred, Ep, Em = (
        base_args[0],
        base_args[1],
        base_args[2],
        base_args[3],
    )


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def lowpass_wts(Tc, N=None):
    """
    Return weights for a windowed-sinc low-pass filter.

    Args:
        Tc (int): Filter cut-off period in units of the data time step.
        N (int): Length of the filter.

    Returns:
        wts: Filter weights normalized to unit sum.
        B: Limits of the transition band from full to zero power. Given as a
            two element array of periods in units of the data time step
    """
    fc = 1 / Tc  # cut-off frequency

    if N is None:
        N = 3 * Tc
    if not N % 2:
        N += 1  # make sure that N is odd

    # compute sinc filter
    n = np.arange(N)
    wts = np.sinc(2 * fc * (n - (N - 1) / 2.0))

    # compute Hamming window
    win = np.hamming(N)

    # multiply sinc filter and window
    wts = wts * win

    # normalize to get unity gain
    wts = wts / np.sum(wts)

    # calculate transition band
    b = 3.3 / N  # width of transition band in frequency for hamming window
    B = 1 / np.array([fc - b / 2, fc + b / 2])

    return wts, B


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def probabilistic_tides(sta):

    # -----------------------------------------------------------------------
    # set up for io

    sta_str = station_string(sta)

    out_path = "./Output/" + sta_str + "/"
    os.makedirs(out_path, exist_ok=True)

    fig_path = "./Figures/" + sta_str + "/"
    os.makedirs(fig_path, exist_ok=True)

    # -----------------------------------------------------------------------
    # load station data ('sl' in cm)
    # also includes stock NOAA tidal hindcast ('td')

    # tg.sl has units of cm above mhhw
    tg, acyc = station_data(sta, reuse_acyc_fit=True)

    # -----------------------------------------------------------------------
    # load info from initial tide fit

    out_name = out_path + "initial_tide_fit.pickle"
    with open(out_name, "rb") as f:
        ifit = pickle.load(f)

    mjr_cnst, coef_epch = ifit["mjr_cnst"], ifit["coef_epch"]

    # get minor constituents
    mnr_cnst = [
        c
        for c, p, s in zip(
            coef_epch.name, coef_epch.diagn["PE"], coef_epch.diagn["SNR"]
        )
        if (c not in mjr_cnst) and (s >= 2.0) and (c not in ["SA", "SSA"])
    ]

    # -------------------------------------------------------------------
    # gather arguments for tidal projections

    ngflgs = [
        coef_epch["aux"]["opt"]["nodsatlint"],
        coef_epch["aux"]["opt"]["nodsatnone"],
        coef_epch["aux"]["opt"]["gwchlint"],
        coef_epch["aux"]["opt"]["gwchnone"],
    ]

    tref = coef_epch["aux"]["reftime"]
    lat = coef_epch["aux"]["lat"]

    # -----------------------------------------------------------------------
    # setup hourly time vector

    dt_pred = pd.date_range(
        pd.Timestamp("1920-01-01"), pd.Timestamp("2101-01-01"), freq="H"
    )[:-1]

    if sta["noaa id"] == 1612340:  # honolulu
        z = dt_pred < pd.Timestamp("1947-06-13")
        zidx = dt_pred[z] + pd.Timedelta("30 m")
        dt_pred = zidx.append(dt_pred[~z])

    jhr_pred = mdates.date2num(dt_pred)

    # -----------------------------------------------------------------------
    # generate prediction of minor constituents for jhr_pred

    rc_mnr_pred = utide.reconstruct(
        jhr_pred, coef_epch, constit=mnr_cnst, verbose=False
    )
    rc_mnr_pred.h -= rc_mnr_pred.h.mean()

    # -----------------------------------------------------------------------
    # generate annual cycle variations for jhr_pred

    acyc_pred = acyc.acyc_pred(jhr_pred)
    acyc_pred -= acyc_pred.mean()

    # -----------------------------------------------------------------------
    # set up for obs and prjn scenarios

    out_name = out_path + "posterior_samples_" + mjr_cnst[0] + ".pickle"
    with open(out_name, "rb") as f:
        pstr = pickle.load(f)

    obs_jyr, prjn_amp, prjn_jyr = (
        pstr["jyr_obs"],
        pstr["amp_prjn"],
        pstr["jyr_prjn"],
    )

    # isolate scenarios
    iso_scn = ["int_low_oc", "int_oc", "int_high_oc", "kopp_oc"]
    if iso_scn is not None:
        prjn_amp = {s: prjn_amp[s] for s in prjn_amp if s in iso_scn}

    obs_d1 = mdates.date2num(
        mdates.num2date(
            mdates.datestr2num(
                str(mdates.num2date(obs_jyr[0]).year) + "-01-01"
            )
        )
    )
    obs_i1 = np.argmax(jhr_pred >= obs_d1)

    obs_d2 = mdates.date2num(
        mdates.num2date(
            mdates.datestr2num(
                str(mdates.num2date(obs_jyr[-1]).year + 1) + "-01-01"
            )
        )
    )
    obs_i2 = np.argmin(jhr_pred < obs_d2)

    obs_jhr = jhr_pred[obs_i1:obs_i2]
    obs_dt = dt_pred[obs_i1:obs_i2]

    prjn_i1 = np.argmax(jhr_pred >= mdates.datestr2num("2010-01-01"))

    prjn_jhr = jhr_pred[prjn_i1:]
    prjn_dt = dt_pred[prjn_i1:]

    # -----------------------------------------------------------------------
    # initialize dataframe for probabilistic tide realizations
    # initialize with deterministic minor constituents
    # the time mean of reconstructed minor consituents is zero

    N_pstr_smpls = 1000
    ptd = {}
    ptd["obs"] = pd.DataFrame(0, index=obs_dt, columns=range(N_pstr_smpls))
    ptd["obs"] = ptd["obs"].add(rc_mnr_pred.h[obs_i1:obs_i2], axis=0)
    for scn in prjn_amp:
        ptd[scn] = pd.DataFrame(0, index=prjn_dt, columns=range(N_pstr_smpls))
        ptd[scn] = ptd[scn].add(rc_mnr_pred.h[prjn_i1:], axis=0)

    cnst_h_rlzns = {}
    cnst99 = {}

    # -----------------------------------------------------------------------
    # loop over major constituents

    for cnst in mjr_cnst:

        print(cnst + ":", end="")
        agents = 5
        chunksize = 1

        # -------------------------------------------------------------------

        ii = np.where(coef_epch.name == cnst)[0][0]
        lind = coef_epch["aux"]["lind"][ii]

        # -------------------------------------------------------------------

        out_name = out_path + "posterior_samples_" + cnst + ".pickle"
        with open(out_name, "rb") as f:
            pstr = pickle.load(f)

        obs_amp, obs_phs, obs_jyr = (
            pstr["amp_obs"],
            pstr["phs_obs"],
            pstr["jyr_obs"],
        )
        prjn_amp, prjn_phs, prjn_jyr = (
            pstr["amp_prjn"],
            pstr["phs_prjn"],
            pstr["jyr_prjn"],
        )

        if iso_scn is not None:
            prjn_amp = {s: prjn_amp[s] for s in prjn_amp if s in iso_scn}
            prjn_phs = {s: prjn_phs[s] for s in prjn_phs if s in iso_scn}

        # -------------------------------------------------------------------
        # arguments to constituent_realizations() that stay the same every loop

        # obs period
        F, U, V = utide.harmonics.FUV(
            np.atleast_1d(obs_jhr), tref, np.atleast_1d(lind), lat, ngflgs
        )
        obs_Ep = (F * np.exp(1j * (U + V) * 2 * np.pi)).flatten()
        obs_Em = np.conj(obs_Ep)
        base_args_obs = {"obs": [obs_jyr, obs_jhr, obs_Ep, obs_Em]}

        # projection period
        F, U, V = utide.harmonics.FUV(
            np.atleast_1d(prjn_jhr), tref, np.atleast_1d(lind), lat, ngflgs
        )
        prjn_Ep = (F * np.exp(1j * (U + V) * 2 * np.pi)).flatten()
        prjn_Em = np.conj(prjn_Ep)
        base_args_prjn = {
            scn: [prjn_jyr, prjn_jhr, prjn_Ep, prjn_Em] for scn in prjn_amp
        }
        base_args = {**base_args_obs, **base_args_prjn}

        # -------------------------------------------------------------------
        # arguments to constituent_realizations() that change with every loop

        loop_args_obs = {
            "obs": [
                [obs_amp[:, n], obs_phs[:, n]] for n in range(N_pstr_smpls)
            ]
        }
        loop_args_prjn = {
            scn: [
                [prjn_amp[scn][:, n], prjn_phs[scn][:, n]]
                for n in range(N_pstr_smpls)
            ]
            for scn in prjn_amp
        }
        loop_args = {**loop_args_obs, **loop_args_prjn}

        # -------------------------------------------------------------------
        # parallelize

        # loop over each scenario
        for scn in loop_args:
            end = "\n" if scn == list(loop_args.keys())[-1] else ""
            print(" " + scn, end=end)
            # loop_func = partial(constituent_realizations, base_args[scn])
            with Pool(
                processes=agents,
                initializer=pool_initializer,
                initargs=[base_args[scn]],
            ) as pool:
                result = pool.map(
                    constituent_realizations, loop_args[scn], chunksize
                )
                pool.close()
                pool.join()
            ptd[scn] += np.vstack(result).T

            # cnst_h_rlzns[cnst] = pd.DataFrame(
            #     np.vstack(result).T, index=ptd[scn].index)
            # cnst99[cnst] = cnst_h_rlzns[cnst].groupby(
            #     pd.Grouper(freq='A')).quantile(0.99)#, axis=0)
            # cnst99[cnst].index = cnst99[cnst].index.year + 0.5
            # plt.figure(num=cnst)
            # plt.clf()
            # ax = plt.gca()
            # cnst99[cnst].plot(ax=ax)
            # plt.show()

    # ptd99 = ptd['kopp'].groupby(pd.Grouper(freq='A')).quantile(0.99, axis=0)
    # ptd99.index = ptd99.index.year + 0.5
    # plt.figure(num='ptd')
    # plt.clf()
    # ax = plt.gca()
    # ptd99.plot(ax=ax)
    # plt.show()
    #
    # def combo_plot(constits):
    #     combo_h = cnst_h_rlzns[constits[0]]*0
    #     for c in constits:
    #         combo_h += cnst_h_rlzns[c]
    #     combo_h99 = combo_h.groupby(
    #         pd.Grouper(freq='A')).quantile(0.99, axis=0)
    #     plt.figure(num='combo_'+ ''.join(constits)); plt.clf(); ax = plt.gca();
    #     combo_h99.plot(ax=ax); plt.show()
    #
    # def intrfnc_plot(c1, c2):
    #     h1 = cnst_h_rlzns[c1]
    #     h2 = cnst_h_rlzns[c2]
    #     L = 365*24
    #     x1 = np.arange(L)
    #     x2 = x1 + L + 1000
    #     col = plt.rcParams['axes.prop_cycle'].by_key()['color']
    #     plt.figure(num='intrfnc'); plt.clf();
    #     plt.plot(x1, h1.iloc[:L], color=col[0])
    #     plt.plot(x2, h1.iloc[-L:], color=col[0])
    #     plt.plot(x1, h2.iloc[:L], color=col[1])
    #     plt.plot(x2, h2.iloc[-L:], color=col[1])
    #     plt.plot(x1, (h1+h2).iloc[:L]-100, color='k')
    #     plt.plot(x2, (h1+h2).iloc[-L:]-100, color='k')
    #     plt.show()
    #
    # import ipdb; ipdb.set_trace()

    # -------------------------------------------------------------------
    # basic loop

    # print(cnst + ':', end='')
    # this_cnst = {}
    # for scn in ['kopp']:#loop_args:
    #     end = '\n' if scn == list(loop_args.keys())[-1] else ''
    #     print(' ' + scn, end=end)
    #     pool_initializer(base_args[scn])
    #     this_cnst[scn] = pd.DataFrame(np.zeros(ptd[scn].shape),
    #         index=ptd[scn].index)
    #     for n, args in enumerate(loop_args[scn]):
    #         mod_td_cnst = constituent_realizations(args)
    #         ptd[scn].iloc[:, n] += mod_td_cnst
    #         this_cnst[scn].iloc[:, n] = mod_td_cnst
    # plt.figure(num=cnst)
    # plt.clf()
    # ax = plt.gca()
    # this_cnst['kopp'].plot(ax=ax)
    # plt.show()

    # -----------------------------------------------------------------------
    # give each tide prediction units of cm above MHHW by matching MSL
    #   of the tide gauge data (already in cm above MHWW) over the NTDE.

    ptd_epch_mn = ptd["obs"].loc["1983":"2001", :].mean(axis=0)
    tg_epch_mn = tg.sl.loc["1983":"2001"].mean()
    for scn in ptd:
        ptd[scn] -= ptd_epch_mn
        ptd[scn] += tg_epch_mn

    # ----------------------------------------------------------------------
    # calculate and save mean and standard deviation of hourly tidal height

    ptd_summary = {
        scn: pd.concat(
            [
                ptd[scn].mean(axis=1).rename("mn"),
                ptd[scn].std(axis=1).rename("sd"),
            ],
            axis=1,
        )
        for scn in ptd
    }

    ptd_summary["acyc"] = {
        "obs": acyc_pred[obs_i1:obs_i2],
        "prjn": acyc_pred[prjn_i1:],
    }

    fname = out_path + "tide_prediction_mn_std.pickle"
    with open(fname, "wb") as f:
        pickle.dump(ptd_summary, f)

    # ----------------------------------------------------------------------
    # calculate and save annual standard deviations of high-pass filtered
    #   differences between probablistic tidal predictions and observations

    wts = lowpass_wts(48)[0]
    res = -ptd["obs"].reindex(tg.index).subtract(tg.sl, axis=0)
    res = res.apply(lambda x: x - np.convolve(x, wts, mode="same"), axis=0)
    std_res = res.groupby(pd.Grouper(freq="A")).apply(lambda x: x.std(axis=0))

    ptl = [2.5, 5, 17, 50, 83, 95, 97.5]
    qtl = [p / 100 for p in ptl]
    std_res_ptl = std_res.quantile(q=qtl, axis=1).T
    std_res_ptl.index = std_res_ptl.index.year
    std_res_ptl.index.name = "year"
    std_res_ptl.columns = ptl
    std_res_ptl.columns.name = "percentile"

    fname = out_path + "residuals_stdev_annual.pickle"
    with open(fname, "wb") as f:
        pickle.dump(std_res_ptl, f)

    # -----------------------------------------------------------------------
    # calculate and save annual 99th percentile of tidal height;
    # these values are relative to mhhw

    td99_yr = {
        scn: ptd[scn]
        .groupby(pd.Grouper(freq="A"))
        .quantile(0.99)
        .set_index(ptd[scn].index.year.unique())
        for scn in ptd
    }

    fname = out_path + "tide_99th_percentile_annual.pickle"
    with open(fname, "wb") as f:
        pickle.dump(td99_yr, f)

    # -----------------------------------------------------------------------
    # calculate and save monthly 99th percentile of tidal height;
    # these values are relative to mhhw

    td99_mo = {
        scn: ptd[scn].groupby(pd.Grouper(freq="MS")).quantile(0.99)
        for scn in ptd
    }
    for scn in td99_mo:
        td99_mo[scn].index = td99_mo[scn].index + pd.Timedelta(days=14)

    fname = out_path + "tide_99th_percentile_monthly.pickle"
    with open(fname, "wb") as f:
        pickle.dump(td99_mo, f)


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
