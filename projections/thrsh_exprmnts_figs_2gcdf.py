# ---------------------------------------------------------------------------

import os
import glob
import numpy as np
import scipy as sp
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt
import pickle

from support_functions import station_string

import pymc3 as pm
import theano.tensor as tt
from pymc3.gp.util import plot_gp_dist

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def main():

    # -----------------------------------------------------------------------

    fname = "../stations.pickle"
    with open(fname, "rb") as f:
        stations = pd.read_pickle(f)

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

    # # only do select stations
    # select = [1612340]
    # keep = [n for n in stations.index if stations.loc[n, "noaa id"] in select]
    # stations = stations.loc[keep]

    for idx, sta in stations.iterrows():

        print(str(idx) + ": " + sta["name"] + " (" + str(sta["noaa id"]) + ")")

        sta_str = station_string(sta)
        sta_path = "./output/" + sta_str + "/"
        fname = sta_path + "bbmodel.pickle"
        with open(fname, "rb") as f:
            p = pickle.load(f)
        trace, x_nrm, tru, x, y, rt = (
            p["trace"],
            p["x_nrm"],
            p["true"],
            p["xy"]["x"],
            p["xy"]["y"],
            p["rt"],
        )

        thrsh_exprmnts_figs(
            sta, trace, rt, x_nrm, tru, x, y, show_or_save="save"
        )


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def thrsh_exprmnts_figs(
    sta, Nhr, trace, rt, x_nrm, tru, x, y, exp_str, show_or_save="save"
):

    sta_str = station_string(sta)

    # -----------------------------------------------------------------------

    # gaussian CDF (GCDF)
    def gcdf(x0, loc0, sc0):
        return 0.5 * (1 + sp.special.erf((x0 - loc0) / (sc0 * np.sqrt(2))))

    def logistic(x0, rt0, z0):
        # rt is the slope of transition, x0 is the location
        return 1 / (1 + np.exp(-rt0 * (x0 - z0)))

    # -----------------------------------------------------------------------

    mu = np.zeros([tru["x"].size, trace["mu_loc1"].size])
    N_xd = np.zeros([tru["x"].size, trace["mu_loc1"].size])
    for k in range(trace["mu_loc1"].size):

        mu1 = logistic(tru["x"], -rt, trace["z"][k]) * gcdf(
            tru["x"], trace["mu_loc1"][k], trace["mu_sc1"][k]
        )
        mu2 = logistic(tru["x"], rt, trace["z"][k]) * gcdf(
            tru["x"], trace["mu_loc2"][k], trace["mu_sc2"][k]
        )
        mu[:, k] = mu1 + mu2

        v = trace["nu"][k] * mu[:, k] * (1 - mu[:, k])

        z = v > 1e-15  # don't divide by zero
        z_min = ~z & (mu[:, k] < 0.5)
        z_max = ~z & (mu[:, k] > 0.5)
        mu_z = mu[z, k]
        v_z = v[z]

        kappa = mu_z * (1 - mu_z) / v_z - 1
        a = mu_z * kappa
        b = (1 - mu_z) * kappa

        p_z = stats.beta.rvs(a, b)

        N_xd[z, k] = stats.binom.rvs(Nhr, p_z)
        N_xd[z_min, :] = 0
        N_xd[z_max, :] = Nhr

    mu_ci05 = np.percentile(mu, 5, axis=1)
    mu_ci95 = np.percentile(mu, 95, axis=1)

    N_xd_ci05 = np.percentile(N_xd, 5, axis=1)
    N_xd_ci95 = np.percentile(N_xd, 95, axis=1)

    e99_smpl = x_nrm[0] + (x_nrm[1] - x_nrm[0]) * x
    e99_tru = x_nrm[0] + (x_nrm[1] - x_nrm[0]) * tru["x"]

    # -----------------------------------------------------------------------

    msmtch = np.std(tru["mu"] - Nhr * np.percentile(mu, 50, axis=1))

    # -----------------------------------------------------------------------

    col = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    fig = plt.figure(num="bayes_betaBinom_mu", figsize=[10, 7.5])
    fig.clf()
    ax = fig.gca()

    ax.fill_between(
        e99_tru, N_xd_ci05, N_xd_ci95, color=col[1], alpha=0.5, lw=0
    )
    ax.fill_between(
        e99_tru, Nhr * mu_ci05, Nhr * mu_ci95, color=col[3], alpha=1.0, lw=0
    )
    ax.plot(e99_smpl, y, ".", color="gray")
    ax.plot(e99_tru, tru["mu"], "--k", lw=2)
    ax.set_title(str(sta["noaa id"]) + ": " + sta["name"])
    ax.set_xlabel("$\Delta_{99}$ (cm)")
    ax.set_ylabel("N$_{xd}$")

    if show_or_save == "save":
        fig_path = "./figures/bb_fits/" + sta_str + "/"
        os.makedirs(fig_path, exist_ok=True)
        fname = fig_path + exp_str + "_bb_fit.png"
        fig.savefig(fname)
        plt.close(fig)
    else:
        fig.show()

    # -----------------------------------------------------------------------

    # fig = plt.figure(num='trace_plot', figsize=[10, 7.5]); fig.clf();

    poss_v = ["z", "mu_loc1", "mu_sc1", "mu_loc2", "mu_sc2"]
    varnames = [v for v in trace.varnames if v in poss_v]
    # fig, ax = plt.subplots(nrows=len(varnames), ncols=2, num='trace_plot')
    pm.traceplot(trace, var_names=varnames)
    fig = plt.gcf()

    if show_or_save == "save":
        fig_path = "./figures/trace_plots/" + sta_str + "/"
        os.makedirs(fig_path, exist_ok=True)
        fname = fig_path + exp_str + "_trace_plot.png"
        fig.savefig(fname)
        plt.close(fig)
    else:
        fig.show()

    # -----------------------------------------------------------------------

    return msmtch


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
