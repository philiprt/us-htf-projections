# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

import numpy as np
import pandas as pd
import glob
import xarray as xr

import utide
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# create a string describing station for directories/filenames


def station_string(sta):

    loc_str = "".join(
        [
            c if c != " " else "_"
            for c in [c for c in sta["name"].lower() if c != ","]
        ]
    )

    sta_str = loc_str + "_" + str(sta["noaa id"])

    return sta_str


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# class for calculating and reusing annual cycle fits


class AnnualCycleFit:
    def __init__(self, tg):

        jdt = mdates.date2num(tg.index)

        self.tref = jdt[0]

        jdt -= self.tref

        h = (tg.sl - tg.td).values.astype(float)
        z = ~np.isnan(h)

        # calculate annual cycle and trend in residuals
        phs = jdt * (2 * np.pi) / 365.25
        harm = np.vstack(
            [np.sin(phs), np.cos(phs), np.sin(2 * phs), np.cos(2 * phs)]
        )
        A = np.vstack([np.ones(jdt.size), jdt, harm]).T
        self.c = np.linalg.lstsq(A[z, :], h[z], rcond=None)[0]

        self.annual_cycle = np.sum(A[:, 2:] * self.c[2:], axis=1)
        self.trend = np.sum(A[:, 0:2] * self.c[0:2], axis=1)

    def acyc_pred(self, jdt):

        jdt0 = jdt - self.tref
        phs = jdt0 * (2 * np.pi) / 365.25
        harm = np.vstack(
            [np.sin(phs), np.cos(phs), np.sin(2 * phs), np.cos(2 * phs)]
        )
        A = np.vstack([np.ones(jdt0.size), jdt0, harm]).T
        acyc = np.sum(A[:, 2:] * self.c[2:], axis=1)

        return acyc


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# load station data


def station_data(sta, reuse_acyc_fit=False):

    fname = "../data/tide_gauge/" + str(sta["noaa id"]) + ".nc"

    fname = glob.glob(fname)
    if len(fname) == 0:
        tg = None
        # return tg
    else:
        fname = fname[0]

    tg = xr.open_dataset(fname).to_dataframe()
    tg.columns = ["sl", "td"]

    if sta["noaa id"] == 1612340:  # honolulu
        tidx = tg.index
        z = tidx < pd.Timestamp("1947-06-13")
        zidx = tidx[z] + pd.Timedelta("30 m")
        tg.index = zidx.append(tidx[~z])

    if sta["noaa id"] == 8545240:  # philadelphia
        tg.loc["1942", :] = None

    tg.dropna(inplace=True)
    tg = tg.loc[:"2019"]
    tg *= 100  # cm above mhhw

    # -----------------------------------------------------------------------
    # calculate annual cycle and trend

    acyc = AnnualCycleFit(tg)
    tg["acyc"] = acyc.annual_cycle
    tg["trnd"] = acyc.trend

    # -----------------------------------------------------------------------

    if reuse_acyc_fit:
        return tg, acyc
    else:
        return tg


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# initial tide fit to distnguish major/minor constituents


def initial_tide_fit(sta, tg):

    sl_epch = tg.loc["2012":"2016", "sl"].dropna()

    # fit all constituents over 5 yr period
    coef_epch = utide.solve(
        mdates.date2num(sl_epch.index.to_pydatetime()),
        sl_epch.values.data,
        lat=sta["lat"],
        method="ols",
        trend=True,
        nodal=False,
        verbose=False,
    )

    coef_epch.slope = 0  # zero out the trend for reconstructing the prediction

    # the following are always considered major constituents
    dflt_mjr_cnst = ["M2", "S2", "N2", "K2", "K1", "O1", "P1", "Q1"]

    # add additional major constituents if sufficient fraction of tidal energy
    mjr_cnst = [
        c
        for c, p, s in zip(
            coef_epch.name, coef_epch.diagn["PE"], coef_epch.diagn["SNR"]
        )
        if (c in dflt_mjr_cnst)
        or (((p >= 0.5) and (s >= 2.0)) and (c not in ["SA", "SSA"]))
    ]

    # all other constituents with signal:noise >= 2 are considered minor
    mnr_cnst = [
        c
        for c, p, s in zip(
            coef_epch.name, coef_epch.diagn["PE"], coef_epch.diagn["SNR"]
        )
        if (c not in mjr_cnst) and (s >= 2.0) and (c not in ["SA", "SSA"])
    ]

    # reconstruct tidal levels from complete set of constituents
    rc_epch = utide.reconstruct(
        mdates.date2num(tg.index.to_pydatetime()),
        coef_epch,
        verbose=False,
        constit=mjr_cnst + mnr_cnst,
    )

    # reconstruct tidal levels from minor constituents only
    rc_epch_mnr = utide.reconstruct(
        mdates.date2num(tg.index.to_pydatetime()),
        coef_epch,
        verbose=False,
        constit=mnr_cnst,
    )

    # add reconstructed levels to dataframes
    tg["utd_epch"] = rc_epch.h
    tg.utd_epch -= tg.utd_epch.loc["1983":"2001"].mean()

    tg["utd_epch_mnr"] = rc_epch_mnr.h
    tg.utd_epch_mnr -= tg.utd_epch_mnr.loc["1983":"2001"].mean()

    # plt.figure(num='predictions0')
    # plt.clf()
    # ax = plt.gca()
    # tg.td.plot(ax=ax)
    # tg.utd_epch.plot(ax=ax)
    # tg.utd_epch_mnr.plot(ax=ax)
    # plt.ylabel('cm')
    # plt.title(sta['name'] + ': Tidal heights')
    # plt.show()
    #
    # import sys; sys.exit()

    return tg, mjr_cnst, coef_epch


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
# the following is adapted from the pymc3 package


def plot_gp_dist(
    x,
    samples,
    ax=None,
    plot_samples=True,
    palette="Reds",
    fill_alpha=1.0,
    samples_alpha=0.1,
    fill_kwargs=None,
    samples_kwargs=None,
):
    """A helper function for plotting 1D GP posteriors from trace

        Parameters
    ----------
    ax : axes
        Matplotlib axes.
    samples : trace or list of traces
        Trace(s) or posterior predictive sample from a GP.
    x : array
        Grid of X values corresponding to the samples.
    plot_samples: bool
        Plot the GP samples along with posterior (defaults True).
    palette: str
        Palette for coloring output (defaults to "Reds").
    fill_alpha : float
        Alpha value for the posterior interval fill (defaults to 0.8).
    samples_alpha : float
        Alpha value for the sample lines (defaults to 0.1).
    fill_kwargs : dict
        Additional arguments for posterior interval fill (fill_between).
    samples_kwargs : dict
        Additional keyword arguments for samples plot.
    Returns
    -------
    ax : Matplotlib axes
    """

    if ax is None:
        ax = plt.gca()
    if fill_kwargs is None:
        fill_kwargs = {}
    if samples_kwargs is None:
        samples_kwargs = {}

    cmap = plt.get_cmap(palette)
    percs = np.linspace(51, 100, 50)
    colors = (percs - np.min(percs)) / (np.max(percs) - np.min(percs))
    percs = percs[:-10]
    colors = colors[3:-7]
    x = x.flatten()
    for i, p in enumerate(percs[::-1]):
        upper = np.percentile(samples, p, axis=1)
        lower = np.percentile(samples, 100 - p, axis=1)
        color_val = colors[i]
        h = ax.fill_between(
            x,
            upper,
            lower,
            color=cmap(color_val),
            alpha=fill_alpha,
            lw=0,
            **fill_kwargs
        )
        if i == 20:
            h_pctl = h
    if plot_samples:
        # plot a few samples
        idx = np.random.randint(0, samples.shape[1], 15)
        h_smpl = ax.plot(
            x,
            samples[:, idx],
            color=cmap(0.9),
            lw=0.5,
            alpha=0.2,
            **samples_kwargs
        )

    return ax, h_pctl, h_smpl


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# make a progress bar for loops


class ProgressBar:
    def __init__(self, total, description=None):

        self.total = total
        self.decimals = 1
        self.length = 50
        self.fill = ">"
        self.iteration = 0
        self.intervals = np.ceil(np.linspace(1, self.total, self.length))
        self.percent = np.linspace(0, 100, self.length + 1)[1:].astype(int)

        unq, cnt = np.unique(self.intervals, return_counts=True)
        idx = np.array([np.where(self.intervals == u)[0][-1] for u in unq])
        self.intervals = self.intervals[idx]
        self.percent = self.percent[idx]
        self.cumcnt = np.cumsum(cnt)

        if description:
            print(description)
        print("|%s| 0%% complete" % (" " * self.length), end="\r")

    def update(self):

        self.iteration += 1

        if self.iteration in self.intervals:
            prog = np.where(self.intervals == self.iteration)[0][-1]
            pctstr = str(self.percent[prog]) + "%"
            bar = self.fill * self.cumcnt[prog] + " " * (
                self.length - self.cumcnt[prog]
            )
            print("\r|%s| %s complete" % (bar, pctstr), end="\r")
            if self.iteration == self.total:
                print()  # blank line on completion


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
