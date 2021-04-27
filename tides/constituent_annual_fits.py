# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import utide
from support_functions import station_string, ProgressBar

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

# Solve for major constituents individually in each year

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def constituent_annual_fits(sta, tg, cnst):

    yrs = tg.index.year.unique()
    yrs.name = "year"

    # # 37 standard noaa constituents below
    # cnst = ['M2', 'S2', 'N2', 'K1', 'M4', 'O1', 'M6', 'MK3', 'S4', 'MN4',
    #         'NU2', 'S6', 'MU2', '2N2', 'OO1', 'LAM2', 'S1', 'M1', 'J1', 'MM',
    #         'SSA', 'SA', 'MSF', 'MF', 'RHO', 'Q1', 'T2', 'R2', '2Q1', 'P1',
    #         '2SM2', 'M3', 'L2', '2MK3', 'K2', 'M8', 'MS4']

    # create containers for fits
    annual_fits = {}
    annual_fits["amp"] = pd.DataFrame(index=yrs, columns=cnst)
    annual_fits["phs"] = pd.DataFrame(index=yrs, columns=cnst)
    annual_fits["dgn"] = pd.DataFrame(index=yrs, columns=cnst)  # diagnostic
    annual_fits["msl"] = pd.Series(index=yrs)
    tg["td2"] = None
    coef = {}

    pbar = ProgressBar(
        yrs.size, description="Calculating annual tidal fits..."
    )

    for y in yrs:

        # first subtract the minor constituents from the year of data
        slyr = (tg.sl.loc[str(y)] - tg.utd_epch_mnr.loc[str(y)]).dropna()

        # only perform fit for years that are at least 75% complete
        if slyr.count() > 0.75 * 365 * 24:

            # time in julian days for utide
            dn = mdates.date2num(slyr.index.to_pydatetime())

            # solve for major constituents
            coef[y] = utide.solve(
                dn,
                slyr.values.data,
                lat=sta["lat"],
                method="ols",
                trend=False,
                nodal=False,
                verbose=False,
                constit=cnst,
            )

            # get indices of each major constituent
            cidx = [
                np.where(coef[y].name == c)[0][0]
                if c in coef[y].name
                else None
                for c in cnst
            ]

            # record the amplitudes and phases of each constituent
            annual_fits["amp"].loc[y, :] = [
                coef[y].A[ii] if ii is not None else None for ii in cidx
            ]
            annual_fits["phs"].loc[y, :] = [
                coef[y].g[ii] if ii is not None else None for ii in cidx
            ]
            annual_fits["dgn"].loc[y, :] = [
                coef[y].diagn["PE"][ii] if ii is not None else None
                for ii in cidx
            ]
            annual_fits["msl"].loc[y] = coef[y].mean

        pbar.update()

    # -----------------------------------------------------------------------
    # keep only major constituents

    drp_col = [c for c in annual_fits["amp"].columns.values if c not in cnst]
    annual_fits["amp"].drop(columns=drp_col, inplace=True)
    annual_fits["phs"].drop(columns=drp_col, inplace=True)
    annual_fits["dgn"].drop(columns=drp_col, inplace=True)

    # if the phases are wrapping around 0/360, adjust to avoid discontinuity
    for n in annual_fits["phs"].columns:
        cphs = annual_fits["phs"].loc[:, n].values
        if (np.sum(cphs > 315) > 0) and (np.sum(cphs < 45) > 0):
            cphs[cphs > 180] -= 360
            annual_fits["phs"].loc[:, n] = cphs

    # -----------------------------------------------------------------------
    # plot annual amplitudes and phases of major constituents

    sta_str = station_string(sta)
    fig_path = "./figures/" + sta_str + "/"
    os.makedirs(fig_path, exist_ok=True)

    fig0 = plt.figure(num="amplitudes", figsize=[8, 6])
    plt.clf()
    ax = plt.gca()
    annual_fits["amp"].plot(ax=ax, marker=".", lw=1, markersize=4)
    plt.ylabel("cm")
    plt.title(sta["name"] + ": Tidal amplitudes")
    plt.legend(bbox_to_anchor=(1.01, 1), loc="upper left")
    plt.tight_layout()
    # plt.show()
    fname = fig_path + "annual_fits_amplitude.pdf"
    fig0.savefig(fname)

    fig1 = plt.figure(num="phases", figsize=[8, 6])
    plt.clf()
    ax = plt.gca()
    annual_fits["phs"].plot(ax=ax, marker=".", lw=1, markersize=4)
    plt.ylabel("degrees")
    plt.title(sta["name"] + ": Tidal phases")
    # plt.show()
    fname = fig_path + "annual_fits_phase.pdf"
    fig1.savefig(fname)

    # -----------------------------------------------------------------------
    # reconstruct tide hindcast from annual fits

    pbar = ProgressBar(yrs.size, description="Reconstructing annual tides ...")
    for y in yrs:

        slyr = tg.loc[str(y), "sl"].dropna()
        if slyr.count() > 0.9 * 365 * 24:

            pred_cnst = [nm for nm in coef[y].name if nm in cnst]
            pred = utide.reconstruct(
                mdates.date2num(tg.loc[str(y)].index.to_pydatetime()),
                coef[y],
                verbose=False,
                constit=pred_cnst,
            )

            tg.loc[str(y), "td2"] = pred.h - annual_fits["msl"].loc[y]

        pbar.update()

    # -----------------------------------------------------------------------
    # add minor constituents from previous fit and annual cycle from least
    # squares fit; latter from basic_functions.py --> station_data()
    tg.td2 += tg.utd_epch_mnr
    tg.td2 += tg["acyc"]

    # -----------------------------------------------------------------------

    return tg, annual_fits


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
