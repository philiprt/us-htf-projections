import glob
import pickle

import numpy as np
import pandas as pd
import xarray as xr

import plotly.graph_objects as go
import plotly.subplots as psb

from scipy.interpolate import interp1d, splev, splrep

import analysis_module as anlyz


def fill_color(hex, opacity):
    hex = hex.lstrip("#")
    hlen = len(hex)
    return "rgba" + str(
        tuple(
            [
                int(hex[i : i + hlen // 3], 16)
                for i in range(0, hlen, hlen // 3)
            ]
            + [opacity]
        )
    )


def annual_projections(
    analysis, subset, quantity, yoi_srch_rng, yr_lims, show_tides_only=False
):

    prjn_subset = [
        analysis[loc["name"]][loc["scenario"]][loc["threshold"]]
        for loc in subset
    ]
    prjn_subset = prjn_subset[:4]

    # get YOIs
    for loc in prjn_subset:
        loc["steps"] = anlyz.yoi_steps(
            loc["xdys_ann_ptl"].loc[:, 50], search_range=yoi_srch_rng
        )

    stn_meta = [
        {
            **loc["station"],
            **loc["experiment"],
            **{"steps": loc["steps"]},
        }
        for loc in prjn_subset
    ]
    prjn_subset = [
        loc[quantity].loc[yr_lims[0] : yr_lims[1]] for loc in prjn_subset
    ]

    col = anlyz.color_palette()
    vspace = 0.15
    fig = psb.make_subplots(
        rows=2,
        cols=2,
        #     subplot_titles=[s["name"] for s in stn_meta],
        #         shared_xaxes=True,
        vertical_spacing=vspace,
    )

    annotations = []
    for n, prjn in enumerate(prjn_subset):
        leg = True if n == 0 else False
        r = int(n / 2) + 1
        c = (n % 2) + 1
        traces = projection_traces(
            prjn, stn_meta[n], col, leg, show_tides_only
        )
        annotations.extend(
            projection_annotations(
                prjn, stn_meta[n], r, c, vspace, show_tides_only
            )
        )
        for trc in traces:
            fig.add_trace(trc, row=r, col=c)

        fig.update_yaxes(title_text="days", row=r, col=c, side="right")
    #     fig.update_xaxes(range=yr_lims, row=r, col=c)

    fig.update_layout(
        width=800,
        height=525,
        template="none",
        margin=dict(
            l=25,
            r=60,
            b=20,
            t=100,
            pad=0,
        ),
        xaxis=dict(
            layer="below traces",
        ),
        yaxis=dict(
            layer="below traces",
        ),
        hovermode="x",
        #     hoverlabel_align="left",
        legend=dict(
            x=0.8,
            y=1.22,
            traceorder="reversed",
            itemclick=False,
            itemdoubleclick=False,
        ),
        title=dict(
            text="Projected High-Tide-Flooding Days",
            x=0.035,
            y=0.97,
            font=dict(size=24),
        ),
        annotations=annotations,
    )

    config = {"displayModeBar": False, "responsive": False}
    fig.show(config=config)

    return fig


def projection_traces(prjn, meta, col, leg, show_tides_only):

    col = [col[n] for n in [1, 5]]
    fcol = [fill_color(c, 0.4) for c in col if c[0] == "#"]

    prjn_traces = [
        {
            "x": prjn.index.values,
            "y": prjn[10].values,
            "type": "scatter",
            "fill": "none",
            "showlegend": False,
            "line": {"color": fcol[0], "width": 0},
            "hoverinfo": "y",
        },
        {
            "x": prjn.index.values,
            "y": prjn[90].values,
            "type": "scatter",
            "fill": "tonexty",
            "fillcolor": fcol[0],
            "showlegend": False,
            "mode": "none",
            "hoverinfo": "none",
        },
        {
            "x": prjn.index.values,
            "y": prjn[50].values,
            "type": "scatter",
            "name": "50th percentile" if leg else None,
            "showlegend": True if leg else False,
            "line": {"color": col[0], "width": 3},
            "hoverinfo": "y",
        },
        {
            "x": prjn.index.values,
            "y": prjn[90].values,
            "type": "scatter",
            "name": "90th percentile" if leg else None,
            "showlegend": True if leg else False,
            "line": {"color": col[1], "width": 3},
            "hoverinfo": "y",
        },
    ]

    if show_tides_only:
        tds_only_prjn = tides_only_projection(meta)
        tds_only_prjn = tds_only_prjn.loc[prjn.index]
        prjn_traces.append(
            {
                "x": tds_only_prjn.index.values,
                "y": tds_only_prjn.values,
                "type": "scatter",
                "name": "SLR & tides only" if leg else None,
                "showlegend": True if leg else False,
                "line": {
                    "color": "slategray",
                    "width": 3,
                },  # , "dash": "dash"},
                "hoverinfo": "y",
            }
        )

    if meta["steps"] is not None and not show_tides_only:

        annotation_traces = []
        yrs = [y for y in meta["steps"]]
        cmmn = {
            "type": "scatter",
            "mode": "lines",
            "showlegend": False,
            "hoverinfo": "none",
        }
        annotation_traces.append(
            {
                **cmmn,
                **{
                    "x": yrs,
                    "y": prjn[50].loc[yrs],
                    "type": "scatter",
                    "mode": "markers",
                    "showlegend": False,
                    "hoverinfo": "none",
                    "line": dict(color="black"),
                },
            }
        )
        annotation_traces.append(
            {
                **cmmn,
                **{
                    "x": [yrs[1]],
                    "y": [prjn[50].loc[yrs[1]]],
                    "type": "scatter",
                    "mode": "markers",
                    "name": "YOI" if leg else None,
                    "showlegend": True if leg else False,
                    "hoverinfo": "none",
                    "marker": dict(size=10, symbol="circle-open"),
                    "line": dict(color="black"),
                },
            }
        )

        for n, stp in enumerate(zip(meta["steps"][:-1], meta["steps"][1:])):
            lower = n < (len(meta["steps"]) - 2)
            dash = "dot" if lower else "solid"
            p12 = [stp[1], stp[0]] if lower else stp

            annotation_traces.append(
                {
                    **{
                        "x": [stp[0], p12[0]],
                        "y": prjn[50].loc[[stp[0], p12[1]]],
                        "line": dict(color="black", width=1, dash=dash),
                    },
                    **cmmn,
                }
            )
            annotation_traces.append(
                {
                    **{
                        "x": [p12[0], stp[1]],
                        "y": prjn[50].loc[[p12[1], stp[1]]],
                        "line": dict(color="black", width=1, dash=dash),
                    },
                    **cmmn,
                }
            )

        prjn_traces.extend(annotation_traces)

    return prjn_traces


def tides_only_projection(meta):

    sta_str = anlyz.station_string(meta)

    # -----------------------------------------------------------------------
    # load station data

    fname = "../data/tide_gauge/" + str(meta["noaa id"]) + ".nc"

    fname = glob.glob(fname)[0]

    tg = xr.open_dataset(fname).to_dataframe()
    tg.columns = ["sl", "td"]

    beg_end = tg.sl.loc[~tg.sl.isna()].iloc[[0, -1]].index
    tg = tg.loc[beg_end[0] : beg_end[1]]

    tg *= 100  # cm above MHHW

    # -----------------------------------------------------------------------
    # load thresholds

    sta_path = "../projections/output/" + sta_str + "/"
    fname = sta_path + "noaa_thresholds.pickle"
    with open(fname, "rb") as f:
        dthrsh = pickle.load(f)

    # -----------------------------------------------------------------------
    # load 21st century tide prediction

    fname = "../data/noaa_21cent_tides/" + str(meta["noaa id"]) + ".nc"
    td21 = xr.open_dataset(fname).to_dataframe() * 100
    td21 = td21.loc["2010":].predicted
    td21 -= tg.td.loc["1983":"2001"].mean()

    # -----------------------------------------------------------------------
    # load mean sea level projection

    pid = meta["psmsl_id"]

    fname = "../data/noaa_scenarios/NOAA_SLR_scenarios.pickle"
    with open(fname, "rb") as f:
        scn = pd.read_pickle(f)
    scn["proj"].index += 0.5

    scn = scn["proj"].loc[:, (pid, str(meta["scenario"]), 50)]
    scn.index = [pd.Timestamp(str(int(y)) + "-07-01") for y in scn.index]

    noaa_prjn = pd.Series(index=td21.index)

    spl = splrep(scn.index.to_julian_date(), scn.values)
    noaa_prjn.loc[:] = splev(td21.index.to_julian_date(), spl)
    noaa_prjn += tg.sl.loc["1991":"2009"].mean()

    # -----------------------------------------------------------------------
    # add slr and tides; get daily max; count exceedances

    dymx_prjn = (noaa_prjn + td21).groupby(pd.Grouper(freq="D")).max()
    htf_slrtd = dymx_prjn.groupby(pd.Grouper(freq="A")).apply(
        lambda x: (x > dthrsh[meta["threshold"]]).sum()
    )
    htf_slrtd.index = htf_slrtd.index.year

    return htf_slrtd


def projection_annotations(prjn, meta, r, c, vspace, show_tides_only):

    x = 2020

    dy = 0.05
    y = 1 + dy if r == 1 else (1 - vspace) / 2 + dy

    cmmn = {
        "x": x,
        "xref": "x" + str(c),
        "yref": "paper",
        "xanchor": "left",
        "yanchor": "top",
        "bgcolor": "white",
        "borderpad": 2,
        "showarrow": False,
    }
    annotations = [
        {
            **{
                "y": y - 0.055,
                "text": "Threshold: NOAA " + meta["threshold"].capitalize(),
            },
            **cmmn,
        },
        {
            **{
                "y": y,
                "text": "<b>" + meta["name"] + "<b>",  # ,
                "font": dict(size=16),
            },
            **cmmn,
        },
    ]

    if meta["steps"] is not None and not show_tides_only:

        for n, stp in enumerate(zip(meta["steps"][:-1], meta["steps"][1:])):

            dlt = prjn[50].loc[stp[1]] - prjn[50].loc[stp[0]]
            day_or_days = " day/year" if dlt == 1 else " days/year"
            inc_str = u"Δ = " + str(int(dlt)) + day_or_days

            if prjn[50].loc[stp[1]] <= 1:
                step_str = (
                    str(stp[0]) + u" → " + str(stp[1]) + ": " + "Few events"
                )
            else:
                step_str = str(stp[0]) + u" → " + str(stp[1]) + ": " + inc_str
            annotations.append(
                {
                    **{"y": y - 0.01 - (n + 2) * 0.045, "text": step_str},
                    **cmmn,
                },
            )

    if (r == 1) & (c == 1):
        scenario_strings = {
            "int_low": "NOAA Intermediate Low",
            "int": "NOAA Intermediate",
            "int_high": "NOAA Intermediate High",
            "kopp": "Kopp et al. (2014)",
        }
        annotations.append(
            {
                **cmmn,
                **{
                    "y": y + 0.105,
                    "text": scenario_strings[meta["scenario"]]
                    + " SLR Scenario",
                    "font": dict(size=14),
                    "bgcolor": None,
                },
            },
        )

    return annotations


def annual_tides_only():

    sta_path = (
        "../Output/json/"
        + str(rw.loc[("Station Metadata", "Location", "NOAA ID")])
        + "/"
    )
    fname = sta_path + "noaa_thresholds.json"
    with open(fname, "r") as f:
        dthrsh = json.load(f)
