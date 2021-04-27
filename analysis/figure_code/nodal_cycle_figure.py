import os
import glob
import pickle
import xarray as xr
import numpy as np
from scipy.interpolate import interp1d, splev, splrep
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import analysis_module as anlyz


def hover_template(is_minor):
    return (
        "<b>%{customdata[0]}</b><br>"
        + "Region: %{customdata[1]}<br>"
        + "Nodal cycle range: %{customdata[2]:0.1f} cm<br>"
        + "SLR during 2030s: %{customdata[3]:0.1f} cm"
        + "<extra></extra>"
    )


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


def load_ncyc_vs_slr(station):

    sta = pd.Series({"name": station.name, "noaa id": station["NOAA ID"]})
    sta_str = anlyz.station_string(sta)

    fname = "../analysis/nodal_cycle/" + sta_str + "/nodal_cycle_vs_slr.pickle"

    with open(fname, "rb") as f:
        d = pickle.load(f)

    try:
        station["ncyc"] = d["ncyc"]
        station["slr"] = d["slr_prjn"]
        station["ratio"] = d["ncyc"] / d["slr_prjn"]
    except:
        station["ncyc"] = None
        station["slr"] = None
        station["ratio"] = None

    return station


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

    sta_str = anlyz.station_string(sta)
    fname = "../tides/output/" + sta_str + "/tide_prediction_mn_std.pickle"
    with open(fname, "rb") as f:
        ptd = pickle.load(f)

    tg = tg.loc[: ptd["obs"].index[-1]]
    tg.td = ptd["obs"].reindex(tg.index)

    msl = tg.td.loc["1983":"2001"].mean()

    tg.td -= tg.td.loc["1983":"2001"].mean()
    tg.td += tg.sl.loc["1983":"2001"].mean()

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

    return tg, msl


def nodal_cycle_example():

    sta = anlyz.station_list(select=8726520)
    sta = sta.iloc[0]
    sta_str = anlyz.station_string(sta)

    # -----------------------------------------------------------------------
    # Directories for I/O

    # station directory
    sta_path = "../projections/output/" + sta_str + "/"

    # -----------------------------------------------------------------------
    # Load resources and setup

    tg, msl = tide_gauge_data(sta)
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

    idx_21 = np.arange(2010, 2101, 1 / 12) + 1 / 24

    A_21 = np.stack(
        [
            np.sin(2 * np.pi * idx_21 / 18.61),
            np.cos(2 * np.pi * idx_21 / 18.61),
        ]
    ).T
    ncyc_21 = pd.Series(A_21 @ ncyc_c[-2:], index=idx_21) - msl

    msl = msl + ncyc_c[0]

    print(ncyc_c)

    # -----------------------------------------------------------------------
    # scenario-based future sea level rise

    pid = sta["psmsl_id"]

    scn_nm = "int"

    fname = "../data/noaa_scenarios/NOAA_SLR_scenarios.pickle"
    with open(fname, "rb") as f:
        scn = pd.read_pickle(f)
    scn["proj"].index += 0.5

    scn = scn["proj"].loc[:, (pid, scn_nm, 50)]

    noaa_prjn = pd.Series(index=idx_21)

    spl = splrep(scn.index, scn.values)
    noaa_prjn.loc[:] = splev(idx_21, spl)
    noaa_prjn += tg.sl.loc["1991":"2009"].mean()

    # -----------------------------------------------------------------------

    return noaa_prjn, ncyc_21, msl


def nodal_cycle_figure(stations, combine_regions=None):

    stations = stations.copy()

    if combine_regions is not None:
        for new_reg in combine_regions:
            for old_reg in combine_regions[new_reg]:
                z = stations["Region"] == old_reg
                stations.loc[z, "Region"] = new_reg

    z = stations["Region"] == "Caribbean"
    stations = pd.concat([stations.loc[~z], stations.loc[z]])

    stations["reg_num"] = None
    for k, reg in enumerate(stations.Region.unique()):
        z = stations["Region"] == reg
        stations.loc[z, "reg_num"] = k
    stations.reg_num += 0.15 * np.random.randn(stations.shape[0])

    stations["ncyc"] = None
    stations["slr"] = None
    stations["ratio"] = None

    stations.apply(lambda x: load_ncyc_vs_slr(x), axis=1)
    stations["Name"] = stations.index

    stations["msize"] = 14
    stations.loc[stations.slr < -0.5, "msize"] = 10

    fig = px.scatter(
        stations,
        y="ratio",
        x="reg_num",
        size="msize",
        color="Region",
        color_discrete_sequence=anlyz.color_palette(),
        hover_data=["Name", "Region", "ncyc", "slr"],
    )
    fig.update_traces(
        marker=dict(size=14, opacity=0.9, line=dict(width=0.5, color="black"))
    )
    fig.for_each_trace(
        lambda trace: trace.update(
            hovertemplate=hover_template(trace.showlegend),
        )
    )
    fig.add_scatter(
        x=[-1, 10],
        y=[0.5, 0.5],
        showlegend=False,
        mode="lines",
        line=dict(width=1, dash="dot"),
    )

    cmmn = {
        "xref": "paper",
        "yref": "paper",
        "xanchor": "left",
        "yanchor": "top",
        "bgcolor": None,
        "showarrow": False,
    }

    right_title_x_adjust = -0.01
    annotations = [
        {
            **{
                "x": 0.5 + right_title_x_adjust,
                "y": 1.1,
                "text": "<b>Nodal cycle vs. sea level rise</b>",
                "font": dict(size=18),
            },
            **cmmn,
        },
        {
            **{
                "x": 0.5 + right_title_x_adjust,
                "y": 1.0275,
                "text": "Local ratios of the nodal cycle range to a decade",
            },
            **cmmn,
        },
        {
            **{
                "x": 0.5 + right_title_x_adjust,
                "y": 0.975,
                "text": "of SLR (2030s, Intermediate scenario)",
            },
            **cmmn,
        },
        {
            **{
                "x": 0.0,
                "y": 1.1,
                "text": "<b>Impact of nodal cycle</b>",
                "font": dict(size=18),
            },
            **cmmn,
        },
        {
            **{
                "x": 0.0,
                "y": 1.0275,
                "text": "NOAA Intermediate SLR scenario",
            },
            **cmmn,
        },
        {
            **{
                "x": 0.0,
                "y": 0.975,
                "text": "St. Petersburg, FL",
            },
            **cmmn,
        },
    ]

    fig.update_layout(
        showlegend=False,
        template="none",
        width=800,
        height=450,
        margin=dict(
            l=70,
            r=15,
            b=80,
            t=40,
            pad=0,
        ),
        xaxis=dict(
            domain=[0.5, 0.95],  # 0.73],
            zeroline=False,
            title_text=None,
            range=[-0.75, 7.5],
            tickvals=np.arange(0, 7),
            #             tickangle=135,
            ticktext=stations.Region.unique(),
        ),
        yaxis=dict(
            domain=[0, 0.9],
            title_text="Ratio",
            range=[-0.07, 2.3],
            #             tickvals=np.arange(0, 14, 2),
            #             zeroline=False,
        ),
        hovermode="closest",
        #         legend_title_text="Region",
        #         legend=dict(x=0.82, y=0.95),
        #         title=dict(
        #             text="Relevance of the 18.6-year nodal cycle",
        #             x=0.13,
        #             y=0.96,
        #             font=dict(size=20),
        #         ),
        annotations=annotations,
        #         shapes=shapes,
    )

    # ---------

    noaa_prjn, ncyc_21, msl = nodal_cycle_example()
    print(msl)

    tot = noaa_prjn + ncyc_21
    ncyc_21 = ncyc_21 + msl  # + 12.3
    print(ncyc_21.loc[2020:2030].idxmax())

    col = anlyz.color_palette()
    fcol = [fill_color(c, 0.15) for c in col if c[0] == "#"]

    cn = [1, 5]
    xmpl_col = [col[n] for n in cn]
    xmpl_fcol = [fcol[n] for n in cn]

    fig.update_layout(
        xaxis2=dict(
            domain=[0, 0.37],  # 0.73],
            zeroline=False,
            #             title_text="Year",
            range=[2020, 2051],
        ),
        yaxis2=dict(
            domain=[0, 0.9],
            title_text="Height above MHHW (cm)",
            range=[-15, 45],
            zeroline=False,
            side="left",
            anchor="x2",
        ),
    )

    fig.add_scatter(
        x=noaa_prjn.index,
        y=noaa_prjn.values,
        line=dict(color=xmpl_fcol[0], width=0),
        xaxis="x2",
        yaxis="y2",
        showlegend=False,
    )

    fig.add_scatter(
        x=noaa_prjn.index,
        y=tot.values,
        mode="none",
        fill="tonexty",
        fillcolor=xmpl_fcol[0],
        hoverinfo="none",
        xaxis="x2",
        yaxis="y2",
        showlegend=False,
    )

    fig.add_scatter(
        x=noaa_prjn.index,
        y=noaa_prjn.values,
        showlegend=False,
        mode="lines",
        line=dict(color=xmpl_col[0], width=5),
        xaxis="x2",
        yaxis="y2",
    )

    fig.add_scatter(
        x=noaa_prjn.index,
        y=tot.values,
        showlegend=False,
        mode="lines",
        line=dict(color=xmpl_col[1], width=5),
        xaxis="x2",
        yaxis="y2",
    )

    fig.add_scatter(
        x=noaa_prjn.index,
        y=ncyc_21.values,
        showlegend=False,
        mode="lines",
        line=dict(color="black", width=3),
        xaxis="x2",
        yaxis="y2",
    )

    #     fig.add_scatter(
    #         x=[2024, 2024],
    #         y=[noaa_prjn.loc[:2024.1].iloc[-1], noaa_prjn.loc[:2034].iloc[-1]],
    #         showlegend=False,
    #         mode="lines",
    #         line=dict(color="black", width=1),
    #         xaxis="x2",
    #         yaxis="y2",
    #     )

    #     fig.add_scatter(
    #         x=[2024, 2034],
    #         y=[noaa_prjn.loc[:2034].iloc[-1], noaa_prjn.loc[:2034].iloc[-1]],
    #         showlegend=False,
    #         mode="lines",
    #         line=dict(color="black", width=1, dash="dot"),
    #         xaxis="x2",
    #         yaxis="y2",
    #     )

    #     fig.add_scatter(
    #         x=[2034, 2034],
    #         y=[noaa_prjn.loc[:2034.1].iloc[-1], noaa_prjn.loc[:2044].iloc[-1]],
    #         showlegend=False,
    #         mode="lines",
    #         line=dict(color="black", width=1),
    #         xaxis="x2",
    #         yaxis="y2",
    #     )

    #     fig.add_scatter(
    #         x=[2034, 2044],
    #         y=[noaa_prjn.loc[:2044].iloc[-1], noaa_prjn.loc[:2044].iloc[-1]],
    #         showlegend=False,
    #         mode="lines",
    #         line=dict(color="black", width=1, dash="dot"),
    #         xaxis="x2",
    #         yaxis="y2",
    #     )

    fig.add_scatter(
        x=[ncyc_21.loc[2020:2030].idxmax(), ncyc_21.loc[2030:2040].idxmin()],
        y=[ncyc_21.loc[2030:2040].min(), ncyc_21.loc[2030:2040].min()],
        showlegend=False,
        mode="lines",
        line=dict(color="black", width=1, dash="dot"),
        xaxis="x2",
        yaxis="y2",
    )

    fig.add_scatter(
        x=[ncyc_21.loc[2020:2030].idxmax(), ncyc_21.loc[2020:2030].idxmax()],
        y=[ncyc_21.loc[2030:2040].min(), ncyc_21.loc[2020:2030].max()],
        showlegend=False,
        mode="lines",
        line=dict(color="black", width=1),
        xaxis="x2",
        yaxis="y2",
    )

    fig.add_scatter(
        x=[2034, 2034],
        y=[noaa_prjn.loc[:2024.1].iloc[-1], noaa_prjn.loc[:2034].iloc[-1]],
        showlegend=False,
        mode="lines",
        line=dict(color="black", width=1),
        xaxis="x2",
        yaxis="y2",
    )

    fig.add_scatter(
        x=[2024, 2034],
        y=[noaa_prjn.loc[:2024.1].iloc[-1], noaa_prjn.loc[:2024.1].iloc[-1]],
        showlegend=False,
        mode="lines",
        line=dict(color="black", width=1, dash="dot"),
        xaxis="x2",
        yaxis="y2",
    )

    fig.add_scatter(
        x=[2044, 2044],
        y=[noaa_prjn.loc[:2034.1].iloc[-1], noaa_prjn.loc[:2044].iloc[-1]],
        showlegend=False,
        mode="lines",
        line=dict(color="black", width=1),
        xaxis="x2",
        yaxis="y2",
    )

    fig.add_scatter(
        x=[2034, 2044],
        y=[noaa_prjn.loc[:2034.1].iloc[-1], noaa_prjn.loc[:2034.1].iloc[-1]],
        showlegend=False,
        mode="lines",
        line=dict(color="black", width=1, dash="dot"),
        xaxis="x2",
        yaxis="y2",
    )

    fig.add_scatter(
        x=[2034, 2034],
        y=[tot.loc[:2024.1].iloc[-1], tot.loc[:2034].iloc[-1]],
        showlegend=False,
        mode="lines",
        line=dict(color="black", width=1),
        xaxis="x2",
        yaxis="y2",
    )

    fig.add_scatter(
        x=[2024, 2034],
        y=[tot.loc[:2024.1].iloc[-1], tot.loc[:2024.1].iloc[-1]],
        showlegend=False,
        mode="lines",
        line=dict(color="black", width=1, dash="dot"),
        xaxis="x2",
        yaxis="y2",
    )

    fig.add_scatter(
        x=[2044, 2044],
        y=[tot.loc[:2034.1].iloc[-1], tot.loc[:2044].iloc[-1]],
        showlegend=False,
        mode="lines",
        line=dict(color="black", width=1),
        xaxis="x2",
        yaxis="y2",
    )

    fig.add_scatter(
        x=[2034, 2044],
        y=[tot.loc[:2034.1].iloc[-1], tot.loc[:2034.1].iloc[-1]],
        showlegend=False,
        mode="lines",
        line=dict(color="black", width=1, dash="dot"),
        xaxis="x2",
        yaxis="y2",
    )

    cmmn = {
        "xref": "x2",
        "yref": "y2",
        "align": "left",
        "xanchor": "left",
        "yanchor": "top",
        "bgcolor": None,
        "showarrow": False,
    }

    print(noaa_prjn.loc[2024:2034].iloc[[0, -1]].diff().iloc[-1])
    print(noaa_prjn.loc[2034:2044].iloc[[0, -1]].diff().iloc[-1])
    annotations = annotations + [
        #         {
        #             **{
        #                 "x": 2025.3,
        #                 "y": -1.5,
        #                 "text": noaa_prjn.loc[2024:2034]
        #                 .iloc[[0, -1]]
        #                 .diff()
        #                 .iloc[-1]
        #                 .round(1)
        # #                 .astype(int)
        #                 .astype(str)
        #                 + " cm",
        #             },
        #             **cmmn,
        #         },
        #         {
        #             **{
        #                 "x": 2035.3,
        #                 "y": -1.5,
        #                 "text": noaa_prjn.loc[2034:2044]
        #                 .iloc[[0, -1]]
        #                 .diff()
        #                 .iloc[-1]
        #                 .round(1)
        # #                 .astype(int)
        #                 .astype(str)
        #                 + " cm",
        #             },
        #             **cmmn,
        #         },
        {
            **{
                "x": 2022.75,
                "y": 7.3,
                "text": np.round(ncyc_21.max() - ncyc_21.min(), 1).astype(str)
                + " cm",
            },
            **cmmn,
        },
        {
            **{
                "x": 2034.5,
                "y": -6.5,
                "text": noaa_prjn.loc[2024:2034]
                .iloc[[0, -1]]
                .diff()
                .iloc[-1]
                .round(1)
                .astype(str)
                + " cm",
            },
            **cmmn,
        },
        {
            **{
                "x": 2044.5,
                "y": 3.5,
                "text": noaa_prjn.loc[2034:2044]
                .iloc[[0, -1]]
                .diff()
                .iloc[-1]
                .round(1)
                .astype(str)
                + " cm",
            },
            **cmmn,
        },
        {
            **{
                "x": 2034.5,
                "y": 25.5,
                "text": tot.loc[2024:2034]
                .iloc[[0, -1]]
                .diff()
                .iloc[-1]
                .round(1)
                .astype(str)
                + " cm",
            },
            **cmmn,
        },
        {
            **{
                "x": 2044.5,
                "y": 35,
                "text": tot.loc[2034:2044]
                .iloc[[0, -1]]
                .diff()
                .iloc[-1]
                .round(1)
                .astype(str)
                + " cm",
            },
            **cmmn,
        },
        {
            **{
                "x": 2025,
                "y": 39,
                "text": "<b>Highest tides +<br>sea level rise</b>",
                "font": dict(size=13, color=col[5]),
            },
            **cmmn,
        },
        {
            **{
                "x": 2021,
                "y": 0.5,
                "text": "<b>Mean sea<br>level rise</b>",
                "font": dict(size=13, color=col[0]),
            },
            **cmmn,
        },
        {
            **{
                "x": 2023,
                "y": 17.3,
                "text": "<b>Highest astronomical tides</b>",
                "font": dict(size=13, color="black"),
            },
            **cmmn,
        },
    ]

    fig.update_layout(
        annotations=annotations,
    )

    config = {"displayModeBar": False, "responsive": False}
    fig.show(config=config)

    return fig, stations, tot
