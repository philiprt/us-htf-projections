import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as psb

import analysis_module as anlyz


def hover_template(is_minor):
    threshold = "Minor" if is_minor else "Moderate"
    threshold_string = "Threshold: NOAA " + threshold + "<br>"
    return (
        "<b>%{customdata[0]}</b><br>"
        + "Region: %{customdata[1]}<br>"
        + threshold_string
        + "YOI: %{x}<br>10-year increase: %{y} HTF days<br>"
        + "10-year multiplier: %{text}"
        + "<extra></extra>"
    )


def yoi_figure(
    yoi_df,
    slr_scenario,
    threshold,
    absolute_increase_cap=100,
    multiplier_cap=10,
    yr_lim=[2018, 2102],
    show_title=True,
    show_legend=True,
    show_xlabel=True,
):

    yoi_df = yoi_df.copy()
    z = (yoi_df["Scenario"] == slr_scenario) & (
        yoi_df["Threshold"] == threshold
    )
    drp_idx = yoi_df.loc[~z, :].index
    yoi_df.drop(drp_idx, inplace=True)

    z = yoi_df["10-year increase"] > absolute_increase_cap
    yoi_df.loc[z, "10-year increase"] = absolute_increase_cap

    z = yoi_df["10-year multiplier"] > multiplier_cap
    yoi_df.loc[z, "10-year multiplier"] = multiplier_cap

    z = (yoi_df["10-year multiplier"] < 0) | yoi_df[
        "10-year multiplier"
    ].isna()
    yoi_df.loc[z, "10-year multiplier"] = 0

    mltplr_txt = [
        "≥10X" if m == 10 else str(m) for m in yoi_df["10-year multiplier"]
    ]

    mtop = 85 if show_title else 55
    mbot = 45 if show_xlabel else 30
    height = 250 + mtop + mbot

    fig = px.scatter(
        yoi_df,
        x="YOI",
        y="10-year increase",
        color="Region",
        size="10-year multiplier",
        hover_data=["Name", "Region"],
        color_discrete_sequence=anlyz.color_palette(),
        text=mltplr_txt,
        height=height,
    )
    fig.for_each_trace(
        lambda trace: trace.update(
            mode="markers",
            hovertemplate=hover_template(trace.showlegend),
            cliponaxis=False,
        )
    )
    fig.update_traces(
        marker=dict(opacity=0.75, line=dict(width=0.5, color="black"))
    )

    scenario_strings = {
        "int_low": "NOAA Intermediate Low",
        "int": "NOAA Intermediate",
        "int_high": "NOAA Intermediate High",
        "kopp": "Kopp et al. (2014)",
    }
    threshold_strings = {
        "minor": "NOAA Minor",
        "moderate": "NOAA Moderate",
    }
    cmmn = {
        "xref": "paper",
        "yref": "paper",
        "xanchor": "center",
        "yanchor": "top",
        "bgcolor": "white",
        "borderpad": 2,
        "showarrow": False,
    }
    annotations = [
        {
            **cmmn,
            **{
                "x": 0.37,
                "y": 1.14,
                "text": "SLR Scenario: "
                + "<b>"
                + scenario_strings[slr_scenario]
                + "</b>"
                + "      "
                + "Threshold: "
                + "<b>"
                + threshold_strings[threshold]
                + "</b>",
                "font": dict(size=14),
                "bgcolor": None,
            },
        },
    ]

    fig.update_layout(
        template="none",
        width=800,
        height=height,
        margin=dict(
            l=70,
            r=25,
            b=mbot,
            t=mtop,
            pad=0,
        ),
        hovermode="closest",
        hoverlabel_align="left",
        xaxis=dict(
            title_text=None,
            range=yr_lim,
            tickmode="array",
            tickvals=[m for m in range(2020, 2101, 10)],
        ),
        yaxis=dict(
            title_text="10-year increase in HTF days",
            range=[-5, 105],
            tickmode="array",
            tickvals=[m for m in range(0, 101, 20)],
            ticktext=["0", "20", "40", "60", "80", "≥100"],
            zeroline=False,
        ),
        showlegend=False,
        annotations=annotations,
    )

    if show_title:
        fig.update_layout(
            height=height,
            title=dict(
                text="10-year increases in HTF following years of inflection",
                x=0.023,
                y=0.96,
                font=dict(size=24),
            ),
        )

    if show_xlabel:
        fig.update_layout(
            xaxis=dict(
                title_text="Year of inflection (YOI)",
                domain=[0, 0.74],
            ),
        )

    # create new axes for legend
    fig.update_layout(
        xaxis2=dict(
            domain=[0.755, 1],
            range=[0, 1],
            fixedrange=True,
            tickvals=[],
            zeroline=False,
        ),
        yaxis2=dict(
            domain=[0, 1],
            range=[0, 1],
            fixedrange=True,
            tickvals=[],
            zeroline=False,
        ),
    )

    regions = yoi_df.Region.unique()
    Nreg = len(regions)
    if show_legend:
        # fig.update_layout(
        #     showlegend=True,
        #     legend_title_text="Region",
        #     legend=dict(x=0.8, y=1),
        # )
        color_legend = pd.DataFrame(
            {
                "region": regions,
                "name": ["  " + r for r in regions],
                "x": [0.13 for k in range(Nreg)],
                "y": [
                    0.87 + k
                    for k in np.arange(0, -(Nreg - 0.5) * 0.075, -0.075)
                ],
            }
        )
        fig.add_trace(
            go.Scatter(
                x=[0.015],
                y=[0.95],
                xaxis="x2",
                yaxis="y2",
                text=["Region"],
                textposition="middle right",
                mode="text",
                showlegend=False,
                hoverinfo="none",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=color_legend["x"],
                y=color_legend["y"],
                xaxis="x2",
                yaxis="y2",
                text=color_legend["name"],
                textposition="middle right",
                mode="markers+text",
                marker=dict(
                    size=11,
                    color=anlyz.color_palette(),
                    opacity=0.75,
                    line=dict(width=0.5, color="black"),
                ),
                showlegend=False,
                hoverinfo="none",
            )
        )
        size_legend = pd.DataFrame(
            {
                "10-year multiplier": [2, 5, 10],
                "name": [" 2X", " 5X", " ≥10X"],
                "x": [0.155 + k for k in [0, 0.32, 0.7]],
                "y": [0.13 for k in range(3)],
            }
        )
        fig.add_trace(
            go.Scatter(
                x=[0.015],
                y=[0.23],
                xaxis="x2",
                yaxis="y2",
                text=["10-year multiplier"],
                textposition="middle right",
                mode="text",
                showlegend=False,
                hoverinfo="none",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=size_legend["x"],
                y=size_legend["y"],
                xaxis="x2",
                yaxis="y2",
                text=size_legend["name"],
                textposition="bottom center",
                mode="markers+text",
                marker=dict(
                    size=size_legend["10-year multiplier"],
                    sizemode="area",
                    sizeref=2.0
                    * max(size_legend["10-year multiplier"])
                    / (29.0 ** 2),
                    sizemin=0,
                    color="rgb(0.3, 0.3, 0.3)",
                ),
                showlegend=False,
                hoverinfo="none",
            )
        )

    #     thrsh_leg_dy = 0.061
    #     threshold_legend = pd.DataFrame(
    #         {
    #             "name": ["   NOAA Moderate", "   NOAA Minor"],
    #             "symbol": ["diamond", "circle"],
    #             "x": [0.15 for k in range(2)],
    #             "y": [0.3 + thrsh_leg_dy * k for k in [0, 1]],
    #         }
    #     )
    #     fig.add_trace(
    #         go.Scatter(
    #             x=[0.015],
    #             y=[threshold_legend["y"].iloc[-1] + thrsh_leg_dy],
    #             xaxis="x2",
    #             yaxis="y2",
    #             text=["HTF Threshold"],
    #             textposition="middle right",
    #             mode="text",
    #             showlegend=False,
    #             hoverinfo="none",
    #         )
    #     )
    #     fig.add_trace(
    #         go.Scatter(
    #             x=threshold_legend["x"],
    #             y=threshold_legend["y"],
    #             xaxis="x2",
    #             yaxis="y2",
    #             text=threshold_legend["name"],
    #             textposition="middle right",
    #             mode="markers+text",
    #             marker=dict(
    #                 size=9,
    #                 symbol=threshold_legend["symbol"],
    #                 color="rgba(0.3, 0.3, 0.3, 0.75)"  # "white",
    #                 #                 line=dict(color="black", width=1)
    #             ),
    #             showlegend=False,
    #             hoverinfo="none",
    #         )
    #     )

    config = {"displayModeBar": False, "responsive": False}
    fig.show(config=config)

    return fig


def yoi_stats(yoi, slr_scenario, threshold, region=None):

    yoi = yoi.copy()
    if region is not None:
        yoi = yoi.loc[yoi["Region"].isin(region)]

    z = (yoi["Scenario"] == slr_scenario) & (yoi["Threshold"] == threshold)
    drp_idx = yoi.loc[~z, :].index
    yoi.drop(drp_idx, inplace=True)

    Ntot = yoi.shape[0]

    before_2040 = yoi["YOI"] <= 2040
    before_2060 = yoi["YOI"] <= 2060

    bgi = 0
    bgm = 7

    big_increase = yoi["10-year increase"] >= bgi
    big_multiplier = yoi["10-year multiplier"] >= bgm
    big_both = big_increase & big_multiplier

    bgi2 = 50
    bgm2 = 1
    big_increase2 = yoi["10-year increase"] >= bgi2
    big_multiplier2 = yoi["10-year multiplier"] >= bgm2
    big_both2 = big_increase2 & big_multiplier2

    bgi3 = 75
    bgm3 = 2
    big_increase3 = yoi["10-year increase"] >= bgi3
    big_multiplier3 = yoi["10-year multiplier"] >= bgm3
    big_both3 = big_increase3 & big_multiplier3

    scenario_strings = {
        "int_low": "Intermediate Low",
        "int": "Intermediate",
        "int_high": "Intermediate High",
        "kopp": "Kopp et al. (2014)",
    }
    threshold_strings = {
        "minor": "Minor",
        "moderate": "Moderate",
    }

    rows = [
        ("Year of inflection (YOI)", "before " + str(2040)),
        ("Year of inflection (YOI)", "before " + str(2060)),
        ("10-yr increase following YOI", ">= " + str(bgi)),
        ("10-yr increase following YOI", ">= " + str(bgi2)),
        ("10-yr increase following YOI", ">= " + str(bgi3)),
        ("10-yr multiplier following YOI", ">= " + str(bgm)),
        ("10-yr multiplier following YOI", ">= " + str(bgm2)),
        ("10-yr multiplier following YOI", ">= " + str(bgm3)),
        (
            "Increase ≥ " + str(bgi) + " & mulitplier ≥ " + str(bgm),
            "YOI before 2040",
        ),
        (
            "Increase ≥ " + str(bgi) + " & mulitplier ≥ " + str(bgm),
            "YOI before 2060",
        ),
        (
            "Increase ≥ " + str(bgi2) + " & mulitplier ≥ " + str(bgm2),
            "YOI before 2040",
        ),
        (
            "Increase ≥ " + str(bgi2) + " & mulitplier ≥ " + str(bgm2),
            "YOI before 2060",
        ),
    ]
    rmidx = pd.MultiIndex.from_tuples(
        rows  # , names=["Total stations: " + str(Ntot), ""]
    )

    cols = [(scenario_strings[slr_scenario], threshold_strings[threshold])]
    cmidx = pd.MultiIndex.from_tuples(
        cols, names=["NOAA SLR Scenario →", "NOAA HTF Threshold →"]
    )

    yoi_stats = pd.DataFrame(index=rmidx, columns=cmidx)

    N = before_2040.sum()
    yoi_stats.loc[
        ("Year of inflection (YOI)", "before " + str(2040))
    ] = "{:0.1f}% ({:d})".format(100 * N / Ntot, N)
    N = before_2060.sum()
    yoi_stats.loc[
        ("Year of inflection (YOI)", "before " + str(2060))
    ] = "{:0.1f}% ({:d})".format(100 * N / Ntot, N)
    N = big_increase.sum()
    yoi_stats.loc[
        ("10-yr increase following YOI", ">= " + str(bgi))
    ] = "{:0.1f}% ({:d})".format(100 * N / Ntot, N)
    N = big_increase2.sum()
    yoi_stats.loc[
        ("10-yr increase following YOI", ">= " + str(bgi2))
    ] = "{:0.1f}% ({:d})".format(100 * N / Ntot, N)
    N = big_increase3.sum()
    yoi_stats.loc[
        ("10-yr increase following YOI", ">= " + str(bgi3))
    ] = "{:0.1f}% ({:d})".format(100 * N / Ntot, N)
    N = big_multiplier.sum()
    yoi_stats.loc[
        ("10-yr multiplier following YOI", ">= " + str(bgm))
    ] = "{:0.1f}% ({:d})".format(100 * N / Ntot, N)
    N = big_multiplier2.sum()
    yoi_stats.loc[
        ("10-yr multiplier following YOI", ">= " + str(bgm2))
    ] = "{:0.1f}% ({:d})".format(100 * N / Ntot, N)
    N = big_multiplier3.sum()
    yoi_stats.loc[
        ("10-yr multiplier following YOI", ">= " + str(bgm3))
    ] = "{:0.1f}% ({:d})".format(100 * N / Ntot, N)
    N = (big_both & before_2040).sum()
    yoi_stats.loc[
        (
            "Increase ≥ " + str(bgi) + " & mulitplier ≥ " + str(bgm),
            "YOI before 2040",
        )
    ] = "{:0.1f}% ({:d})".format(100 * N / Ntot, N)
    N = (big_both & before_2060).sum()
    yoi_stats.loc[
        (
            "Increase ≥ " + str(bgi) + " & mulitplier ≥ " + str(bgm),
            "YOI before 2060",
        )
    ] = "{:0.1f}% ({:d})".format(100 * N / Ntot, N)
    N = (big_both2 & before_2040).sum()
    yoi_stats.loc[
        (
            "Increase ≥ " + str(bgi2) + " & mulitplier ≥ " + str(bgm2),
            "YOI before 2040",
        )
    ] = "{:0.1f}% ({:d})".format(100 * N / Ntot, N)
    N = (big_both2 & before_2060).sum()
    yoi_stats.loc[
        (
            "Increase ≥ " + str(bgi2) + " & mulitplier ≥ " + str(bgm2),
            "YOI before 2060",
        )
    ] = "{:0.1f}% ({:d})".format(100 * N / Ntot, N)

    yoi_stats.iloc[:, 0] = yoi_stats.iloc[:, 0].str.pad(10)

    stn_of_interest = yoi.loc[big_both & before_2040, :]

    return yoi_stats, stn_of_interest
