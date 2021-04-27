import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import analysis_module as anlyz


def hover_template(is_minor):
    threshold = "Minor" if is_minor else "Moderate"
    threshold_string = "Threshold: NOAA " + threshold + "<br>"
    return (
        "<b>%{customdata[0]}</b><br>"
        + "Region: %{customdata[1]}<br>"
        + threshold_string
        + "Year: %{x}"
        + "<extra></extra>"
    )


def first_month_figure(
    fmo_df,
    slr_scenario,
    year_range=[2015, 2080],
    threshold=None,
    threshold_by_region=None,
    min_mod_switch=None,
):

    fmo_df = fmo_df.loc[fmo_df.Scenario == slr_scenario]

    if threshold_by_region is not None:
        z_thrsh = np.zeros(fmo_df.shape[0]).astype(bool)
        for reg in fmo_df.Region.unique():
            kp_thrsh = (
                threshold_by_region[reg]
                if reg in threshold_by_region
                else "minor"
            )
            z_thrsh[
                (fmo_df.Region == reg) & (fmo_df.Threshold == kp_thrsh)
            ] = True
    elif min_mod_switch is not None:
        z_thrsh = np.zeros(fmo_df.shape[0]).astype(bool)
        for nm in fmo_df.Name.unique():
            z_chck = (
                (fmo_df.Name == nm)
                & (fmo_df.Quantity == min_mod_switch[0])
                & (fmo_df.Threshold == "minor")
            )
            stn_thrsh = (
                "minor"
                if fmo_df.Year.loc[z_chck].values[0] >= min_mod_switch[1]
                else "moderate"
            )
            z_thrsh[
                (fmo_df.Name == nm) & (fmo_df.Threshold == stn_thrsh)
            ] = True
    elif threshold is not None:
        z_thrsh = fmo_df.Threshold == threshold
    else:
        raise (
            "One of 'threshold_by_region', 'min_mod_switch', or 'threshold' "
            + "needs to be set."
        )
    fmo_df = fmo_df.loc[z_thrsh, :]

    quantities = fmo_df["Quantity"].unique()
    fmo_df["Q_val"] = None
    for n, q in enumerate(quantities):
        z = fmo_df["Quantity"] == q
        fmo_df.loc[z, "Q_val"] = n

    regions = fmo_df["Region"].unique()
    offsets = 0.1 * np.arange(regions.size)
    offsets -= offsets.mean()
    for n, r in enumerate(regions):
        z = fmo_df["Region"] == r
        fmo_df.loc[z, "Q_val"] += offsets[n]
    fmo_df.loc[:, "Q_val"] += (
        0.08 * np.random.random_sample((fmo_df.shape[0],)) - 0.05
    )

    fig = px.scatter(
        fmo_df,
        y="Q_val",
        x="Year",
        color="Region",
        color_discrete_sequence=anlyz.color_palette(),
        symbol="Threshold",
        hover_data=["Name", "Region", "Quantity"],
    )
    # )
    fig.update_traces(
        marker=dict(size=10, opacity=0.9, line=dict(width=0.5, color="black"))
    )
    fig.update_traces(
        patch=dict(showlegend=False),
        selector=dict(marker_symbol="diamond"),
        overwrite=True,
    )
    fig.for_each_trace(
        lambda trace: trace.update(
            legendgroup=trace.name[:-7]
            if trace.showlegend
            else trace.name[:-10],
            name=trace.name[:-7] if trace.showlegend else trace.name[:-10],
            hovertemplate=hover_template(trace.showlegend),
        )
    )

    shapes = [pentad_shading(yrw) for yrw in [-0.5, 1.5]]

    scenario_strings = {
        "int_low": "NOAA Intermediate Low",
        "int": "NOAA Intermediate",
        "int_high": "NOAA Intermediate High",
        "kopp": "Kopp et al. (2014)",
    }
    cmmn = {
        "xref": "paper",
        "yref": "paper",
        "xanchor": "left",
        "yanchor": "top",
        "bgcolor": "white",
        "borderpad": 2,
        "showarrow": False,
    }
    annotations = [
        {
            **cmmn,
            **{
                "x": -0.286,
                "y": 1.08,
                "text": scenario_strings[slr_scenario] + " SLR Scenario",
                "font": dict(size=14),
                "bgcolor": None,
            },
        },
    ]

    prob_tick_end = " that the first<br>such month has occurred"
    fig.update_layout(
        template="none",
        width=800,
        height=500,
        margin=dict(
            l=195,
            r=15,
            b=35,
            t=75,
            pad=0,
        ),
        xaxis=dict(
            domain=[0, 0.68],
            title_text=None,
            range=[year_range[0], year_range[1] + 0.22],
            tickmode="array",
            tickvals=[2020 + 10 * v for v in range(10)],
        ),
        yaxis=dict(
            title_text=None,
            range=[-0.5, 3.5],
            showgrid=False,
            zeroline=False,
            tickmode="array",
            tickvals=[v for v in range(4)],
            ticktext=[
                "<b>25% chance</b>" + prob_tick_end,
                "<b>50% chance</b>" + prob_tick_end,
                "<b>Expected</b> during a <br>single month each year",
                "<b>Expected</b> on average<br>over the whole year",
            ],
        ),
        hovermode="closest",
        legend_title_text="Region",
        legend=dict(x=0.7, y=1),
        title=dict(
            text="When will U.S. locations experience HTF on a majority of days?",
            x=0.035,
            y=0.96,
            font=dict(size=22),
        ),
        annotations=annotations,
        shapes=shapes,
    )

    if threshold is None:

        # create new axes for threshold legend
        fig.update_layout(
            xaxis2=dict(
                domain=[0.7, 1],
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

        thrsh_leg_dy = 0.047
        threshold_legend = pd.DataFrame(
            {
                "name": ["   NOAA Moderate", "   NOAA Minor"],
                "symbol": ["diamond", "circle"],
                "x": [0.115 for k in range(2)],
                "y": [0.43 + thrsh_leg_dy * k for k in [0, 1]],
            }
        )
        fig.add_trace(
            go.Scatter(
                x=[0.015],
                y=[threshold_legend["y"].iloc[-1] + thrsh_leg_dy],
                xaxis="x2",
                yaxis="y2",
                text=["HTF Threshold"],
                textposition="middle right",
                mode="text",
                showlegend=False,
                hoverinfo="none",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=threshold_legend["x"],
                y=threshold_legend["y"],
                xaxis="x2",
                yaxis="y2",
                text=threshold_legend["name"],
                textposition="middle right",
                mode="markers+text",
                marker=dict(
                    size=9,
                    symbol=threshold_legend["symbol"],
                    color="rgba(0.3, 0.3, 0.3, 0.75)"  # "white",
                    #                 line=dict(color="black", width=1)
                ),
                showlegend=False,
                hoverinfo="none",
            )
        )

    config = {"displayModeBar": False, "responsive": False}
    fig.show(config=config)

    return fig


def pentad_shading(yrw):
    return dict(
        type="rect",
        x0=1900,
        x1=2200,
        y0=yrw,
        y1=yrw + 1,
        layer="below",
        fillcolor="#f5f5f5",
        line=dict(color="black", width=0),
    )


def first_month_stats(
    fmo_df, scenario="int", threshold="minor", year_diff=[15, 25]
):

    fmo_pvt = fmo_df.loc[fmo_df["Threshold"] == threshold].pivot_table(
        values="Year",
        index=["Region", "Name"],
        columns=["Scenario", "Quantity"],
    )

    yrs = [2040, 2050]
    print(
        "{:0.1f}% ({:0.1f}%) of stations have a 50% chance of experiencing first month prior to {:d} ({:d})".format(
            100 * (fmo_pvt["int", "P50"] < yrs[0]).sum() / fmo_pvt.shape[0],
            100 * (fmo_pvt["int", "P50"] < yrs[1]).sum() / fmo_pvt.shape[0],
            yrs[0],
            yrs[1],
        )
    )

    fmo_pvt["int", "EA-P25"] = (
        fmo_pvt[scenario, "Expected annually"] - fmo_pvt["int", "P25"]
    )
    fmo_pvt["int", "EA-P50"] = (
        fmo_pvt[scenario, "Expected annually"] - fmo_pvt["int", "P50"]
    )
    #     fmo_pvt["int_low", "EA-P25"] = (
    #         fmo_pvt["int_low", "Expected annually"] - fmo_pvt["int_low", "P25"]
    #     )
    #     fmo_pvt["int_low", "EA-P50"] = (
    #         fmo_pvt["int_low", "Expected annually"] - fmo_pvt["int_low", "P50"]
    #     )

    region = fmo_pvt.index.get_level_values("Region").unique().to_list()
    region = [
        r + " ({:02.0f})".format(fmo_pvt.loc[r].shape[0]) for r in region
    ]
    region = ["All ({:02.0f})".format(fmo_pvt.shape[0])] + region

    row_midx = pd.MultiIndex.from_product(
        [region, year_diff], names=["Region", "≥"]
    )
    col_midx = pd.MultiIndex.from_product([["EA-P25", "EA-P50"], ["%", "N"]])
    fmo_table = pd.DataFrame(index=row_midx, columns=col_midx)

    for r in region:
        reg = r[:-5]
        for d in year_diff:
            if reg == "All":
                df = fmo_pvt[scenario][["EA-P25", "EA-P50"]]
            else:
                df = fmo_pvt.loc[reg][scenario][["EA-P25", "EA-P50"]]
            N = (df >= d).sum(axis=0)
            P = 100 * N / df.shape[0]
            fmo_table.loc[(r, d), (slice(None), "N")] = N.values
            fmo_table.loc[(r, d), (slice(None), "%")] = P.values.round(1)

    return fmo_table
