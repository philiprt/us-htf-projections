import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as psb

import analysis_module as anlyz


def pentad_figures(analysis, subset, yr_lims):

    colors = anlyz.color_palette()
    colors = [colors[n] for n in [1, 6, 5]]

    prjn_subset = [
        analysis[loc["name"]][loc["scenario"]][loc["threshold"]]
        for loc in subset
    ]
    prjn_subset = prjn_subset[:4]

    stn_meta = [
        {
            **loc["station"],
            **loc["experiment"],
            **{"steps": loc["steps"]},
        }
        for loc in prjn_subset
    ]

    pent_avg = [
        loc["xdys_pent_mn_ptl"].loc[yr_lims[0] : yr_lims[1]] / 12
        for loc in prjn_subset
    ]

    pent_mxssn = [
        loc["xdys_pent_mxssn_ptl"].loc[yr_lims[0] : yr_lims[1]] / 3
        for loc in prjn_subset
    ]

    pent_mxmo = [
        loc["xdys_pent_mxmo_ptl"].loc[yr_lims[0] : yr_lims[1]]
        for loc in prjn_subset
    ]

    vspace = 0.12
    hspace = 0.12
    fig = psb.make_subplots(
        rows=2,
        cols=2,
        vertical_spacing=vspace,
        horizontal_spacing=hspace,
    )

    yaxis_format = dict(
        layer="below traces", range=[-2, 32], zeroline=False, side="right"
    )

    annotations = []
    shapes = []
    for n, (avg, mxsn, mxmo) in enumerate(
        zip(pent_avg, pent_mxssn, pent_mxmo)
    ):

        # row and column index
        r = int(n / 2) + 1
        c = (n % 2) + 1

        # shade alternating 5 year periods
        shapes.extend(
            [pentad_shading(y5, n) for y5 in avg.index[avg.index % 10 == 5]]
        )

        # add data traces
        showlegend = True if n == 0 else False
        trace_inputs = [
            dict(
                q=avg, dx=1.5, c=colors[0], leg=showlegend, nm="5-year average"
            ),
            dict(
                q=mxsn,
                dx=2.5,
                c=colors[1],
                leg=showlegend,
                nm="5-year peak season (3-month period)",
            ),
            dict(
                q=mxmo,
                dx=3.5,
                c=colors[2],
                leg=showlegend,
                nm="5-year peak month",
            ),
        ]
        traces = [pentad_trace(trc) for trc in trace_inputs]
        for trc in traces:
            fig.add_trace(trc, row=r, col=c)

        # create annotations for station and threshold in each subplot
        annotations.extend(pentad_annotations(stn_meta[n], r, c, vspace))

        fig.update_yaxes(
            patch=yaxis_format, title_text="days/month", row=r, col=c
        )
        fig.update_xaxes(
            range=[yr_lims[0] - 1, yr_lims[1] + 6],
            tickvals=[2032.5 + 5 * v for v in range(5)],
            ticktext=[
                #                 str(2030 + 5 * v) + "–" + str((5 * v + 4) % 10)
                "'" + str(30 + 5 * v) + "–'" + str(30 + 5 * v + 4)
                for v in range(5)
            ],
            showgrid=False,
            row=r,
            col=c,
        )

    fig.update_layout(
        width=800,
        height=525,
        template="none",
        shapes=shapes,
        margin=dict(
            l=25,
            r=70,
            b=30,
            t=70,
            pad=0,
        ),
        hovermode="closest",
        # hoverlabel_align="left",
        legend=dict(
            orientation="v",
            x=0.69,
            y=1.20,
            itemclick=False,
            itemdoubleclick=False,
        ),
        title=dict(
            text="Clustering of High-Tide-Flooding Days",
            x=0.035,
            y=0.96,
            font=dict(size=24),
        ),
        annotations=annotations,
    )

    config = {"displayModeBar": False, "responsive": False}
    fig.show(config=config)

    return fig


def pentad_trace(trace_input):

    q = trace_input["q"]

    #     return dict(
    #         type="scatter",
    #         x=q.index + trace_input["dx"],
    #         y=q[50].values,
    #         error_y=dict(
    #             array=(q[90] - q[50]).values,
    #             arrayminus=(q[50] - q[10]).values,
    #             thickness=10.90,
    #             width=0,
    #             color="rgba" + trace_input["c"][3:-1] + ", 0.9)",
    #         ),
    #         mode="markers",
    #         marker=dict(
    #             symbol="line-ew",
    #             color=trace_input["c"],
    #             size=7.5,
    #             line=dict(color="black", width=3),
    #         ),
    #         showlegend=trace_input["leg"],
    #     )

    return dict(
        type="scatter",
        name=trace_input["nm"],
        x=q.index + trace_input["dx"],
        y=(q[50].values * 10).round() / 10,
        error_y=dict(
            array=(q[90] - q[50]).values,
            arrayminus=(q[50] - q[10]).values,
            thickness=2,
            width=0,
            #             color="rgba" + trace_input["c"][3:-1] + ", 0.8)",
            color=trace_input["c"],
        ),
        mode="markers",
        marker=dict(
            # symbol="line-ew",
            color=trace_input["c"],
            opacity=1.0,
            size=11,
            line=dict(color="black", width=0.5),
        ),
        showlegend=trace_input["leg"],
        hovertemplate="%{y:.0f}",
    )


def pentad_shading(y5, n):
    return dict(
        type="rect",
        xref="x" + str(n + 1),
        yref="y" + str(n + 1),
        x0=y5,
        x1=y5 + 5,
        y0=-2,
        y1=32,
        layer="below",
        fillcolor="#f5f5f5",
        line=dict(color="black", width=0),
    )


def pentad_annotations(meta, r, c, vspace):

    x = 2029

    dy = 0.025
    y = 1 + dy if r == 1 else (1 - vspace) / 2 + dy

    Lnm = len(meta["name"])
    xspc = " " * (20 - Lnm)

    cmmn = {
        "x": x,
        "xref": "x" + str(c),
        "yref": "paper",
        "xanchor": "left",
        "yanchor": "top",
        "bgcolor": "white",
        "borderpad": 3,
        "showarrow": False,
    }
    annotations = [
        {
            **{
                "y": y - 0.055,
                "text": "Threshold: NOAA "
                + meta["threshold"].capitalize()
                + " ",
            },
            **cmmn,
        },
        {
            **{
                "y": y,
                "text": "<b>" + meta["name"] + "<b>" + xspc,  # ,
                "font": dict(size=16),
            },
            **cmmn,
        },
    ]
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
                    "y": y + 0.098,
                    "text": scenario_strings[meta["scenario"]]
                    + " SLR Scenario",
                    "font": dict(size=14),
                    "bgcolor": None,
                },
            },
        )

    return annotations
