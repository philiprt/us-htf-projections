import numpy as np
import plotly.graph_objects as go
import plotly.subplots as psb

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


def climatology_projections(analysis, subset, quantity, yoi_srch_rng):

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
        loc[quantity].loc[loc["steps"][1:], :] for loc in prjn_subset
    ]

    col = anlyz.color_palette()
    col = [col[n] for n in [1, 5]]

    nrows = len(prjn_subset)
    vspace = 0.05
    sph = (1 - (nrows - 1) * vspace) / nrows
    fig = psb.make_subplots(rows=nrows, cols=1, vertical_spacing=vspace)

    mo_labels = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]

    for n, prjn in enumerate(prjn_subset):
        leg = True  # if r == 0 else False
        r = n + 1
        traces = projection_traces(prjn, stn_meta[n], col, leg)
        annotations, shapes = projection_annotations(
            prjn, stn_meta[n], r, vspace, sph, col
        )
        for trc in traces:
            fig.add_trace(trc, row=r, col=1)
        for ann in annotations:
            fig.add_annotation(ann)
        for shp in shapes:
            fig.add_shape(shp)

        fig.update_yaxes(title_text="days/month", row=r, col=1)  # , side="right")
        fig.update_xaxes(
            range=[0.5, 12.5],
            tickmode="array",
            tickvals=[m for m in range(1, 13)],
            ticktext=mo_labels,
        )

    fig.update_layout(
        width=550,
        height=800,
        template="none",
        margin=dict(
            l=55,
            r=180,
            b=25,
            t=85,
            pad=0,
        ),
        xaxis=dict(
            layer="below traces",
        ),
        yaxis=dict(
            layer="below traces",
        ),
        hovermode="x",
        legend=dict(
            orientation="h",
            x=0.675,
            y=1.17,
            itemclick=False,
            itemdoubleclick=False,
        ),
        title=dict(
            text="Projected HTF Annual Cycles",
            x=0.07,
            y=0.975,
            font=dict(size=24),
        ),
    )

    config = {"displayModeBar": False, "responsive": False}
    fig.show(config=config)

    return fig


def projection_traces(prjn, meta, col, leg):

    years = prjn.index.get_level_values("year").unique().values

    fcol = [fill_color(c, 0.5) for c in col if c[0] == "#"]

    prjn_traces = []
    for n, y in enumerate(years):

        prjn_traces.extend(
            [
                {
                    "x": prjn.loc[y].index.values,
                    "y": prjn.loc[y][10].values,
                    "type": "scatter",
                    "mode": "lines",
                    "fill": "none",
                    "showlegend": False,
                    "line": {"color": fcol[n], "width": 0},
                    "hoverinfo": "none",
                },
                {
                    "x": prjn.loc[y].index.values,
                    "y": prjn.loc[y][90].values,
                    "type": "scatter",
                    "mode": "lines",
                    "fill": "tonexty",
                    "fillcolor": fcol[n],
                    "showlegend": False,
                    "mode": "none",
                    "hoverinfo": "none",
                },
                {
                    "x": prjn.loc[y].index.values,
                    "y": prjn.loc[y][50].values,
                    "type": "scatter",
                    "name": str(y) if leg else None,
                    "showlegend": False,
                    "line": {"color": col[n], "width": 2},
                    "hoverinfo": "y",
                },
            ]
        )

    return prjn_traces


def projection_annotations(prjn, meta, r, vspace, sph, col):

    x = 1.0

    dy = 0.032
    y = 1 - (r - 1) * (vspace + sph)

    cmmn = {
        "x": x,
        "xref": "paper",
        "yref": "paper",
        "xanchor": "left",
        "yanchor": "top",
        "borderpad": 5,
        "showarrow": False,
    }
    annotations = [
        {
            **{
                "y": y - dy,
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
    for n, yr in enumerate(meta["steps"][1:]):
        txt = str(yr)
        if n == 0:
            txt += " (YOI)"
        annotations.append(
            {
                **cmmn,
                **{
                    "x": x + 0.125,
                    "y": y - (n + 2) * dy,
                    "text": txt,
                },
            }
        )

    if r == 1:
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
                    "x": -0.07,
                    "y": 1.065,
                    "text": scenario_strings[meta["scenario"]]
                    + " SLR Scenario",
                    "font": dict(size=14),
                    "bgcolor": None,
                },
            },
        )

    cmmn = {
        "xref": "paper",
        "yref": "paper",
        "type": "line",
        "x0": 1.02,
        "x1": 1.12,
    }
    shapes = []
    for n, yr in enumerate(meta["steps"][1:]):
        yshp = y - (n + 2) * dy - 0.02
        shapes.append(
            {
                **cmmn,
                **{
                    "y0": yshp,
                    "y1": yshp,
                    "line": dict(color=col[n], width=3),
                },
            }
        )

    return annotations, shapes
