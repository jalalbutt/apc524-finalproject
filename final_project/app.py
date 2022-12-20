"""
app.py
------
The dash app of the project
"""

from dash import Dash, dcc, html, Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# import model to generate project data
import model
import xarray as xr

external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]

app = Dash(__name__, external_stylesheets=external_stylesheets)

gws = "graph-with-slider"

logo_path = "assets/logo.jpeg"
princeton_path = "assets/princeton.png"

timesteps = pd.Series([0, 1])

app.layout = html.Div(
    [
        html.Img(
            src=princeton_path,
            height=150,
            style={
                "float": "right",
                "margin-left": "auto",
                "margin-right": 115,
            },
        ),
        html.Img(src=logo_path, height=150),
        dcc.Graph(id=gws),
        html.Div(
            [
                html.H3("Set temporal state", style={"textAlign": "center"}),
                dcc.Slider(
                    timesteps.min(),
                    timesteps.max(),
                    step=None,
                    value=timesteps.min(),
                    marks={0: "Init", 1: "Final"},
                    id="time-slider",
                ),
            ],
            style={
                "padding-left": "33%",
                "width": 500,
            },
        ),
        html.Div(
            [
                html.Div(
                    [
                        html.H3("Set location of perturbation center"),
                        html.Div(
                            [
                                "Latitude",
                                dcc.Input(
                                    id="input-lat",
                                    type="text",
                                    value="40.34578",
                                    style={"float": "center"},
                                ),
                            ],
                            style={"textAlign": "center"},
                        ),
                        html.Div(
                            [
                                "Longitude",
                                dcc.Input(
                                    id="input-lon",
                                    type="text",
                                    value="-74.65256",
                                    style={"float": "center"},
                                ),
                            ],
                            style={"textAlign": "center"},
                        ),
                    ],
                    style={"textAlign": "center"},
                ),
                html.Div(
                    [
                        html.H3("Set radius of perturbation"),
                        html.Div(
                            [
                                "Radius in km",
                                dcc.Input(
                                    id="input-blastradius",
                                    type="text",
                                    value="35",
                                    style={"float": "center"},
                                ),
                            ],
                            style={"textAlign": "center"},
                        ),
                    ],
                    style={"textAlign": "center"},
                ),
                html.Div(
                    [
                        html.Button(
                            id="submit-button-state",
                            n_clicks=0,
                            children="Submit",
                        ),
                        html.H3("Please note, calculation takes about 10-20s"),
                    ],
                    style={"textAlign": "center"},
                ),
                html.Div(id="output-state"),
            ]
        ),
    ]
)


@app.callback(
    Output(gws, "figure"),
    Output("output-state", "children"),
    Input("submit-button-state", "n_clicks"),
    State("time-slider", "value"),
    State("input-lat", "value"),
    State("input-lon", "value"),
    State("input-blastradius", "value"),
)
def update_figure(submit, selected_timestep, lat, lon, radius):
    """
    Updates figure based on user input.
    """

    (
        array_result,
        opt_result,
        A_p,
        A_g,
        nodes_p,
        nodes_g,
    ) = model.run_model(float(lat), float(lon), float(radius))

    # threshold array
    array_result = xr.where(
        array_result < array_result.max() * 0.5, 0, array_result
    )

    perturbation = (
        array_result.sel(time=selected_timestep)
        .to_dataframe(name="value")
        .reset_index()
    )

    nodemaster_p = pd.concat(
        (opt_result[selected_timestep]["nse_power"], nodes_p), axis=1
    )
    nodemaster_p["relative_service"] = (
        1 - nodemaster_p["non_served_energy"] / nodemaster_p["load"]
    )
    nodemaster_p["index"] = nodemaster_p.index

    fig = px.density_mapbox(
        perturbation,
        lat="lat",
        lon="lon",
        z="value",
        radius=3,
        zoom=0,
        color_continuous_scale=px.colors.diverging.RdGy,
        range_color=(0.5, 1.5),
        mapbox_style="stamen-terrain",
    )
    fig.update_layout(coloraxis_colorbar_title_text="Perturb.")

    fig2 = px.scatter_mapbox(
        nodemaster_p,
        lat="lat",
        lon="lon",
        hover_name="index",
        hover_data=["non_served_energy", "load"],
        color="relative_service",
        color_continuous_scale="RdBu",
        range_color=(0, 1.01),
        zoom=3,
        height=400,
    )
    fig2.update_layout(coloraxis_colorbar_title_text="Load")

    fig.add_trace(fig2.data[0])
    fig.layout.coloraxis2 = fig2.layout.coloraxis

    fig["data"][1]["marker"] = {
        "color": nodemaster_p["relative_service"],
        "coloraxis": "coloraxis2",
        "opacity": 1,
        "sizemode": "area",
        "sizeref": 2 * nodemaster_p.load.max() / (40**2),
        "autocolorscale": False,
        "size": nodemaster_p["load"],
    }

    fig.layout.coloraxis2.colorbar.x = -0.07
    fig.layout.coloraxis.colorbar.x = -0.13

    for i in range(A_p.T.shape[0]):
        for j in range(A_p.T.shape[1]):
            if A_p.T[i][j] == 1:
                start_index = j
                start_node = nodemaster_p.iloc[start_index]
            if A_p.T[i][j] == -1:
                end_index = j
                end_node = nodemaster_p.iloc[end_index]
        edge = pd.concat((start_node, end_node), axis=1).T

        fig.add_trace(
            go.Scattermapbox(
                lat=edge["lat"],
                lon=edge["lon"],
                mode="lines",
                line={"color": "#000000"},
                hoverinfo="skip",
            )
        )

    fig.update_layout(
        mapbox_style="carto-positron",
        mapbox_zoom=3,
        mapbox_center={"lat": 37.0902, "lon": -95.7129},
    )
    fig.update_layout(
        margin={"r": 0, "t": 0, "l": 0, "b": 0}, showlegend=False
    )

    return (
        fig,
        """
        Lat: "{}",
        Lon: "{}",
        Radius: "{}",
        Submit: "{}"
        """.format(
            lat, lon, radius, submit
        ),
    )


if __name__ == "__main__":
    app.run_server(debug=True)
