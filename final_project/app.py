"""
app.py
------
The dash app of the project
"""

from dash import Dash, dcc, html, Input, Output
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# import model to generate project data
import model

(
    array_result,
    opt_result,
    A_p,
    A_g,
    nodes_p,
    nodes_g,
) = model.test_network_model()

logo_path = "assets/logo.jpeg"
princeton_path = "assets/princeton.png"

timesteps = pd.Series([0, 1])

# dummy data
us_cities = pd.read_csv(
    "https://raw.githubusercontent.com/plotly/"
    "datasets/master/us-cities-top-1k.csv"
)

app = Dash(__name__)

gws = "graph-with-slider"

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
        dcc.Slider(
            timesteps.min(),
            timesteps.max(),
            step=None,
            value=timesteps.min(),
            marks={str(year): str(year) for year in timesteps.unique()},
            id="time-slider",
        ),
    ]
)


@app.callback(Output(gws, "figure"), Input("time-slider", "value"))
def update_figure(selected_timestep):
    """
    Updates figure based on user input.
    """
    pertubation = (
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

    flow_p = opt_result[selected_timestep]["flow_power"]

    fig = px.density_mapbox(
        pertubation,
        lat="lat",
        lon="lon",
        z="value",
        radius=10,
        zoom=0,
        color_continuous_scale=px.colors.diverging.RdGy,
        mapbox_style="stamen-terrain",
    )

    fig2 = px.scatter_mapbox(
        nodemaster_p,
        lat="lat",
        lon="lon",
        hover_name="index",
        hover_data=["non_served_energy", "load"],
        color="relative_service",
        color_continuous_scale="Viridis",  # range_color=(0, 20),
        zoom=3,
        height=400,
    )

    fig.add_trace(fig2.data[0])
    fig.layout.coloraxis2 = fig2.layout.coloraxis

    fig["data"][1]["marker"] = {
        "color": nodemaster_p["relative_service"],
        "coloraxis": "coloraxis2",
        "opacity": 1,
        "sizemode": "area",
        "sizeref": 0.01,
        "autocolorscale": False,
        "size": nodemaster_p["load"],
    }

    fig.layout.coloraxis2.colorbar.x = -0.05
    fig.layout.coloraxis.colorbar.x = -0.1

    for i in range(len(A_p.T)):
        for j in range(len(A_p.T[i])):
            if A_p.T[i][j] == 1:
                start_index = j
                start_node = nodemaster_p.iloc[start_index]
            if A_p.T[i][j] == -1:
                end_index = j
                end_node = nodemaster_p.iloc[end_index]
        edge = pd.concat((start_node, end_node), axis=1).T
        edge["flow"] = [flow_p[i], -flow_p[i]]

        fig.add_trace(
            go.Scattermapbox(
                lat=edge["lat"],
                lon=edge["lon"],
                mode="lines",
                line={"color": "#000000"},
                # text="Flow: " + str(flow_p[i])+" MW\n"
                #     "Direction: "+str(start_index)+"->"+str(end_index),
                # hoverinfo="text",
                hoverinfo="skip",
            )
        )

    fig.update_layout(
        mapbox_style="carto-positron",
        mapbox_zoom=3,
        mapbox_center={"lat": 37.0902, "lon": -95.7129},
    )
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})

    return fig


if __name__ == "__main__":
    app.run_server(debug=True)
