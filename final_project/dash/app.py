from dash import Dash, dcc, html, Input, Output
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import random
import networkx as nx

timesteps = pd.Series(range(5))

df_density = pd.read_csv(
    "https://raw.githubusercontent.com/plotly/"
    "datasets/master/earthquakes-23k.csv"
)
us_cities = pd.read_csv(
    "https://raw.githubusercontent.com/plotly/"
    "datasets/master/us-cities-top-1k.csv"
)

g = nx.Graph()
g.add_nodes_from(us_cities["City"].tolist())

i = 0
while i < 20:
    randomCities = random.sample(us_cities["City"].tolist(), 2)
    g.add_edge(randomCities[0], randomCities[1], capacity=random.random())
    i += 1

app = Dash(__name__)

gws = "graph-with-slider"

app.layout = html.Div(
    [
        dcc.Graph(id=gws),
        dcc.Slider(
            timesteps.min(),
            timesteps.max(),
            step=None,
            value=timesteps.min(),
            marks={str(year): str(year) for year in timesteps.unique()},
            id="year-slider",
        ),
    ]
)


@app.callback(Output(gws, "figure"), Input("year-slider", "value"))
def update_figure(selected_year):
    # filtered_df = df[df.year == selected_year]

    fig = px.density_mapbox(
        df_density,
        lat="Latitude",
        lon="Longitude",
        z="Magnitude",
        radius=10,
        center=dict(lat=0, lon=180),
        zoom=0,
        color_continuous_scale=px.colors.diverging.RdGy,
        mapbox_style="stamen-terrain",
    )

    fig2 = px.scatter_mapbox(
        us_cities,
        lat="lat",
        lon="lon",
        hover_name="City",
        hover_data=["State", "Population"],
        color="lat",
        color_continuous_scale="Viridis",  # range_color=(0, 20),
        zoom=100,
        height=400,
    )
    fig2.update_traces(marker={"size": 135})

    fig.add_trace(fig2.data[0])
    fig.layout.coloraxis2 = fig2.layout.coloraxis

    fig["data"][1]["marker"] = {
        "color": us_cities["lat"],
        "coloraxis": "coloraxis2",
        "opacity": 1,
        "sizemode": "area",
        "sizeref": 0.01,
        "autocolorscale": False,
        "size": 10,
    }

    fig.layout.coloraxis2.colorbar.x = -0.05
    fig.layout.coloraxis.colorbar.x = -0.1

    for edge in g.edges():
        city1 = us_cities[us_cities["City"] == edge[0]]
        city2 = us_cities[us_cities["City"] == edge[1]]

        edgecities = pd.concat((city1.iloc[[0]], city2.iloc[[0]]))

        capacity = g[edge[0]][edge[1]]["capacity"]
        if capacity < 0.3:
            color = "#FF0000"
        if 0.1 <= capacity < 0.6:
            color = "#FF7F00"
        if 0.6 <= capacity:
            color = "#00FF00"

        fig.add_trace(
            go.Scattermapbox(
                lat=edgecities["lat"],
                lon=edgecities["lon"],
                mode="lines",
                line={"color": color},
                text="Capacity =" + str(capacity),
                hoverinfo="text",
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
