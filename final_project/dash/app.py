from dash import Dash, dcc, html, Input, Output
import plotly.express as px
import pandas as pd

df = pd.read_csv(
    "https://raw.githubusercontent.com/plotly/"
    "datasets/master/gapminderDataFiveYear.csv"
)

app = Dash(__name__)

gws = "graph-with-slider"

app.layout = html.Div(
    [
        dcc.Graph(id=gws),
        dcc.Slider(
            df["year"].min(),
            df["year"].max(),
            step=None,
            value=df["year"].min(),
            marks={str(year): str(year) for year in df["year"].unique()},
            id="year-slider",
        ),
    ]
)


@app.callback(Output(gws, "figure"), Input("year-slider", "value"))
def update_figure(selected_year):
    # filtered_df = df[df.year == selected_year]

    df_density = pd.read_csv(
        "https://raw.githubusercontent.com/plotly/"
        "datasets/master/earthquakes-23k.csv"
    )
    us_cities = pd.read_csv(
        "https://raw.githubusercontent.com/plotly/"
        "datasets/master/us-cities-top-1k.csv"
    )

    fig = px.density_mapbox(
        df_density,
        lat="Latitude",
        lon="Longitude",
        z="Magnitude",
        radius=10,
        center=dict(lat=0, lon=180),
        zoom=0,
        mapbox_style="stamen-terrain",
    )

    fig2 = px.scatter_mapbox(
        us_cities,
        lat="lat",
        lon="lon",
        hover_name="City",
        hover_data=["State", "Population"],
        color="lat",  # color_continuous_scale="Viridis", range_color=(0, 20),
        zoom=3,
        height=400,
    )

    fig.add_trace(fig2.data[0])
    fig.layout.coloraxis2 = fig2.layout.coloraxis

    fig["data"][1]["marker"] = {
        "color": us_cities["lat"],
        "coloraxis": "coloraxis2",
        "opacity": 1,
        "sizemode": "area",
        "sizeref": 0.01,
        "autocolorscale": False,
    }

    fig.layout.coloraxis2.colorbar.x = -0.05

    fig.update_layout(
        mapbox_style="carto-positron",
        mapbox_zoom=3,
        mapbox_center={"lat": 37.0902, "lon": -95.7129},
    )
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})

    return fig


if __name__ == "__main__":
    app.run_server(debug=True)
