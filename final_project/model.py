"""
model.py
--------
The module coupling the two networks
"""
import numpy as np
import pandas as pd
import xarray as xr
from dataclasses import dataclass, field
import typing
import optimization
import perturb
from pathlib import Path


@dataclass
class NetworkModel:
    """
    A class which holds data for a coupled PDE-optimization model,
    and which runs the simulation.
    """

    # optimization inputs before perturbation
    A_p: np.ndarray
    A_g: np.ndarray
    generators: pd.DataFrame
    lines: pd.DataFrame
    load: pd.DataFrame
    gas_supply: pd.DataFrame
    gas_demand: pd.DataFrame
    pipelines: pd.DataFrame

    # coordinates of nodes on power and gas networks
    nodes_p: pd.DataFrame
    nodes_g: pd.DataFrame

    # PDE params
    latlon_source: list
    lat_bounds: list
    lon_bounds: list
    num_lat_gridpoints: int
    num_lon_gridpoints: int
    lat_bounds_distance: float
    lon_bounds_distance: float
    source_radius: float
    source_strength: float
    max_impact_threshold: float

    # default inputs that are not likely to change
    p_hat_mw: float = 100
    voltage_angle_cap_radians: float = 2 * np.pi
    slack_node_index: int = 0
    nse_cost: float = 1000
    source_type: str = "delta"
    solve_type: str = "static"
    timesteps: list = field(
        default_factory=lambda: [
            0,
            1,
        ]
    )
    # times at which to run the optimization, after initialization
    times_opt: list = field(default_factory=lambda: [1])

    # # outputs- not meant to be inputted
    out_array_result: typing.Optional[xr.DataArray] = None
    out_opt_result: typing.Optional[dict] = None

    def __post_init__(self):
        """
        Run initial optimization before perturbation
        Note this will conveniently also run the validation steps
        from OptimizedNetwork so we don't have to repeat them here.
        """
        network_init = optimization.OptimizedNetwork(
            self.A_p,
            self.A_g,
            self.generators,
            self.lines,
            self.load,
            self.gas_supply,
            self.gas_demand,
            self.pipelines,
            self.p_hat_mw,
            self.voltage_angle_cap_radians,
            self.slack_node_index,
            self.nse_cost,
        )

        network_init.optimize()

        self.out_opt_result = {}
        self.out_opt_result[0] = network_init.out_energy_and_flows

    def run_simulation(self):
        """
        Run the PDE perturbation simulation, translate the results
        into their impact on the optimization, and then run the
        optimization for the desired time steps.
        """

        # Run PerturbedNetwork

        # get coordinates of source in array space

        # first, get coordinates of eventual output
        lat = np.linspace(
            self.lat_bounds[0], self.lat_bounds[1], self.num_lat_gridpoints
        )
        lon = np.linspace(
            self.lon_bounds[0], self.lon_bounds[1], self.num_lon_gridpoints
        )
        time = np.array(self.timesteps)

        # now, get the coordinates of the source in m,n space
        source_coord_m = np.absolute(lat - self.latlon_source[0]).argmin()
        source_coord_n = np.absolute(lat - self.latlon_source[0]).argmin()

        # this can be anything which follows the ArrayModifier protocol
        PDE_model = perturb.PerturbedNetwork(
            m=self.num_lat_gridpoints,
            n=self.num_lon_gridpoints,
            L_m=self.lat_bounds_distance,
            L_n=self.lon_bounds_distance,
            f_type=self.source_type,
            source_center=[source_coord_m, source_coord_n],
            radius=self.source_radius,
            source_strength=self.source_strength,
        )

        PDE_output_array = PDE_model.solve(
            timesteps=self.timesteps, method="static_solve"
        )

        # create out_array_result by adding coordinates
        self.out_array_result = xr.DataArray(
            np.absolute(PDE_output_array[:, 1:-1, 1:-1]),
            dims=["time", "lat", "lon"],
            coords=[time, lat[::-1], lon],
        )

        # at desired timesteps of out_array_result, calculate impact on
        # optimization inputs, and run optimization
        for t in self.times_opt:
            nodes_impact = self.nodes_p.copy(deep=True)
            nodes_impact["perturb_abs"] = 0
            nodes_impact["perturb_rel"] = 0

            # for each node, calculate gen_multiplier, as 1 minus
            # the ratio between the perturbation value and the threshold.
            for i in nodes_impact.index:
                perturb_val = self.out_array_result.sel(
                    lat=nodes_impact.loc[i, "lat"],
                    lon=nodes_impact.loc[i, "lon"],
                    time=t,
                    method="nearest",
                )
                nodes_impact.loc[i, "perturb_abs"] = perturb_val
                nodes_impact.loc[i, "gen_multiplier"] = 1 - (
                    perturb_val / self.max_impact_threshold
                )

            # perturb available generation for each generator
            # according to the gen_multiplier.
            generators_perturb = self.generators.copy(deep=True)
            for i in generators_perturb.index:
                generators_perturb.loc[i, "max_cap_mw"] = (
                    generators_perturb.loc[i, "max_cap_mw"]
                    * nodes_impact.loc[
                        generators_perturb.loc[i, "node_p"], "gen_multiplier"
                    ]
                )

            # optimize the perturbed network.
            network_perturbed = optimization.OptimizedNetwork(
                self.A_p,
                self.A_g,
                generators_perturb,
                self.lines,
                self.load,
                self.gas_supply,
                self.gas_demand,
                self.pipelines,
                self.p_hat_mw,
                self.voltage_angle_cap_radians,
                self.slack_node_index,
                self.nse_cost,
            )
            network_perturbed.optimize()

            # record results
            self.out_opt_result[t] = network_perturbed.out_energy_and_flows


def run_model(
    lat: float, lon: float, radius: float
) -> typing.Tuple[
    xr.DataArray, dict, np.ndarray, np.ndarray, pd.Series, pd.Series
]:
    """Run the model."""

    # optimization network characteristics

    data_path = Path("./final_project/data")
    # set up parameters
    A_p = pd.read_csv(data_path / "A_p.csv", index_col=0).values
    A_g = A_p  # same for gas

    generators = pd.read_csv(data_path / "generators.csv")
    lines = pd.read_csv(data_path / "lines.csv")
    load = pd.read_csv(data_path / "load.csv")
    gas_supply = pd.read_csv(data_path / "gas_supply.csv")
    gas_demand = pd.read_csv(data_path / "gas_demand.csv")
    pipelines = pd.read_csv(data_path / "pipelines.csv")
    nodes_g = pd.read_csv(data_path / "nodes_g.csv")
    nodes_p = pd.read_csv(data_path / "nodes_p.csv")

    # inputs for PDE

    latlon_source = [lat, lon]
    lat_bounds = [18, 52]
    lon_bounds = [-125, -64]
    num_lat_gridpoints = 300
    num_lon_gridpoints = 300
    lat_bounds_distance = 3500  # km
    lon_bounds_distance = 3500  # km
    source_radius = radius
    source_strength = 1000
    max_impact_threshold = 10000000000

    # initialize model
    model = NetworkModel(
        A_p=A_p,
        A_g=A_g,
        generators=generators,
        lines=lines,
        load=load,
        gas_supply=gas_supply,
        gas_demand=gas_demand,
        pipelines=pipelines,
        nodes_p=nodes_p,
        nodes_g=nodes_g,
        latlon_source=latlon_source,
        lat_bounds=lat_bounds,
        lon_bounds=lon_bounds,
        num_lat_gridpoints=num_lat_gridpoints,
        num_lon_gridpoints=num_lon_gridpoints,
        lat_bounds_distance=lat_bounds_distance,
        lon_bounds_distance=lon_bounds_distance,
        source_radius=source_radius,
        source_strength=source_strength,
        max_impact_threshold=max_impact_threshold,
    )
    model.run_simulation()

    # our results are:
    # array_result: self-described dimensions (in xarray)
    # opt_result: dictionary, keys are time steps that optimization is run
    # for. values are each a dict containing the data for each time step.
    # A_p and A_g: arrays with edge-node incidence matrix (fine to focus
    # on power network) nodes_p and nodes_g: DFs with coordinates of the
    # nodes for the gas and power network
    return (
        model.out_array_result,
        model.out_opt_result,
        model.A_p,
        model.A_g,
        model.nodes_p,
        model.nodes_g,
    )


if __name__ == "__main__":
    lat = 40.34578
    lon = -74.65256
    radius = 35
    ar, a, b, c, d, e = run_model(lat, lon, radius)
    print("")
