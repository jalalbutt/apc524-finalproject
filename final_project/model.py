import numpy as np
import pandas as pd
import xarray as xr
from dataclasses import dataclass

# import typing
import optimization

# import perturb  # does not exist yet


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

    # times at which to run the optimization, after initialization
    times_opt: list

    # PDE params
    array_PDE: xr.DataArray
    latlon_start: list
    timesteps: list  # could be in a different form
    # add other PDE params

    # default inputs that are not likely to change
    p_hat_mw: float = 100
    voltage_angle_cap_radians: float = 2 * np.pi
    slack_node_index: int = 0
    nse_cost: float = 1000

    # # outputs- not meant to be inputted
    # out_array_result: typing.optional(xr.DataArray) = None
    # out_opt_result: typing.optional(dict) = None

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

        # PDE_input_array = self.array_PDE.values

        # # get coordinate closest to latlon_start
        # coordinate_start = [5, 5]

        # PDE_model = perturb.Model(
        #     PDE_input_array, coordinate_start, self.timesteps
        # )

        # PDE_model.simulate()

        # PDE_output_array = PDE_model.out_array

        # # create out_array_result by adding coordinates
        # self.out_array_result = xr.DataArray(PDE_output_array)

        # # at desired timesteps of out_array_result, calculate impact on
        # # optimization inputs, and run optimization


def test_network_model():
    """currently returns the fake outputs"""

    # optimization network characteristics

    # set up parameters
    A_p = np.array(
        [[1, 0, -1], [-1, 1, 0], [0, -1, 1]]
    )  # edge-node incidence matrix for power
    A_g = A_p  # same for gas

    generators = pd.DataFrame()
    generators["name"] = ["coal", "gas", "pv"]
    generators["node_p"] = [0, 1, 2]
    generators["node_g"] = [0, 1, 2]
    generators["is_gas"] = [False, True, False]
    generators["min_cap_mw"] = [0, 0, 0]
    generators["max_cap_mw"] = [100, 200, 50]
    generators["fuel_cost_dollars_per_mwh"] = [5, 10, 50]
    generators["efficiency_gj_in_per_mwh_out"] = [5, 5, 5]

    lines = pd.DataFrame()
    lines["from_node"] = [0, 1, 2]
    lines["to_node"] = [1, 2, 0]
    lines["reactance_pu"] = [0.3, 0.3, 0.3]
    lines["capacity_mw"] = [500, 500, 500]

    load = pd.DataFrame()
    load["node"] = [0, 1, 2]
    load["load_mw"] = [0, 0, 40]

    gas_supply = pd.DataFrame()
    gas_supply["node_g"] = [0, 1, 2]
    gas_supply["supply_gj"] = [1000, 1000, 1000]  # exajoules

    gas_demand = pd.DataFrame()
    gas_demand["node_g"] = [0, 1, 2]
    gas_demand["demand_gj"] = [100, 100, 100]

    pipelines = pd.DataFrame()
    pipelines["from_node"] = [0, 1, 2]
    pipelines["to_node"] = [1, 2, 0]
    pipelines["capacity_gj"] = [100, 100, 100]
    pipelines["cost"] = [5, 5, 5]

    # coordinates of nodes
    nodes_p = pd.DataFrame(
        index=pd.Index(np.arange(A_p.shape[0]), name="node")
    )
    nodes_p["lat"] = [30, 35, 40]
    nodes_p["lon"] = [75, 80, 85]

    nodes_g = nodes_p.copy(deep=True)

    # inputs for PDE

    # initial array
    lat = np.linspace(25, 45, 10)
    lon = np.linspace(70, 90, 10)
    init_array_PDE = np.zeros([10, 10])
    array_PDE = xr.DataArray(
        init_array_PDE, coords=[lat, lon], dims=["lat", "lon"]
    )

    # params for PDE go here
    # start of the disturbance
    times_opt = [1, 2, 3, 4, 5]
    latlon_start = [45, 85]
    timesteps = [0, 1, 2, 3, 4, 5]
    # other stuff- time, params, etc

    # initialize model
    model = NetworkModel(
        A_p,
        A_g,
        generators,
        lines,
        load,
        gas_supply,
        gas_demand,
        pipelines,
        nodes_p,
        nodes_g,
        times_opt,
        array_PDE,
        latlon_start,
        timesteps,
    )
    model.run_simulation()

    # fake results that show what the format should be
    array_result = xr.DataArray(
        np.random.rand(10, 10, 5),
        coords=[lat, lon, [0, 1, 2, 3, 4]],
        dims=["lat", "lon", "time"],
    )
    results_dict = {
        "nse_power": pd.DataFrame(
            {"non_served_energy": [0, 1, 2], "load": [2, 2, 2]},
            index=pd.Index([0, 1, 2], name="node"),
        ),
        "nse_gas": pd.DataFrame(
            {"non_served_energy": [0, 1, 2], "load": [2, 2, 2]},
            index=pd.Index([0, 1, 2], name="node"),
        ),
        "flow_power": pd.Series(
            [5, 6, 7],
            index=pd.Index([0, 1, 2], name="edge"),
            name="Power flow by line (MW)",
        ),
        "flow_gas": pd.Series(
            [5, 6, 7],
            index=pd.Index([0, 1, 2], name="edge"),
            name="Power flow by line (MW)",
        ),
    }
    opt_result = {0: results_dict, 5: results_dict}

    # our results are:
    # array_result: self-described dimensions
    # opt_result: dictionary, keys are time steps that optimization is run
    # for. values are each a dict containing the data for each time step.
    # A_p and A_g: arrays with edge-node incidence matrix (fine to focus
    # on power network) nodes_p and nodes_g: DFs with coordinates of the
    # nodes for the gas and power network (fine to focus on power network)
    return array_result, opt_result, A_p, A_g, nodes_p, nodes_g


if __name__ == "__main__":
    array_result, opt_result, A_p, A_g, nodes_p, nodes_g = test_network_model()

    print("")
