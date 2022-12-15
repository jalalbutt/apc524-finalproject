import numpy as np
import pandas as pd
import xarray as xr


class NetworkModel:
    """_summary_"""

    def __init__(self, array_PDE, latlon_start):
        self.x = np.zeros(4)

    def run_simulation(self):
        ...


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
    latlon_start = [45, 85]
    # other stuff- time, params, etc

    # initialize model
    model = NetworkModel(array_PDE, latlon_start)
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
            index=pd.Index([0, 1, 2], name="node"),
            name="Power flow by line (MW)",
        ),
        "flow_gas": pd.Series(
            [5, 6, 7],
            index=pd.Index([0, 1, 2], name="node"),
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
