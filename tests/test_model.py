import numpy as np
import pandas as pd
from model import NetworkModel


def test_network_model():
    """Test the NetworkModel object."""

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

    latlon_source = [40.36, -74.66]  # Princeton, NJ
    lat_bounds = [18, 52]
    lon_bounds = [-125, -64]
    num_lat_gridpoints = 100
    num_lon_gridpoints = 100
    lat_bounds_distance = 3500  # km
    lon_bounds_distance = 3500  # km
    source_radius = lat_bounds_distance * 0.01
    source_strength = 1000
    max_impact_threshold = 0.9

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

    assert (
        abs(model.out_array_result.values[1][50][-10] - 0.1338398663729742)
    ) < 0.01
