import optimization
import numpy as np
import pandas as pd


def test_optimization():
    """
    Test optimization with known test case (made up data for small network)
    """

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

    network = optimization.OptimizedNetwork(
        A_p=A_p,
        A_g=A_g,
        generators=generators,
        lines=lines,
        load=load,
        gas_supply=gas_supply,
        gas_demand=gas_demand,
        pipelines=pipelines,
    )

    network.optimize()

    assert abs(network.out_flow_p.loc[0] - 13.33) <= 0.01
