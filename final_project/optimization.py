"""
optimization.py
---------------
The optimization module of the project
"""

import numpy as np
import cvxpy as cp
import pandas as pd
import typing
from dataclasses import dataclass


@dataclass
class OptimizedNetwork:
    """
    The class defining the optimization problem for the network.
    Set up as a dataclass, with additional methods.

    Args:
        A_p (np.ndarray): Edge-node incidence matrix for power network.
            Rows are nodes, columns are edges.
        A_g (np.ndarray): Edge-node incidence matrix for gas network.
            Rows are nodes, columns are edges.
        generators (pd.DataFrame): Pandas DF with electric generator
            characteristics. Columns specified in assert statements.
        lines (pd.DataFrame): Pandas DF with electric transmission line
            (edge) characteristics. Columns specified in assert
            statements.
        load (pd.DataFrame): Pandas DF with load characteristics. Columns
            specified in assert statements.
        gas_supply (pd.DataFrame): Pandas DF with gas supply
            characteristics. Columns specified in assert statements.
        gas_demand (pd.DataFrame): Pandas DF with gas demand
            characteristics. Columns specified in assert statements.
        pipelines (pd.DataFrame): Pandas DF with gas pipeline
            characteristics. Columns specified in assert statements.
        p_hat_mw (float, optional): System base power (MW). Defaults to
            100.
        voltage_angle_cap_radians (float, optional): Maximum voltage angle
            difference in radians. Defaults to 2*np.pi.
        slack_node_index (int, optional): Index of slack node
            (zero-indexed). Defaults to 0.
        nse_cost (float, optional): Cost of non-served energy. Defaults to
            1000.

    There are also output variables that are initialized as part of the
    dataclass, but are not meant to be inputted.

    """

    A_p: np.ndarray
    A_g: np.ndarray
    generators: pd.DataFrame
    lines: pd.DataFrame
    load: pd.DataFrame
    gas_supply: pd.DataFrame
    gas_demand: pd.DataFrame
    pipelines: pd.DataFrame
    p_hat_mw: float = 100
    voltage_angle_cap_radians: float = 2 * np.pi
    slack_node_index: int = 0
    nse_cost: float = 1000

    # output variables-- recorded after optimization

    # solver status
    out_status: typing.Optional[str] = None

    # objective function value
    out_ofv: typing.Optional[float] = None

    # non served energy for power
    out_nse_power: typing.Optional[pd.Series] = None

    # non served energy for gas
    out_nse_gas: typing.Optional[pd.Series] = None

    # flow on power lines
    out_flow_p: typing.Optional[pd.Series] = None

    # flow on gas lines
    out_flow_g: typing.Optional[pd.Series] = None

    # summary of NSE and flows
    out_energy_and_flows: typing.Optional[dict] = None

    # generation by generator
    out_generation: typing.Optional[pd.Series] = None

    # gas supply
    out_gas_supply: typing.Optional[pd.Series] = None

    # full problem for debugging purposes
    out_prob = None

    def __post_init__(self):
        """
        Assert that inputs are in the correct form so that the optimization
        will run.
        """
        assert list(self.generators.columns) == [
            "name",
            "node_p",
            "node_g",
            "is_gas",
            "min_cap_mw",
            "max_cap_mw",
            "fuel_cost_dollars_per_mwh",
            "efficiency_gj_in_per_mwh_out",
        ]
        assert list(self.lines.columns) == [
            "from_node",
            "to_node",
            "reactance_pu",
            "capacity_mw",
        ]
        assert list(self.load.columns) == ["node", "load_mw"]
        assert list(self.gas_supply.columns) == ["node_g", "supply_gj"]
        assert list(self.gas_demand.columns) == ["node_g", "demand_gj"]
        assert list(self.pipelines.columns) == [
            "from_node",
            "to_node",
            "capacity_gj",
            "cost",
        ]
        assert self.load.shape[0] == self.A_p.shape[0]
        assert self.lines.shape[0] == self.A_p.shape[1]
        assert self.gas_supply.shape[0] == self.A_g.shape[0]
        assert self.gas_demand.shape[0] == self.A_g.shape[0]
        assert self.pipelines.shape[0] == self.A_g.shape[1]

    def optimize(self):
        """
        Optimize network and record results.
        """

        # sets
        nodes_p = list(range(self.A_p.shape[0]))
        edges_p = list(range(self.A_p.shape[0]))
        nodes_g = list(range(self.A_g.shape[0]))
        edges_g = list(range(self.A_g.shape[0]))
        gens = list(self.generators.index)

        # decision variables

        # voltage angle at each node (radians)
        v_n = cp.Variable(len(nodes_p))

        # generation
        gen = cp.Variable(len(self.generators), nonneg=True)

        # slack generation for power
        slack_p = cp.Variable(len(nodes_p), nonneg=True)

        # gas supply
        supply_gas = cp.Variable(len(nodes_g), nonneg=True)

        # gas flow
        flow_gas = cp.Variable(len(edges_g))

        # gas flow positive component
        flow_gas_pos = cp.Variable(len(edges_g), nonneg=True)

        # gas flow negative component
        flow_gas_neg = cp.Variable(len(edges_g), nonneg=True)

        # slack demand for gas
        slack_g = cp.Variable(len(nodes_g), nonneg=True)

        # expressions

        # power flow on an edge
        p_e = []

        for e in edges_p:
            p_e += [
                self.p_hat_mw
                * (1 / self.lines.reactance_pu[e])
                * sum(self.A_p[n, e] * v_n[n] for n in nodes_p)
            ]

        # constraints

        constraints = []

        # power energy balance
        for n in nodes_p:
            constraints += [
                sum(
                    gen[g] if self.generators.loc[g].node_p == n else 0
                    for g in gens
                )
                - sum(self.A_p[n, e] * p_e[e] for e in edges_p)
                - self.load.loc[n].load_mw
                - slack_p[n]
                == 0
            ]

        # tx line limits
        for e in edges_p:
            constraints += [-1 * self.lines.loc[e].capacity_mw <= p_e[e]]
            constraints += [p_e[e] <= self.lines.loc[e].capacity_mw]

        # voltage angle differential limits
        for e in edges_p:
            constraints += [
                -1 * self.voltage_angle_cap_radians
                <= sum(self.A_p[n, e] * v_n[n] for n in nodes_p)
            ]
            constraints += [
                sum(self.A_p[n, e] * v_n[n] for n in nodes_p)
                <= self.voltage_angle_cap_radians
            ]

        # slack bus
        constraints += [v_n[self.slack_node_index] == 0]

        # generator limits
        for g in gens:
            constraints += [self.generators.loc[g].min_cap_mw <= gen[g]]
            constraints += [gen[g] <= self.generators.loc[g].max_cap_mw]

        # gas energy balance
        for n in nodes_g:
            constraints += [
                supply_gas[n]
                - sum(self.A_g[n, e] * flow_gas[e] for e in edges_g)
                - self.gas_demand.loc[n].demand_gj
                - sum(
                    gen[g]
                    * self.generators.loc[g].efficiency_gj_in_per_mwh_out
                    if self.generators.loc[g].is_gas
                    else 0
                    for g in gens
                )
                - slack_g[n]
                == 0
            ]

        # gas supply limits
        for n in nodes_g:
            constraints += [0 <= supply_gas[n]]
            constraints += [supply_gas[n] <= self.gas_supply.loc[n].supply_gj]

        # gas flow constraints
        for e in edges_g:
            constraints += [
                -1 * self.pipelines.loc[e].capacity_gj <= flow_gas[e]
            ]
            constraints += [flow_gas[e] <= self.pipelines.loc[e].capacity_gj]

        # set up absolute value variables
        for e in edges_g:
            constraints += [flow_gas_pos[e] >= flow_gas[e]]
            constraints += [flow_gas_neg[e] >= -1 * flow_gas[e]]

        # objective
        obj = cp.Minimize(
            sum(
                gen[g] * self.generators.loc[g].fuel_cost_dollars_per_mwh
                for g in gens
            )
            + sum(slack_p[n] * self.nse_cost for n in nodes_p)
            + sum(
                (flow_gas_pos[e] + flow_gas_neg[e])
                * self.pipelines.loc[e].cost
                for e in edges_g
            )
            + sum(slack_g[n] * self.nse_cost for n in nodes_g)
        )

        # form and solve problem
        prob = cp.Problem(obj, constraints)
        prob.solve()

        # record results

        # solver status
        self.out_status = prob.status

        # objective function value
        self.out_ofv = prob.value

        # non served energy for power
        self.out_nse_power = pd.DataFrame(
            {
                "non_served_energy": slack_p.value,
                "load": self.load.load_mw.values,
            },
            index=pd.Index(nodes_p, name="node"),
        )

        # non served energy for gas
        self.out_nse_gas = pd.DataFrame(
            {
                "non_served_energy": slack_g.value,
                "load": self.gas_demand.demand_gj.values,
            },
            index=pd.Index(nodes_g, name="node"),
        )

        # flow on power lines
        self.out_flow_p = pd.Series(
            [x.value for x in p_e],
            index=pd.Index(nodes_p, name="node"),
            name="Power flow by line (MW)",
        )

        # flow on gas lines
        self.out_flow_g = pd.Series(
            flow_gas.value,
            index=pd.Index(nodes_g, name="node"),
            name="Gas flow by line (GJ)",
        )

        # NSE and flows combined into one dict
        self.out_energy_and_flows = {
            "nse_power": self.out_nse_power,
            "nse_gas": self.out_nse_gas,
            "flow_power": self.out_flow_p,
            "flow_gas": self.out_flow_g,
        }

        # generation by generator
        self.out_generation = pd.Series(
            gen.value,
            index=pd.Index(gens, name="generator"),
            name="Generation by generator (MWh)",
        )

        # gas supply
        self.out_gas_supply = pd.Series(
            supply_gas.value,
            index=pd.Index(nodes_g, name="node"),
            name="Gas supply (GJ)",
        )

        # full problem for debugging purposes
        self.out_prob = prob
