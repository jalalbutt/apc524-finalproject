import numpy as np
import cvxpy as cp
import pandas as pd
import typing


class OptimizedNetwork:
    def __init__(
        self,
        A_p: np.ndarray,
        A_g: np.ndarray,
        generators: pd.DataFrame,
        lines: pd.DataFrame,
        load: pd.DataFrame,
        gas_supply: pd.DataFrame,
        gas_demand: pd.DataFrame,
        pipelines: pd.DataFrame,
        p_hat_mw: float = 100,
        voltage_angle_cap_radians: float = 2 * np.pi,
        slack_node_index: int = 0,
        nse_cost: float = 1000,
    ):
        """

        Initialize OptimizedNetwork with data and validate that inputs are
            in the correct form.

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
        """
        assert list(generators.columns) == [
            "name",
            "node_p",
            "node_g",
            "is_gas",
            "min_cap_mw",
            "max_cap_mw",
            "fuel_cost_dollars_per_mwh",
            "efficiency_gj_in_per_mwh_out",
        ]
        assert list(lines.columns) == [
            "from_node",
            "to_node",
            "reactance_pu",
            "capacity_mw",
        ]
        assert list(load.columns) == ["node", "load_mw"]
        assert list(gas_supply.columns) == ["node_g", "supply_gj"]
        assert list(gas_demand.columns) == ["node_g", "demand_gj"]
        assert list(pipelines.columns) == [
            "from_node",
            "to_node",
            "capacity_gj",
            "cost",
        ]
        assert load.shape[0] == A_p.shape[0]
        assert lines.shape[0] == A_p.shape[1]
        assert gas_supply.shape[0] == A_g.shape[0]
        assert gas_demand.shape[0] == A_g.shape[0]
        assert pipelines.shape[0] == A_g.shape[1]

        self.A_p = A_p
        self.A_g = A_g
        self.generators = generators
        self.lines = lines
        self.load = load
        self.gas_supply = gas_supply
        self.gas_demand = gas_demand
        self.pipelines = pipelines
        self.p_hat_mw = p_hat_mw
        self.voltage_angle_cap_radians = voltage_angle_cap_radians
        self.slack_node_index = slack_node_index
        self.nse_cost = nse_cost

        # output variables-- recorded after optimization

        # solver status
        self.out_status: typing.optional(str) = None

        # objective function value
        self.out_ofv: typing.optional(float) = None

        # non served energy for power
        self.out_nse_power: typing.optional(pd.Series) = None

        # non served energy for gas
        self.out_nse_gas: typing.optional(pd.Series) = None

        # flow on power lines
        self.out_flow_p: typing.optional(pd.Series) = None

        # flow on gas lines
        self.out_flow_g: typing.optional(pd.Series) = None

        # generation by generator
        self.out_generation: typing.optional(pd.Series) = None

        # gas supply
        self.out_gas_supply: typing.optional(pd.Series) = None

        # full problem for debugging purposes
        self.out_prob = None

    def optimize(self):
        """
        Optimize network and record results
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

        # Form and solve problem.
        prob = cp.Problem(obj, constraints)
        prob.solve()

        # record results

        # solver status
        self.out_status = prob.status

        # objective function value
        self.out_ofv = prob.value

        # non served energy for power
        self.out_nse_power = pd.Series(
            slack_p.value,
            index=pd.Index(nodes_p, name="node"),
            name="non-served energy (MWh)",
        )

        # non served energy for gas
        self.out_nse_gas = pd.Series(
            slack_g.value,
            index=pd.Index(nodes_g, name="node"),
            name="non-served energy (GJ)",
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
