"""
APC 524
Final Project
Quantifying resilience of coupled networks

@author Jalal Butt
Date: 12-03-22


Module file handling the perturbation network
----------------------------------------------


Outputs:
--------
    - time-evolution of perturbation
        --> entirely independent of what's happening on the grid. I.e.,
            need not output individual time-steps
"""


# (Elliptic Equations hw)
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sparse
import scipy.sparse.linalg as linalg

# from scipy import signal

from dataclasses import dataclass
import traceback
import warnings


# Arb parameters for testing.
"""
m = 100 # number of grid-points
n = 100
n = 100 #  currently not implemented
L_m = 10
L_n = 10

f_type = 'oscillatory'
f_type = 'delta'
source_point = (-2,-2)
source_strength = 1000
radius = 10
"""


@dataclass
class PerturbedNetwork:
    """
    Class for the perturbation network model.

    Inputs:
    ------
                    f_type :   forcing function type (from a finite set)
                    m      :   # of grid points
                    radius :   perturbation extent in terms of grid [points]
                    source_center :   coordinate pair; % of (L_m, L_n)
                        (not indices)


    Outputs:
    -------
                    U :   np.array (2D if len(timesteps)== 1); 3D if > 1)

    Methods:
    --------
                    construct_A:   construct the spatial operator
                    construct_source:   construct perturb source (several
                        options)
                    solve_laplacian_system:   numerically solve laplacian
                    compute_discretization_error:   <only valid for an
                                                    analytic RHS>
                    plot_solution:   plot the solution
                    plot_error:    plot the error

                    static_solve:   compute the time-independent solution


    Usage:
    -----
    Example:
                    m = 100 # number of grid-points
                    n = 100
                    n = 100 #  currently not implemented
                    L_m = 10
                    L_n = 10

                    f_type = 'oscillatory'
                    f_type = 'delta'
                    source_center=  = ()
                    source_strength = 1000
                    radius = m/L_m* 0.1

                    pertNet2 = PerturbedNetwork(f_type= f_type,
                                source_point= source_center,
                                m= m, L_m= L_m, L_n= L_n, radius= radius)
                    U = pertNet2.static_solve()
                    pertNet2.plot_solution()


    """

    # -- Inputs:
    f_type: str
    source_center: [
        tuple,
        list,
        np.ndarray,
    ]  # coordinate pair; % of (L_m, L_n)
    m: int
    n: int
    L_m: int
    L_n: int
    radius_grid_fract: float = 0.01  # frac of grid pts (of the smaller dim.)
    source_strength: float = 1000
    source_center_basis: str = "grid_index"  # expected pipeline input

    def __post_init__(self):  # what should i do w this? generate tests?
        self.failure = False
        self.delta_x = self.L_m / (self.m + 1)
        self.delta_y = self.L_n / (self.n + 1)
        self.radius = self.m / self.L_m * self.radius_grid_fract
        self.x = np.linspace(0, self.L_m, self.m + 2)
        self.y = np.linspace(0, self.L_n, self.n + 2)

        # grid-choice for source center:
        if self.source_center_basis == "grid_index":  # need to translate;
            self.source_center = (
                self.source_center[1] / self.m,
                (self.m - 1 - self.source_center[0]) / self.n,
            )
        elif self.source_center_basis == "coordinate":  # good;
            self.source_center = (
                self.source_center[0] / self.m,
                self.source_center[1] / self.n,
            )
        elif self.source_center_basis == "fract":  # bad
            self.source_center = self.source_center

        try:
            assert (
                len(self.source_center) == 2
            ), "Source point must be coord. pair"
            assert self.m > 10, "Grid must be AT LEAST 10x10 points"
            assert self.n > 10, "Grid must be AT LEAST 10x10 points"
            assert (
                self.radius < 0.1 * self.L_m
            ), "radius can't span >10% of grid"
            assert self.f_type in [
                "delta",
                "oscillatory",
            ], "Requested source-type is not yet configured."
            assert (
                self.radius > 0.6 * self.delta_x
            ), "Source cannot be resolved on the grid due to small radius"

            # temporary restrictions
            assert (
                self.m == self.n
            ), "Cannot handle non-square grid atm. Please make m==n"

        except AssertionError:
            traceback.print_exc()
            self.failure = True
            self.intialization_failure_message = (
                "Cannot complete this action due to initialization failure"
            )

    # Construct A
    def construct_A(self, m, delta_x):
        """
        Construct the 5-point Laplacian operator.

        Inputs:
        ------
        m [int]
        delta_x [float] : spatial-spacing

        Outputs:
        -------
        A [sparse obj] : 2D 5-point Laplacian operator

        Improvements TBD:
        ----------------
        [] Square grid --> rectangular grid
        """
        e = np.ones(m)
        T = sparse.spdiags([e, -4.0 * e, e], [-1, 0, 1], m, m)
        S = sparse.spdiags([e, e], [-1, 1], m, m)
        EYE = sparse.eye(m)
        A = sparse.kron(EYE, T) + sparse.kron(S, EYE)
        A /= delta_x**2

        return A

    # self.A = construct_A(self.m, self.delta_x)

    # Right-hand-side (forcing-function)
    def construct_source(
        self,
        x,
        y,
        f_type="oscillatory",
        source_strength=1,
        impulse_extent=30,
        source_center=(10, 12),
        radius=2,
    ):
        """
        Construct source (i.e. bomb source)

        Inputs:
        ------
        x, y [np.array]  : x and y arrays, to be put into mesh form
        f_type [str]     : forcing-function type (oscillatory, localized,
                            etc.)
                        - oscillatory could correspond to a heat-wave.
        source_point     :
        source_strength  :
        radius :

        Outputs:
        -------
        X, Y [np.array, [len(x) x len(y)]] : x,y mesh
        f [np.array, [len(x) x len(y)] ]   : discrete forcing function

        """
        X_source, Y_source = np.meshgrid(x[1:-1], y[1:-1])

        if f_type.lower() == "oscillatory":
            f = -2.0 * np.sin(X_source) * np.sin(Y_source)
        elif f_type.lower() == "delta":
            circle = np.sqrt(
                (X_source - source_center[0] * self.L_m) ** 2
                + (Y_source - source_center[1] * self.L_n) ** 2
            )
            # circle = np.sqrt(
            #     (X_source - source_center[1] * self.L_m) ** 2
            #     + (Y_source - source_center[0] * self.L_n) ** 2
            # )
            f = np.zeros(X_source.shape)
            f[circle < radius] += source_strength

        else:
            print(f_type + " doesn't exist yet as a source constructor.")

        return X_source, Y_source, f

    # -- -----Discretizations
    # -- ---------------------------------
    def solve_laplacian_system(self, m, A, f):
        """
        Solve the Poisson eq. w/ the respective forcing-function.

        Inputs:
        -------
        m : # points
        A : discretized laplacian
        f : forcing function

        Outputs:
        -------
        U [np.array]  : heat-kernel
        """
        U = np.zeros((m + 2, m + 2))  # m+2 to accommodate the BCs
        U[1:-1, 1:-1] = linalg.spsolve(
            A, f.reshape(m**2, order="F")
        ).reshape((m, m), order="F")

        return U

    def forward_time_step(self, U, delta_t, A, f, k, method="forward euler"):
        """
        Forward time-step on the perturbation equation.

        Inputs:
        ------
                        U :   previous timestep solution
                        A :   discretized laplacian system
                        f :   source
                        method:   type of forward time-step

        Outputs:
        -------
                        U_new
        """

        if method == "forward euler":
            U_new = np.zeros(U.shape)
            # R = slice(1, -1)  # cut off ends
            # R = slice(0, None)  # all points  # not used atm, flake mad

            # print("\n")
            # print("A.shape:  {}".format(A.shape))
            # print("U.shape:  {}".format(U.shape))
            # print("f.shape:  {}".format(f.shape))
            # print("\n")
            # U_new[R,R] = U[R,R] + delta_t * (A * U[R,R] + f)  # U e [m, m];
            #   but A isn't
            U_new = U + delta_t * (A * U + f)  # U e [m, m]; but A isn't

        return U_new

    def compute_discretization_error(self, x, y, U):
        """
        Compute numerical error introduced by discretization.

        **Currently only works for the oscillatory solution**

        Inputs:
        -------
        x,y [np.array, grid]  :
        U [np.array, grid]    :

        Outputs:
        -------
        X,Y
        norm_error
        grid_error

        """
        # Error
        X, Y = np.meshgrid(x, y)
        norm_error = np.linalg.norm(
            (x[1] - x[0]) * (U - np.sin(X) * np.sin(Y)), ord=1
        )

        grid_error = np.abs(U - np.sin(X) * np.sin(Y))

        print("Normalization error" + str(norm_error))

        return X, Y, norm_error, grid_error

    # ------ Solution plotting
    # ------------------------
    def plot_solution(
        self, U: str = "static", addendum: str = "", show: bool = False
    ):
        if self.failure:
            return print(self.intialization_failure_message)

        # tests for solution plotting:

        if U != "static" and (U.shape != (self.X.shape)):
            warnings.warn(
                "The requested solution plot does not match the grid\
                 dimensions (X,Y). Check again."
            )

        # Plot solution
        fig1 = plt.figure()
        axes1 = fig1.add_subplot(1, 1, 1)
        if U == "static":
            sol_plot = axes1.pcolor(
                self.X, self.Y, self.U, cmap=plt.get_cmap("RdBu_r")
            )
        else:
            if len(U.shape) > 2 and U.shape[0] > 1:
                print(
                    "\nNOTE: the requested solution to plot contained \
                    multiple time-steps"
                )
                print("-----    only plotting the first one")
                U = U[0]
            sol_plot = axes1.pcolor(
                self.X, self.Y, U, cmap=plt.get_cmap("RdBu_r")
            )
        axes1.set_title("Solution u(x,y)  " + addendum)
        axes1.set_xlabel("x")
        axes1.set_ylabel("y")
        axes1.set_xlim((0.0, self.L_m))
        axes1.set_ylim((0.0, self.L_n))
        cbar1 = fig1.colorbar(sol_plot, ax=axes1)
        cbar1.set_label("u(x, y)")

        if show:
            plt.show()

    def plot_error(self):  # , X,Y,norm_error,grid_error):
        fig2 = plt.figure()
        axes2 = fig2.add_subplot(1, 1, 1)
        sol_plot = axes2.pcolor(
            self.X, self.Y, self.grid_error, cmap=plt.get_cmap("RdBu_r")
        )
        axes2.set_title("Error |U - u|")
        axes2.set_xlabel("x")
        axes2.set_ylabel("y")
        axes2.set_xlim((0.0, self.L_m))
        axes2.set_ylim((0.0, self.L_n))
        cbar2 = fig2.colorbar(sol_plot, ax=axes2)
        cbar2.set_label("u(x, y)")

        plt.show()

    # ------- Solutions
    # -----------------
    def static_solve(self, **kwargs):
        """
        Solution to the laplacian.

        Outputs:
        -------
        U

        """
        if self.failure:
            return print(self.intialization_failure_message)

        self.A = self.construct_A(self.m, self.delta_x)

        # what is this X,Y used for besides source construction?
        self.X_source, self.Y_source, self.f = self.construct_source(
            self.x,
            self.y,
            f_type=self.f_type,
            source_strength=self.source_strength,
            source_center=self.source_center,
            radius=self.radius,
        )
        self.U = self.solve_laplacian_system(self.m, self.A, self.f)

        (
            self.X,
            self.Y,
            self.norm_error,
            self.grid_error,
        ) = self.compute_discretization_error(self.x, self.y, self.U)

        # first output is for infrastr network initialization,
        # second is first solution
        self.U_t = np.zeros(shape=(2, *self.U.shape))
        self.U_t[1] = self.U.copy()

        return self.U_t

    def time_evolve(self, timesteps, k: float = 1.0):
        """
        Time-stepped solution of the diffusive-perturbation model.

        Inputs:
        ------
                        k :   diffusion coefficient

        Outputs:
        -------
                        U_time :   [np.array (time, space1, space2)]
        """
        if self.failure:
            return print(self.intialization_failure_message)

        # build Laplacian stencil
        self.A = self.construct_A(self.m, self.delta_x)

        # build source function
        self.X_source, self.Y_source, self.f = self.construct_source(
            self.x,
            self.y,
            f_type=self.f_type,
            source_strength=self.source_strength,
            source_center=self.source_center,
            radius=self.radius,
        )
        print(f"Maximum value of self.f:  {np.max(self.f)}")

        # determine initial condition:
        R = slice(1, -1)
        U_0 = self.static_solve()[R, R].reshape(
            -1
        )  # subset... fuck it. we'll take the L on the edges.

        # initialize for time-evolution
        delta_t = self.delta_x**2.0 / (4 * k) * 0.8  # constr.: dt < dx^2/4k
        sol_timebase = np.linspace(
            timesteps[0],
            timesteps[-1],
            int((timesteps[-1] - timesteps[0]) / delta_t),
        )  # define new timebase
        U_time = np.zeros(shape=[len(sol_timebase), *U_0.shape])
        U_time[0] = U_0

        # step-forward in time
        print_count = 0.0

        for i, t in enumerate(sol_timebase):
            U_time[i] = self.forward_time_step(
                U_time[i],
                delta_t,
                self.A,
                self.f.reshape(-1),
                k,
                method="forward euler",
            )

            if i / len(sol_timebase) >= print_count:
                U_plot = np.zeros(shape=(self.m + 2, self.m + 2))
                U_plot[1:-1, 1:-1] = U_time[i].reshape(
                    int(np.sqrt(U_time.shape[-1])), -1
                )
                print_count += 0.25
                self.plot_solution(
                    U_plot, addendum="t=%.2fsec" % t, show=False
                )

        U_t = np.zeros(shape=(U_time.shape[0], self.m + 2, self.m + 2))
        U_t[:, 1:-1, 1:-1] = U_time.reshape(
            U_time.shape[0], int(np.sqrt(U_time.shape[-1])), -1
        )
        return U_t

    def solve(
        self, timesteps: tuple = (0, 1), method: str = "static_solve"
    ) -> np.ndarray:
        """
        Solve system based on desired mode.
        """

        if method.lower() == "static_solve":
            U_t = self.static_solve()

        elif method.lower() == "time_evolve":
            U_t = self.time_evolve(timesteps)
        else:
            print("Did not receive a valid solution method.")
            return ()

        return U_t
