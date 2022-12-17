import typing
from abc import abstractmethod
import numpy as np


class ArrayModifier(typing.Protocol):
    """

    THIS DOCSTRING NEEDS UPDATING

    A class that takes the dimensions of a 2D array as inputs,
    plus other params, and contains a method called "solve" that
    returns a 3 dimensional numpy array (m x n x t).

    Args:
    - m: y-dimension of array to be perturbed (number of rows)
    - n: x-dimension of array to be perturbed (number of columns)
    - L_m: physical length of y-dimension (units not important,
        only used to define the meaning of source_radius relative
        to the array)
    - L_n: physical length of x-dimension (units not important, see
        above)
    - source_indices: list containing the coordinates of the numpy array
        to start the perturbation at. first coordinate is for rows (length m),
        second coordinate is for columns (length n). *ZERO INDEXED*
    - source_type: string specifying the type of perturbation to use.
    - solve_type: string specifying the solve method (e.g. static or
        time-dependent)
    - timesteps: list of length t specifying the timesteps to run the
        array modification for. first element must be zero and refers to
        the initial state.
    - source_radius: float representing radius of the source.
    - source_strength: float representing strength of the source.

    Methods:
    - solve: returns a three-dimensional np.ndarray with dimensions
        m x n x t, where t is the length of the "timesteps" list.
        the array at t = 0 should be identical to array_init,
        and the other timesteps should represent the array state
        at each time step.

    """

    m: int
    n: int
    L_m: float
    L_n: float
    f_type: str
    source_center: list
    radius: float
    source_strength: float
    source_center_basis: float

    @abstractmethod
    def solve(self, timesteps: list, method: str) -> np.ndarray:
        ...
