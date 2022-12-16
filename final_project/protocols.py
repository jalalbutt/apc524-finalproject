import typing
from abc import abstractmethod
import numpy as np


class ArrayModifier(typing.Protocol):
    """
    A class that takes a 2 dimensional numpy array (m x n) as an input
    (plus other args), and contains a method called "solve" that
    returns a 3 dimensional numpy array (m x n x t).

    Args:
    - array_init: two-dimensional np.ndarray with dimensions m x n.
        represents the array that will be modified.
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

    array_init: np.ndarray
    source_indices: list
    source_type: str
    solve_type: str
    timesteps: list
    source_radius: float
    source_strength: float

    @abstractmethod
    def solve(self) -> np.ndarray:
        ...
