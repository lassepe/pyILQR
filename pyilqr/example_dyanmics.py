import numpy as np

from dataclasses import dataclass
from pyilqr.dynamics import AbstractSampledDynamics


@dataclass
class UnicycleDynamics(AbstractSampledDynamics):
    def dx(self, x: np.ndarray, u: np.ndarray, t: float = 0):
        # state layout:
        px, py, phi, v = x
        # input layout:
        dphi, dv = u
        return np.array([v * np.cos(phi), v * np.sin(phi), dphi, dv])

    def linearized_continuous(self, x: np.ndarray, u: np.ndarray):
        px, py, phi, v = x
        sPhi = np.sin(phi)
        cPhi = np.cos(phi)
        A = np.array(
            [[0, 0, v * sPhi, cPhi], [0, 0, v * cPhi, sPhi], [0, 0, 0, 0], [0, 0, 0, 0]]
        )
        B = np.array([[0, 0], [0, 0], [1, 0], [0, 1]])
        return (A, B)
