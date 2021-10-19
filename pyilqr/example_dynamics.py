import numpy as np
import matplotlib.axes

from dataclasses import dataclass
from pyilqr.dynamics import AbstractDynamics


@dataclass(frozen=True)
class BicycleDynamics(AbstractDynamics):
    """
    The dynamics of a 4D Bicycle with state layout `x = px, py, phi, v`. Where
    - `px` is the position along the x-axis
    - `py` is the position along the y-axis
    - `phi` is the orientation of the vehicle in rad.
    - `v` is the velocity
    """

    # These come from system identification
    L: float = 2.69989371
    av: float = 5.93727029
    bv: float = -0.21911138

    @property
    def dims(self):
        return 4, 2

    def dx(self, x: np.ndarray, u: np.ndarray, t: float = 0):
        # state layout:
        px, py, phi, v = x
        # input layout:
        dphi, dv = u
        return np.array(
            [
                v * np.cos(phi),
                v * np.sin(phi),
                v * np.tan(dphi) / self.L,
                self.av * dv + self.bv * v,
            ]
        )

    def linearized_continuous(self, x: np.ndarray, u: np.ndarray):
        px, py, phi, v = x
        dphi, dv = u
        sPhi = np.sin(phi)
        cPhi = np.cos(phi)
        tDphi = np.tan(dphi)
        dtDphi = np.cos(dphi) ** (-2)
        A = np.array(
            [
                [0, 0, -v * sPhi, cPhi],
                [0, 0, v * cPhi, sPhi],
                [0, 0, 0, tDphi / self.L],
                [0, 0, 0, self.bv],
            ]
        )
        B = np.array([[0, 0], [0, 0], [v / self.L * dtDphi, 0], [0, self.av]])
        return (A, B)

    def visualize_state(self, ax: matplotlib.axes.Axes, x: np.ndarray):
        px, py, phi, v = x

        car_x_vert = [
            px + self.viz_length / 2 * np.cos(phi) - self.viz_width / 2 * np.sin(phi),
            px + self.viz_length / 2 * np.cos(phi) + self.viz_width / 2 * np.sin(phi),
            px - self.viz_length / 2 * np.cos(phi) + self.viz_width / 2 * np.sin(phi),
            px - self.viz_length / 2 * np.cos(phi) - self.viz_width / 2 * np.sin(phi),
        ]

        car_y_vert = [
            py + self.viz_width / 2 * np.cos(phi) + self.viz_length / 2 * np.sin(phi),
            py - self.viz_width / 2 * np.cos(phi) + self.viz_length / 2 * np.sin(phi),
            py - self.viz_width / 2 * np.cos(phi) - self.viz_length / 2 * np.sin(phi),
            py + self.viz_width / 2 * np.cos(phi) - self.viz_length / 2 * np.sin(phi),
        ]

        ax.fill(car_x_vert, car_y_vert)

@dataclass(frozen=True)
class UnicycleDynamics(AbstractDynamics):
    """
    The dynamics of a 4D unicycle with state layout `x = px, py, phi, v`. Where
    - `px` is the position along the x-axis
    - `py` is the position along the y-axis
    - `phi` is the orientation of the vehicle in rad.
    - `v` is the velocity
    """

    # These are just for visualization
    viz_length: float = 0.1
    viz_width: float = 0.05

    @property
    def dims(self):
        return 4, 2

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
            [
                [0, 0, -v * sPhi, cPhi],
                [0, 0, v * cPhi, sPhi],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ]
        )
        B = np.array([[0, 0], [0, 0], [1, 0], [0, 1]])
        return (A, B)

    def visualize_state(self, ax: matplotlib.axes.Axes, x: np.ndarray):
        px, py, phi, v = x

        car_x_vert = [
            px + self.viz_length / 2 * np.cos(phi) - self.viz_width / 2 * np.sin(phi),
            px + self.viz_length / 2 * np.cos(phi) + self.viz_width / 2 * np.sin(phi),
            px - self.viz_length / 2 * np.cos(phi) + self.viz_width / 2 * np.sin(phi),
            px - self.viz_length / 2 * np.cos(phi) - self.viz_width / 2 * np.sin(phi),
        ]

        car_y_vert = [
            py + self.viz_width / 2 * np.cos(phi) + self.viz_length / 2 * np.sin(phi),
            py - self.viz_width / 2 * np.cos(phi) + self.viz_length / 2 * np.sin(phi),
            py - self.viz_width / 2 * np.cos(phi) - self.viz_length / 2 * np.sin(phi),
            py + self.viz_width / 2 * np.cos(phi) - self.viz_length / 2 * np.sin(phi),
        ]

        ax.fill(car_x_vert, car_y_vert)
