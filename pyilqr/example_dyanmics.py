import numpy as np
import matplotlib.axes

from dataclasses import dataclass
from pyilqr.dynamics import AbstractSampledDynamics


@dataclass(frozen=True)
class UnicycleDynamics(AbstractSampledDynamics):
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
        return np.array(
            [
                v * np.cos(phi),
                v * np.sin(phi),
                dphi,
                dv,
            ]
        )

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

    def render_state(self, ax: matplotlib.axes.Axes, x: np.ndarray):
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
