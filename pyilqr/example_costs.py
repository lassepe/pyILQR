import numpy as np

from dataclasses import dataclass
from pyilqr.costs import AbstractCost


@dataclass
class TrackingCost(AbstractCost):
    R: np.ndarray
    Q: np.ndarray
    x_target: np.ndarray

    def __call__(self, x, u):
        ex = x - self.x_target
        return 0.5 * ex.T @ self.Q @ ex + u.T @ self.R @ u

    def stage_state_hessian(self, x):
        return self.Q

    def stage_state_gradient(self, x):
        return self.Q @ (x - self.x_target)

    def stage_input_hessian(self, u):
        return self.R

    def stage_input_gradient(self, u):
        return self.R @ u
