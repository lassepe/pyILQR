from dataclasses import dataclass
import numpy as np

@dataclass
class QuadraticCostPrimitive:
    """
    A simple wrapper for a quadratic cost primitive that maps a vector `x` to a scalar cost:
    x.T * Q * x + 2*x.T * l.
    """

    Q: np.ndarray
    l: np.ndarray

    def __call__(self, x: np.ndarray):
        return 0.5 * x.T @ self.Q @ x + self.l.T @ x

@dataclass
class QuadraticCost:
    state_cost: list[QuadraticCostPrimitive]
    input_cost: list[QuadraticCostPrimitive]

    def Q(self, k: int):
        return self.state_cost[k].Q

    def l(self, k: int):
        return self.state_cost[k].l

    def R(self, k: int):
        return self.input_cost[k].Q

    def r(self, k: int):
        return self.input_cost[k].l

    def __call__(self, x: np.ndarray, u: np.ndarray, k:int):
        return self.state_cost[k](x) + self.input_cost[k](u)
