from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np


class AbstractCosts(ABC):
    @abstractmethod
    def state_hessian(self, x) -> np.ndarray:
        pass

    @abstractmethod
    def state_gradient(self, u) -> np.ndarray:
        pass

    @abstractmethod
    def input_hessian(self, x) -> np.ndarray:
        pass

    @abstractmethod
    def input_gradient(self, u) -> np.ndarray:
        pass

    def _quadratisized(self, x, to_hessian, to_gradient) -> "QuadraticCostPrimitive":
        H = to_hessian(x)
        g = to_gradient(x)
        # TODO: think about scaling
        return QuadraticCostPrimitive(H, g)

    def quadratisized_along_trajectory(self, x_op, u_op) -> "QuadraticCost":
        state_cost = [
            self._quadratisized(x, self.state_hessian, self.state_gradient)
            for x in x_op
        ]
        input_cost = [
            self._quadratisized(u, self.input_hessian, self.input_gradient)
            for u in u_op
        ]
        return QuadraticCost(state_cost, input_cost)


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
class QuadraticCost(AbstractCosts):
    state_cost: list[QuadraticCostPrimitive]
    input_cost: list[QuadraticCostPrimitive]

    def state_hessian(self, x):
        raise NotImplementedError

    def state_gradient(self, x):
        raise NotImplementedError

    def input_hessian(self, x):
        raise NotImplementedError

    def input_gradient(self, x):
        raise NotImplementedError

    def Q(self, k: int):
        return self.state_cost[k].Q

    def l(self, k: int):
        return self.state_cost[k].l

    def R(self, k: int):
        return self.input_cost[k].Q

    def r(self, k: int):
        return self.input_cost[k].l

    def __call__(self, x: np.ndarray, u: np.ndarray, k: int):
        return self.state_cost[k](x) + self.input_cost[k](u)
