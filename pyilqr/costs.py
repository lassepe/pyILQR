from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence
import numpy as np


class AbstractCost(ABC):
    @abstractmethod
    def cost(self, x: np.ndarray) -> float:
        pass

    @abstractmethod
    def gradient(self, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def hessian(self, x: np.ndarray) -> np.ndarray:
        pass

    def _quadratisized(self, x, to_hessian, to_gradient) -> "QuadraticCost":
        H = to_hessian(x)
        g = to_gradient(x)
        return QuadraticCost(H, g)

    def quadratisized_along_trajectory(
        self, x_op: Sequence[np.ndarray]
    ) -> Sequence["QuadraticCost"]:
        return [QuadraticCost(self.hessian(x), self.gradient(x)) for x in x_op]

    def trajectory_cost(self, xs) -> float:
        return sum(self.cost(x) for x in xs)


@dataclass
class QuadraticCost(AbstractCost):
    """
    A simple wrapper for a quadratic cost primitive that maps a vector `x` to a scalar cost:
    x.T * Q * x + 2*x.T * l.
    """

    Q: np.ndarray
    l: np.ndarray

    def cost(self, x: np.ndarray):
        return 0.5 * x.T @ self.Q @ x + self.l.T @ x

    def gradient(self, x: np.ndarray):
        return self.Q @ x + self.l

    def hessian(self, x: np.ndarray):
        return self.Q


@dataclass
class CompositeCost(AbstractCost):
    components: Sequence[AbstractCost]

    def cost(self, x: np.ndarray):
        return sum(c.cost(x) for c in self.components)

    def gradient(self, x: np.ndarray):
        return sum(c.gradient(x) for c in self.components)

    def hessian(self, x: np.ndarray):
        return sum(c.hessian(x) for c in self.components)
