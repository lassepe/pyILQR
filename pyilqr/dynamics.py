import numpy as np
import matplotlib.axes

from abc import ABC, abstractmethod
from dataclasses import dataclass
from math import factorial
from typing import Sequence, Tuple

from pyilqr.strategies import AbstractStrategy


@dataclass(frozen=True)
class AbstractDynamics(ABC):
    dt: float

    def __post_init__(self):
        if self.dt <= 0:
            raise ValueError(
                "Pleas provide a positive sampling rate `self.dt` in order to facilitate discretization."
            )

    @property
    @abstractmethod
    def dims(self) -> Tuple[int, int]:
        pass

    def dx(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        raise NotImplementedError(
            "This system does not implement contious time derivatives. Either implement `next_state` directly for this system or implement state derivatives in continuous time via `dx`."
        )

    def next_state(
        self,
        x: np.ndarray,
        u: np.ndarray,
        method: str = "ForwardEuler",
    ) -> np.ndarray:

        if method == "ForwardEuler":
            return x + self.dt * self.dx(x, u)
        else:
            raise NotImplementedError

    def linearized_continuous(
        self, x: np.ndarray, u: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError(
            "This system does not provide continous time jacobians. Either implement `linearized_discrete` directly or provide the concontinuous time jacobians via `linearized_continuous`."
        )

    def linearized_discrete(
        self, x: np.ndarray, u: np.ndarray, accuracy: int = 1
    ) -> "LinearDiscreteDynamics":
        """
        Returns the discrete linearization of the system about the operating point `(x, u)`.
        Accuracy parameter determines the number of terms of the matrix exponential series used to
        approximate the discretization.  For `accuracy = 1` this recovers the forward euler
        discretization. See https://en.wikipedia.org/wiki/Discretization for more details.
        """
        A, B = self.linearized_continuous(x, u)

        C = sum(
            1 / factorial(k) * np.linalg.matrix_power(A, k - 1) * self.dt ** k
            for k in range(1, accuracy + 1)
        )
        Ad = np.eye(x.size) + C @ A
        Bd = C @ B
        return LinearDiscreteDynamics(self.dt, Ad, Bd)

    def visualize_state(self, ax: matplotlib.axes.Axes, x: np.ndarray):
        """
        Render the state `x` of the system on a given axis
        """
        raise NotImplementedError

    def rollout(self, x0: np.ndarray, strategy: AbstractStrategy, horizon: int):
        """
        Simulates the dynamical system forward in time for `horizon` steps by choosing controls
        according to `strategy` starting from initial state `x0`.
        """
        n_states, n_inputs = self.dims
        xs = np.zeros((horizon + 1, n_states))
        xs[0] = x0
        us = np.zeros((horizon, n_inputs))
        infos = []
        for t in range(horizon):
            x = xs[t]
            u, info = strategy.control_input(x, t)
            us[t] = u
            infos.append(info)
            xs[t + 1] = self.next_state(x, u)

        return xs, us, infos

    def linearized_along_trajectory(
        self, x_op: np.ndarray, u_op: np.ndarray
    ) -> Sequence["LinearDiscreteDynamics"]:
        return [self.linearized_discrete(x, u) for (x, u) in zip(x_op, u_op)]


@dataclass(frozen=True)
class LinearDiscreteDynamics(AbstractDynamics):
    A: np.ndarray
    B: np.ndarray

    @property
    def dims(self):
        return self.B.shape

    def next_state(self, x: np.ndarray, u: np.ndarray):
        return self.A @ x + self.B @ u
