import numpy as np

from abc import ABC, abstractmethod
from dataclasses import dataclass
from math import factorial

from pyilqr.strategies import AbstractStrategy


class AbstractStageDynamics(ABC):
    @abstractmethod
    def dims(self):
        pass

    @abstractmethod
    def next_state(self, x: np.ndarray, u: np.ndarray):
        pass


@dataclass(frozen=True)
class LinearStageDynamics(AbstractStageDynamics):
    A: np.ndarray
    B: np.ndarray

    @property
    def dims(self):
        return self.B.shape

    def next_state(self, x: np.ndarray, u: np.ndarray):
        return self.A @ x + self.B @ u


@dataclass
class AbstractDiscreteDynamics(ABC):
    @abstractmethod
    def next_state(self, x: np.ndarray, u: np.ndarray, t: int):
        pass

    def linearized_along_trajectory(
        self, x_op: list[np.ndarray], u_op: list[np.ndarray]
    ):
        return LinearDynamics(
            [self.linearized_discrete(x, u) for (x, u) in zip(x_op, u_op)]
        )

    @abstractmethod
    def linearized_discrete(
        self, x: np.ndarray, u_op: np.ndarray
    ) -> "LinearStageDynamics":
        pass

    def rollout(self, x0: np.ndarray, strategy: AbstractStrategy, horizon: int):
        """
        Simulates dynamics forward in time by choosing controls according to
        `stage_strategies` starting from initial state `x0`.
        """
        xs = [x0]
        us = []

        for t in range(horizon):
            x = xs[-1]
            u = strategy.control_input(x, t)
            xs.append(self.next_state(x, u, t))
            us.append(u)

        return xs, us


@dataclass
class AbstractSampledDynamics(AbstractDiscreteDynamics):
    dt: float = 0.1

    @abstractmethod
    def dx(self, x: np.ndarray, u: np.ndarray, t: float) -> np.ndarray:
        pass

    @abstractmethod
    def linearized_continuous(
        self, x: np.ndarray, u: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        pass

    def next_state(
        self, x: np.ndarray, u: np.ndarray, t: int, method: str = "ForwardEuler"
    ) -> np.ndarray:

        if method == "ForwardEuler":
            return x + self.dt * self.dx(x, u, (t * self.dt))
        else:
            raise NotImplementedError

    def linearized_discrete(
        self, x: np.ndarray, u: np.ndarray, accuracy: int = 1
    ) -> LinearStageDynamics:
        """
        Linearizes the system about the operating point (`x`, `u`). Accuracy parameter determines
        the number of terms of the matrix exponential series used to approximate the discretization.
        For `accuracy = 1` this recovers the forward euler discretization. See
        https://en.wikipedia.org/wiki/Discretization for more details.
        """
        A, B = self.linearized_continuous(x, u)

        C = sum(
            1 / factorial(k) * np.linalg.matrix_power(A, k - 1) * self.dt ** k
            for k in range(1, accuracy + 1)
        )
        Ad = np.eye(x.size) + C @ A
        Bd = C @ B
        return LinearStageDynamics(Ad, Bd)


@dataclass
class LinearDynamics(AbstractDiscreteDynamics):
    stage_dynamics: list[LinearStageDynamics]

    @property
    def dims(self):
        return self.stage_dynamics[0].dims

    @property
    def horizon(self):
        return len(self.stage_dynamics)

    def linearized_discrete(self, x, u):
        raise NotImplementedError

    def next_state(self, x: np.ndarray, u: np.ndarray, t: int):
        return self.stage_dynamics[t].next_state(x, u)

    def rollout(self, x0: np.ndarray, strategy: AbstractStrategy):
        return super().rollout(x0, strategy, self.horizon)

    def A(self, t):
        return self.stage_dynamics[t].A

    def B(self, t):
        return self.stage_dynamics[t].B
