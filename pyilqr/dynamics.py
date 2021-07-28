import numpy as np

from abc import ABC, abstractmethod
from dataclasses import dataclass

from pyilqr.strategies import AbstractStrategy


class AbstractStageDynamics(ABC):
    @abstractmethod
    def dims(self):
        raise NotImplementedError

    @abstractmethod
    def next_state(self, x: np.ndarray, u: np.ndarray):
        raise NotImplementedError


@dataclass(frozen=True)
class LinearStageDynamics(AbstractStageDynamics):
    A: np.ndarray
    B: np.ndarray

    @property
    def dims(self):
        return self.B.shape

    def next_state(self, x: np.ndarray, u: np.ndarray):
        return self.A @ x + self.B @ u


class AbstractDynamics(ABC):
    @abstractmethod
    def next_state(self, x: np.ndarray, u: np.ndarray, k: int):
        raise NotImplementedError

    def rollout(self, x0: np.ndarray, strategy: AbstractStrategy, horizon: int):
        """
        Simulates dynamics forward in time by choosing controls according to
        `stage_strategies` starting from initial state `x0`.
        """
        trajectory = [x0]
        inputs = []

        for k in range(horizon):
            x = trajectory[-1]
            u = strategy.control_input(x, k)
            trajectory.append(self.next_state(x, u, k))
            inputs.append(u)

        return trajectory, inputs


@dataclass
class TimeVaryingDynamics(AbstractDynamics):
    stage_dynamics: list[AbstractStageDynamics]

    @property
    def dims(self):
        return self.stage_dynamics[0].dims

    @property
    def horizon(self):
        return len(self.stage_dynamics)

    def next_state(self, x: np.ndarray, u: np.ndarray, k: int):
        return self.stage_dynamics[k].next_state(x, u)

    def rollout(self, x0: np.ndarray, strategy: AbstractStrategy):
        return super().rollout(x0, strategy, self.horizon)


@dataclass
class LinearDynamics(TimeVaryingDynamics):
    stage_dynamics: list[LinearStageDynamics]

    def A(self, k):
        return self.stage_dynamics[k].A

    def B(self, k):
        return self.stage_dynamics[k].B
