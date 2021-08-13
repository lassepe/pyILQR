import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable


class AbstractStrategy(ABC):
    @abstractmethod
    def control_input(self, x: np.ndarray, t: int) -> np.ndarray:
        pass


@dataclass(frozen=True)
class AffineStageStrategy:
    P: np.ndarray
    a: np.ndarray

    def control_input(self, x: np.ndarray):
        return -self.P @ x - self.a


@dataclass(frozen=True)
class AffineStrategy(AbstractStrategy):
    stage_strategies: list[AffineStageStrategy]

    def control_input(self, x: np.ndarray, t: int):
        return self.stage_strategies[t].control_input(x)


@dataclass(frozen=True)
class FunctionStrategy(AbstractStrategy):
    controller: Callable[[np.ndarray, int], np.ndarray]

    def control_input(self, x: np.ndarray, t: int):
        return self.controller(x, t)


@dataclass(frozen=True)
class OpenLoopStrategy(AbstractStrategy):
    inputs: np.ndarray

    def control_input(self, x: np.ndarray, t: int):
        return self.inputs[t]
