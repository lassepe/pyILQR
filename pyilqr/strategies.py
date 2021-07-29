import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable


class AbstractStrategy(ABC):
    @abstractmethod
    def control_input(self, x: np.ndarray, k: int):
        pass


@dataclass(frozen=True)
class AffineStageStrategy:
    P: np.ndarray
    a: np.ndarray

    def control_input(self, x: np.ndarray):
        return -self.P @ x - self.a


@dataclass
class AffineStrategy(AbstractStrategy):
    stage_strategies: list[AffineStageStrategy]

    def control_input(self, x: np.ndarray, k: int):
        return self.stage_strategies[k].control_input(x)


@dataclass
class FunctionStrategy(AbstractStrategy):
    controller: Callable[[np.ndarray, int], np.ndarray]
    def control_input(self, x: np.ndarray, k: int):
        return self.controller(x, k)
