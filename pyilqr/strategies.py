import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Callable, List, Tuple, Union


class AbstractStrategy(ABC):
    @abstractmethod
    def control_input(
        self, x: np.ndarray, t: int
    ) -> Tuple[np.ndarray, Union[Dict, None]]:
        pass


@dataclass(frozen=True)
class AffineStageStrategy:
    P: np.ndarray
    a: np.ndarray

    def control_input(self, x: np.ndarray):
        return -self.P @ x - self.a, None


@dataclass(frozen=True)
class AffineStrategy(AbstractStrategy):
    stage_strategies: List[AffineStageStrategy]

    def control_input(self, x: np.ndarray, t: int):
        return self.stage_strategies[t].control_input(x)


@dataclass(frozen=True)
class FunctionStrategy(AbstractStrategy):
    controller: Callable[[np.ndarray, int], np.ndarray]

    def control_input(self, x: np.ndarray, t: int):
        return self.controller(x, t), None


@dataclass(frozen=True)
class OpenLoopStrategy(AbstractStrategy):
    inputs: np.ndarray

    def control_input(self, x: np.ndarray, t: int):
        return self.inputs[t], None
