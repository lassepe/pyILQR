from dataclasses import dataclass

from pyilqr.dynamics import AbstractDiscreteDynamics, LinearDynamics
from pyilqr.costs import AbstractCost, QuadraticCost
from typing import Sequence


@dataclass
class OptimalControlProblem:
    dynamics: AbstractDiscreteDynamics
    state_cost: AbstractCost
    input_cost: AbstractCost
    horizon: int


@dataclass
class LQRProblem:
    dynamics: LinearDynamics
    state_cost: Sequence[QuadraticCost]
    input_cost: Sequence[QuadraticCost]

    @property
    def horizon(self):
        return len(self.input_cost)
