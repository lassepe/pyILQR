from dataclasses import dataclass

from pyilqr.dynamics import AbstractDynamics, LinearDiscreteDynamics
from pyilqr.costs import AbstractCost, QuadraticCost
from typing import Sequence


@dataclass
class OptimalControlProblem:
    dynamics: AbstractDynamics
    state_cost: AbstractCost
    input_cost: AbstractCost
    horizon: int


@dataclass
class LQRProblem:
    def __post_init__(self):
        if not (self.dynamics or self.state_cost or self.input_cost):
            return

        if not (len(self.dynamics) == len(self.input_cost) == len(self.state_cost) - 1):
            raise ValueError(
                f"""
            Dynamics and costs must have matching horizon length.
            len(self.dynamics): {len(self.dynamics)},
            len(self.state_cost): {len(self.state_cost)},
            len(self.input_cost): {len(self.input_cost)},
            """
            )

    dynamics: Sequence[LinearDiscreteDynamics]
    state_cost: Sequence[QuadraticCost]
    input_cost: Sequence[QuadraticCost]

    @property
    def horizon(self):
        return len(self.input_cost)
