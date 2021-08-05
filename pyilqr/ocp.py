from dataclasses import dataclass

from pyilqr.dynamics import AbstractDiscreteDynamics, LinearDynamics
from pyilqr.costs import AbstractCost, QuadraticCost

@dataclass
class OptimalControlProblem:
    dynamics: AbstractDiscreteDynamics
    cost: AbstractCost
    horizon: int

@dataclass
class LQRProblem(OptimalControlProblem):
    dynamics: LinearDynamics
    cost: QuadraticCost
