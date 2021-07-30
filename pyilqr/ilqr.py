import numpy as np
import copy

from pyilqr.strategies import AbstractStrategy
from pyilqr.dynamics import AbstractDiscreteDynamics
from pyilqr.costs import AbstractCosts
from pyilqr.lqr import solve_lqr


def _update_operating_point(
    current_operating_point, last_operating_point, local_strategy
) -> bool:
    return True


def solve_ilqr(
    dynamics: AbstractDiscreteDynamics,
    costs: AbstractCosts,
    initial_strategy: AbstractStrategy,
    x0: np.ndarray,
    horizon: int,
    max_iteartions: int = 5,
):

    last_operating_point = dynamics.rollout(x0, initial_strategy, horizon)
    current_operating_point = dynamics.rollout(x0, initial_strategy, horizon)

    for _ in range(max_iteartions):
        # TODO: could probably avoid some allocations here
        linear_dynamics = dynamics.linearized_along_trajectory(*current_operating_point)
        quadratic_costs = costs.quadratisized_along_trajectory(*current_operating_point)
        local_strategy = solve_lqr(linear_dynamics, quadratic_costs)
        has_converged = _update_operating_point(
            current_operating_point, last_operating_point, local_strategy
        )

        if has_converged:
            break

    return current_operating_point
