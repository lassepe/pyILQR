import numpy as np

from copy import copy
from pyilqr.strategies import AbstractStrategy, AffineStrategy
from pyilqr.dynamics import AbstractDiscreteDynamics
from pyilqr.costs import AbstractCost
from pyilqr.lqr import solve_lqr
from typing import Any, Tuple


def _local_rollout(
    last_operating_point,
    nonlinear_dynamics,
    local_strategy: AffineStrategy,
    step_size,
):
    x_op, u_op = last_operating_point
    horizon = len(u_op)

    xs = [x_op[0]]
    us = []

    for t in range(horizon):
        x = xs[-1]
        du = local_strategy.control_input(x - x_op[t], t)
        u = u_op[t] + step_size * du
        xs.append(nonlinear_dynamics.next_state(x, u, t))
        us.append(u)

    return xs, us


def _update_operating_point(
    last_operating_point,
    cost_model,
    last_cost,
    nonlinear_dynamics,
    local_strategy,
    n_backtracking_steps,
    step_scale=0.5,
):

    step_size = 1
    updated_cost = float("inf")
    updated_operating_point = None
    found_decent_step = False

    for _ in range(n_backtracking_steps):
        updated_operating_point = _local_rollout(
            last_operating_point, nonlinear_dynamics, local_strategy, step_size
        )
        # TODO: technically, we would want to have some *sufficient* decrease.
        updated_cost = cost_model.trajectory_cost(*updated_operating_point)
        if updated_cost < last_cost:
            found_decent_step = True
            break
        step_size *= step_scale

    if not found_decent_step:
        updated_operating_point = last_operating_point
        updated_cost = last_cost

    return updated_operating_point, updated_cost, found_decent_step


# TODO: could probably avoid some allocations here
def solve_ilqr(
    dynamics: AbstractDiscreteDynamics,
    cost_model: AbstractCost,
    initial_strategy: AbstractStrategy,
    x0: np.ndarray,
    horizon: int,
    max_iterations: int = 100,
    n_backtracking_steps=5,
    gradient_tolerance=1e-1,
    verbose=False,
) -> Tuple:

    last_operating_point = dynamics.rollout(x0, initial_strategy, horizon)
    last_cost = cost_model.trajectory_cost(*last_operating_point)
    has_converged = False

    for it in range(max_iterations):
        if has_converged:
            break
        quadratic_costs = cost_model.quadratisized_along_trajectory(
            *last_operating_point
        )
        linear_dynamics = dynamics.linearized_along_trajectory(*last_operating_point)

        local_strategy = solve_lqr(linear_dynamics, quadratic_costs)
        (
            last_operating_point,
            updated_cost,
            found_decent_step,
        ) = _update_operating_point(
            last_operating_point,
            cost_model,
            last_cost,
            dynamics,
            local_strategy,
            n_backtracking_steps,
        )

        if verbose:
            print("Cost Delta:", updated_cost - last_cost)
        last_cost = updated_cost
        # This could be replaced with a more accurate convergence criterion.
        has_converged = not found_decent_step

    return last_operating_point, has_converged
