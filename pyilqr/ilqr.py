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
        du = local_strategy.control_input(x, t)
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

    for _ in range(n_backtracking_steps):
        updated_operating_point = _local_rollout(
            last_operating_point, nonlinear_dynamics, local_strategy, step_size
        )
        # TODO: technically, we would want to have some *sufficient* decrease.
        updated_cost = cost_model.trajectory_cost(*updated_operating_point)
        print(updated_cost - last_cost)
        if updated_cost < last_cost:
            return updated_operating_point, True
        else:
            step_size *= step_scale

    return last_operating_point, False


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
):

    last_operating_point = dynamics.rollout(x0, initial_strategy, horizon)
    last_cost = cost_model.trajectory_cost(*last_operating_point)
    has_converged = False

    for _ in range(max_iterations):
        quadratic_costs = cost_model.quadratisized_along_trajectory(
            *last_operating_point
        )
        cost_gradient_norm = sum(
            (cs.l ** 2).sum() for cs in quadratic_costs.state_cost
        ) + sum((cs.l ** 2).sum() for cs in quadratic_costs.input_cost)
        print("cost_gradient_norm: ", cost_gradient_norm)
        # TODO: this convergence check is wrong since it does not account for the dynamics
        # constraints, this should rather check for something like the total derivative in u
        has_converged = cost_gradient_norm < gradient_tolerance
        if has_converged:
            break
        linear_dynamics = dynamics.linearized_along_trajectory(*last_operating_point)

        local_strategy = solve_lqr(linear_dynamics, quadratic_costs)
        last_operating_point, found_decent_direction = _update_operating_point(
            last_operating_point,
            cost_model,
            last_cost,
            dynamics,
            local_strategy,
            n_backtracking_steps,
        )

        if not found_decent_direction:
            break

    return last_operating_point, has_converged
