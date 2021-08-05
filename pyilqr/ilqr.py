import numpy as np

from dataclasses import dataclass, field
from pyilqr.ocp import OptimalControlProblem, LQRProblem
from pyilqr.strategies import AbstractStrategy, AffineStrategy
from pyilqr.lqr import LQRSolver
from typing import Any, Tuple


@dataclass
class ILQRSolver:
    ocp: OptimalControlProblem
    max_iterations: int = 100
    n_backtracking_steps = 5
    verbose = False
    _lqr_solver: LQRSolver = field(init=False)

    def __post_init__(self):
        self._lqr_solver = LQRSolver(
            LQRProblem(None, None, self.ocp.horizon)  # type: ignore
        )

    def solve(
        self,
        x0: np.ndarray,
        initial_strategy: AbstractStrategy,
    ) -> Tuple:

        has_converged = False
        last_operating_point = self.ocp.dynamics.rollout(
            x0, initial_strategy, self.ocp.horizon
        )
        last_cost = self.ocp.cost.trajectory_cost(*last_operating_point)

        for it in range(self.max_iterations):
            if has_converged:
                break
            self._lqr_solver.ocp.cost = self.ocp.cost.quadratisized_along_trajectory(
                *last_operating_point
            )
            self._lqr_solver.ocp.dynamics = (
                self.ocp.dynamics.linearized_along_trajectory(*last_operating_point)
            )

            local_strategy = self._lqr_solver.solve()
            (
                last_operating_point,
                updated_cost,
                found_decent_step,
            ) = self._update_operating_point(
                last_operating_point,
                last_cost,
                local_strategy,
                self.n_backtracking_steps,
            )

            if self.verbose:
                print("Cost Delta:", updated_cost - last_cost)
            last_cost = updated_cost
            # This could be replaced with a more accurate convergence criterion.
            has_converged = not found_decent_step

        return last_operating_point, has_converged

    def _update_operating_point(
        self,
        last_operating_point,
        last_cost: float,
        local_strategy: AffineStrategy,
        n_backtracking_steps: int,
        step_scale: float = 0.5,
    ):

        step_size = 1
        updated_cost = float("inf")
        updated_operating_point = None
        found_decent_step = False

        for _ in range(n_backtracking_steps):
            updated_operating_point = self._local_rollout(
                last_operating_point, self.ocp.dynamics, local_strategy, step_size
            )
            # TODO: technically, we would want to have some *sufficient* decrease.
            updated_cost = self.ocp.cost.trajectory_cost(*updated_operating_point)
            if updated_cost < last_cost:
                found_decent_step = True
                break
            step_size *= step_scale

        if not found_decent_step:
            updated_operating_point = last_operating_point
            updated_cost = last_cost

        return updated_operating_point, updated_cost, found_decent_step

    def _local_rollout(
        self,
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
