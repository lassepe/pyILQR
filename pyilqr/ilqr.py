import numpy as np

from dataclasses import dataclass, field
from pyilqr.ocp import OptimalControlProblem, LQRProblem
from pyilqr.strategies import AbstractStrategy, AffineStrategy
from pyilqr.lqr import LQRSolver
from typing import Any, Tuple
from copy import copy


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
        last_xop, last_uop = self.ocp.dynamics.rollout(
            x0, initial_strategy, self.ocp.horizon
        )

        last_cost = self.ocp.state_cost.trajectory_cost(
            last_xop
        ) + self.ocp.input_cost.trajectory_cost(last_uop)

        for it in range(self.max_iterations):
            if has_converged:
                break
            self._lqr_solver.ocp.state_cost = (
                self.ocp.state_cost.quadratisized_along_trajectory(last_xop)
            )
            self._lqr_solver.ocp.input_cost = (
                self.ocp.input_cost.quadratisized_along_trajectory(last_uop)
            )
            self._lqr_solver.ocp.dynamics = (
                self.ocp.dynamics.linearized_along_trajectory(last_xop, last_uop)
            )

            local_strategy = self._lqr_solver.solve()
            (
                last_xop,
                last_uop,
                updated_cost,
                found_decent_step,
            ) = self._update_operating_point(
                last_xop,
                last_uop,
                last_cost,
                local_strategy,
                self.n_backtracking_steps,
            )

            if self.verbose:
                print("Cost Delta:", updated_cost - last_cost)
            last_cost = updated_cost
            # This could be replaced with a more accurate convergence criterion.
            has_converged = not found_decent_step

        return last_xop, last_uop, has_converged

    def _update_operating_point(
        self,
        last_xop,
        last_uop,
        last_cost: float,
        local_strategy: AffineStrategy,
        n_backtracking_steps: int,
        step_scale: float = 0.5,
    ):

        step_size = 1
        updated_cost = float("inf")
        updated_xop, updated_uop = None, None
        found_decent_step = False

        for _ in range(n_backtracking_steps):
            updated_xop, updated_uop = self._local_rollout(
                last_xop, last_uop, self.ocp.dynamics, local_strategy, step_size
            )
            # TODO: technically, we would want to have some *sufficient* decrease.
            updated_cost = self.ocp.state_cost.trajectory_cost(
                updated_xop
            ) + self.ocp.input_cost.trajectory_cost(updated_uop)
            if updated_cost < last_cost:
                found_decent_step = True
                break
            step_size *= step_scale

        if not found_decent_step:
            updated_xop, updated_uop = last_xop, last_uop
            updated_cost = last_cost

        return updated_xop, updated_uop, updated_cost, found_decent_step

    def _local_rollout(
        self,
        last_xop,
        last_uop,
        nonlinear_dynamics,
        local_strategy: AffineStrategy,
        step_size,
    ):
        xs = copy(last_xop)
        us = copy(last_uop)

        for t in range(len(last_uop)):
            x = xs[t]
            du = local_strategy.control_input(x - last_xop[t], t)
            u = last_uop[t] + step_size * du
            xs[t + 1] = nonlinear_dynamics.next_state(x, u, t)
            us[t] = u

        return xs, us
