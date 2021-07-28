import numpy as np
import math

from pyilqr.dynamics import LinearDynamics, LinearStageDynamics
from pyilqr.costs import QuadraticCost, QuadraticCostPrimitive
from pyilqr.lqr import solve_lqr


def test_solve_lqr():
    horizon = 100
    # a simple double integrator
    dt = 0.5
    A = np.array([[1, dt], [0, 1]])
    B = np.array([[0, dt]]).T
    dynamics = LinearDynamics([LinearStageDynamics(A, B) for _ in range(horizon)])

    x_target = np.array([1, 0])

    Q = np.eye(2)
    l = -Q @ x_target

    R = np.eye(1)
    r = np.zeros(1)
    costs = QuadraticCost(
        state_cost=[QuadraticCostPrimitive(Q, l) for _ in range(horizon + 1)],
        input_cost=[QuadraticCostPrimitive(R, r) for _ in range(horizon)],
    )
    strategy = solve_lqr(dynamics, costs)
    assert len(strategy.stage_strategies) == dynamics.horizon

    x0 = np.zeros(2)
    trajectory, inputs = dynamics.rollout(x0, strategy)
    assert len(trajectory) == dynamics.horizon + 1
    assert len(inputs) == dynamics.horizon
    assert math.isclose(np.linalg.norm(trajectory[-1] - x_target), 0, abs_tol=1e-5)
    assert math.isclose(np.linalg.norm(inputs[-1]), 0, abs_tol=1e-5)
