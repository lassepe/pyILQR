import numpy as np
import math

from pyilqr.dynamics import LinearDiscreteDynamics
from pyilqr.costs import QuadraticCost
from pyilqr.ocp import LQRProblem
from pyilqr.lqr import LQRSolver


def setup_double_integrator_ocp(horizon=100, dt=0.20, x_target=np.array([1, 0])):
    # a simple double integrator
    A = np.array([[1, dt], [0, 1]])
    B = np.array([[0, dt]]).T
    dynamics = [LinearDiscreteDynamics(0.1, A, B) for _ in range(horizon)]

    Q = np.eye(2)
    l = -Q @ x_target

    R = np.eye(1)
    r = np.zeros(1)

    state_cost = [QuadraticCost(Q, l) for _ in range(horizon + 1)]
    input_cost = [QuadraticCost(R, r) for _ in range(horizon)]

    return LQRProblem(dynamics, state_cost, input_cost), x_target


def test_solve_lqr(tol=1e-5):
    ocp, x_target = setup_double_integrator_ocp()
    solver = LQRSolver(ocp)
    strategy, expected_decrease = solver.solve()
    assert len(strategy.stage_strategies) == ocp.horizon
    assert expected_decrease >= 0

    x0 = np.zeros(2)
    trajectory, inputs, _ = ocp.dynamics[0].rollout(x0, strategy, ocp.horizon)
    assert len(trajectory) == ocp.horizon + 1
    assert len(inputs) == ocp.horizon
    assert math.isclose(np.linalg.norm(trajectory[-1] - x_target), 0, abs_tol=tol)
    assert math.isclose(np.linalg.norm(inputs[-1]), 0, abs_tol=tol)

    x0 = x_target
    trajectory, inputs, _ = ocp.dynamics[0].rollout(x0, strategy, ocp.horizon)
    assert math.isclose(np.linalg.norm(inputs[0]), 0, abs_tol=tol)
    assert math.isclose(np.linalg.norm(trajectory[-1] - x_target), 0, abs_tol=tol)
