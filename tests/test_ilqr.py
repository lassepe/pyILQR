import numpy as np
import matplotlib.pyplot as plt

from pyilqr.example_costs import TrackingCost
from pyilqr.example_dyanmics import UnicycleDynamics
from pyilqr.ilqr import solve_ilqr
from pyilqr.strategies import FunctionStrategy


def test_ilqr():
    dynamics = UnicycleDynamics()
    horizon = 100
    x0 = np.array([0, 0, -0.3, 0.1])
    R = np.eye(2)
    Q = np.eye(4)
    cost = TrackingCost(R, Q, x_target=np.array([1, 1, 0, 0]))

    initial_strategy = FunctionStrategy(lambda x, t: np.array([0, 0]))
    initial_operating_point = dynamics.rollout(x0, initial_strategy, horizon)
    initial_cost = cost.trajectory_cost(*initial_operating_point)

    (xs, us), has_converged = solve_ilqr(dynamics, cost, initial_strategy, x0, horizon)

    assert has_converged
    assert cost.trajectory_cost(xs, us) < initial_cost


if __name__ == "__main__":
    test_ilqr()
