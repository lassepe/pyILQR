import numpy as np
import matplotlib.pyplot as plt

from pyilqr.example_costs import TrackingCost
from pyilqr.example_dyanmics import UnicycleDynamics
from pyilqr.ilqr import solve_ilqr
from pyilqr.strategies import FunctionStrategy


def test_ilqr():
    dynamics = UnicycleDynamics()
    x0 = np.zeros(4)
    R = np.eye(2)
    Q = np.eye(4)
    cost = TrackingCost(R, Q, x_target=np.array([1, 1, 0, 0]))

    initial_strategy = FunctionStrategy(lambda x, k: np.array([0, 1]))
    xs, us = solve_ilqr(
        dynamics, cost, initial_strategy, x0, horizon=100, max_iteartions=5
    )
    plt.plot([x[0] for x in xs], [x[1] for x in xs])
    plt.show()

if __name__ == '__main__':
    test_ilqr()
