import numpy as np

from pyilqr.dynamics import LinearDynamics, LinearStageDynamics
from pyilqr.costs import QuadraticCost, QuadraticCostPrimitive
from pyilqr.lqr import solve_lqr

import matplotlib.pyplot as plt

def main():
    horizon = 100
    # a simple double integrator
    dt = 0.5
    A = np.array([[1, dt], [0, 1]])
    B = np.array([[0, dt]]).T
    dynamics = LinearDynamics([LinearStageDynamics(A, B) for _ in range(horizon)])

    Q = np.eye(2)
    l = np.array([-1, -1])

    R = np.eye(1)
    r = np.zeros(1)
    costs = QuadraticCost(
        state_cost=[QuadraticCostPrimitive(Q, l) for _ in range(horizon + 1)],
        input_cost=[QuadraticCostPrimitive(R, r) for _ in range(horizon)],
    )

    strategy = solve_lqr(dynamics, costs)

    x0 = np.array([1, 0])
    trajectory, inputs = dynamics.rollout(x0, strategy)

    print("len(traj): ", len(trajectory))
    print("len(inputs): ", len(trajectory))

    plt.plot(inputs)
    plt.show()


if __name__ == "__main__":
    main()
