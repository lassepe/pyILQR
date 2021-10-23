import numpy as np
import math

from pyilqr.example_dynamics import UnicycleDynamics, BicycleDynamics
from pyilqr.strategies import FunctionStrategy

import matplotlib.pyplot as plt


def test_unicycle(tol=1e-5):
    dyn = UnicycleDynamics(0.2)
    # properties
    assert dyn.dt == 0.2
    # basic dynamics
    assert all(dyn.dx(x=np.zeros(4), u=np.zeros(2)) == np.zeros(4))
    # linearization
    x_op, u_op = np.random.rand(4), np.random.rand(2)
    dyn_discrete = dyn.linearized_discrete(x_op, u_op)
    # for zero deviation from the operating point the we should predict no deviation from the future
    # operating point (since the linearization only predicts the local deviation dynamics)
    assert all(dyn_discrete.next_state(np.zeros(4), np.zeros(2)) == 0)

    # simulation
    trajectory, *_ = dyn.rollout(
        x0=np.zeros(4), strategy=FunctionStrategy(lambda x, k: np.zeros(2)), horizon=10
    )
    assert all(math.isclose(np.linalg.norm(x), 0, abs_tol=tol) for x in trajectory)


def test_bicycle(tol=1e-5):
    dyn = BicycleDynamics(0.2)
    # properties
    assert dyn.dt == 0.2
    # basic dynamics
    assert all(dyn.dx(x=np.zeros(5), u=np.zeros(2)) == np.zeros(5))
    # linearization
    x_op, u_op = np.random.rand(5), np.random.rand(2)
    dyn_discrete = dyn.linearized_discrete(x_op, u_op)
    # for zero deviation from the operating point the we should predict no deviation from the future
    # operating point (since the linearization only predicts the local deviation dynamics)
    assert all(dyn_discrete.next_state(np.zeros(5), np.zeros(2)) == 0)

    # simulation
    trajectory, *_ = dyn.rollout(
        x0=np.zeros(5), strategy=FunctionStrategy(lambda x, k: np.zeros(2)), horizon=10
    )
    assert all(math.isclose(np.linalg.norm(x), 0, abs_tol=tol) for x in trajectory)


def visual_sanity_check_unicycle():
    dyn = UnicycleDynamics(0.1)

    print("We would expect the unicycyle to drive in a circle at constant velocity.")
    trajectory, *_ = dyn.rollout(
        x0=np.array([0, 0, 0, 2]),
        strategy=FunctionStrategy(lambda x, k: np.array([3, 0])),
        horizon=10,
    )
    plt.plot([x[0] for x in trajectory], [x[1] for x in trajectory])
    plt.gca().set_aspect("equal", adjustable="box")
    plt.show()


if __name__ == "__main__":
    visual_sanity_check_unicycle()
