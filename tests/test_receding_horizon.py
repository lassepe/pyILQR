import matplotlib.pyplot as plt
import numpy as np
import time

from pyilqr.costs import CompositeCost, QuadraticCost
from pyilqr.example_costs import PolylineTrackingCost, SetpointTrackingCost, Polyline
from pyilqr.example_dyanmics import UnicycleDynamics
from pyilqr.ocp import OptimalControlProblem
from pyilqr.receding_horizon import RecedingHorizonStrategy, ILQRSolver


def test_receding_horizon_parking():
    dynamics = UnicycleDynamics()
    simulation_horizon = 100
    prediction_horizon = 20
    x0 = np.array([0, 0, 0, 0.5])
    x_target = np.array([2, 1, 0, 0])

    state_cost = SetpointTrackingCost(np.eye(4), x_target)
    input_cost = QuadraticCost(np.eye(2), np.zeros(2))

    # setup the per-horizon solver:
    per_horizon_ocp = OptimalControlProblem(
        dynamics, state_cost, input_cost, prediction_horizon
    )
    inner_solver = ILQRSolver(per_horizon_ocp)
    receding_horizon_strategy = RecedingHorizonStrategy(inner_solver)
    xs, us = dynamics.rollout(x0, receding_horizon_strategy, simulation_horizon)
    return xs, us, dynamics
    # TODO: actually sanity-check the results


def test_receding_horizon_path_following():
    dynamics = UnicycleDynamics()
    simulation_horizon = 200
    prediction_horizon = 10
    x0 = np.array([0, 0, 0, 0.5])

    state_cost = CompositeCost(
        [
            PolylineTrackingCost(
                Polyline(np.array([[0, 0], [1, 1], [3, 1], [3, -1], [0, -3]])), 1
            ),
            SetpointTrackingCost(np.diag([0, 0, 0, 1]), np.array([0, 0, 0, 1])),
        ]
    )

    input_cost = QuadraticCost(0.1 * np.eye(2), np.zeros(2))

    per_horizon_ocp = OptimalControlProblem(
        dynamics, state_cost, input_cost, prediction_horizon
    )
    inner_solver = ILQRSolver(per_horizon_ocp)
    receding_horizon_strategy = RecedingHorizonStrategy(inner_solver)
    xs, us = dynamics.rollout(x0, receding_horizon_strategy, simulation_horizon)
    return xs, us, dynamics
    # TODO: actually sanity-check the results


def visual_sanity_check():
    xs, us, dynamics = test_receding_horizon_path_following()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.show(block=False)
    dt_target = dynamics.dt

    for x in xs:
        last_wall_time = time.monotonic()

        ax.clear()
        dynamics.render_state(ax, x)
        ax.plot(xs[:, 0], xs[:, 1])
        fig.canvas.draw()
        fig.canvas.flush_events()
        dt_measured = time.monotonic() - last_wall_time
        time_to_sleep = max(0, dt_target - dt_measured)
        time.sleep(time_to_sleep)


if __name__ == "__main__":
    visual_sanity_check()
