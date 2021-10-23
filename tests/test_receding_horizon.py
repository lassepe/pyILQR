import matplotlib.pyplot as plt
import numpy as np

from matplotlib.animation import FuncAnimation, writers
from pyilqr.costs import CompositeCost, QuadraticCost
from pyilqr.example_costs import PolylineTrackingCost, SetpointTrackingCost, Polyline
from pyilqr.example_dynamics import UnicycleDynamics, BicycleDynamics
from pyilqr.ocp import OptimalControlProblem
from pyilqr.receding_horizon import RecedingHorizonStrategy, ILQRSolver


def test_receding_horizon_parking():
    dynamics = UnicycleDynamics(0.05)
    simulation_horizon = 50
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
    xs, us, info = dynamics.rollout(x0, receding_horizon_strategy, simulation_horizon)
    return xs, us, dynamics
    # TODO: actually sanity-check the results


def _test_receding_horizon_path_following(dynamics=UnicycleDynamics(0.075)):
    simulation_horizon = 200
    prediction_horizon = 10
    x0 = np.array([0, 0, 0, 0.5])

    state_cost = CompositeCost(
        [
            PolylineTrackingCost(
                Polyline(
                    np.array(
                        [
                            [0, 0],
                            [1, 0],
                            [3, 1],
                            [4, 0],
                            [4, -0.5],
                            [3, -1],
                            [2, -1],
                            [0, -2],
                            [-1, -1],
                            [-1, -0.5],
                            [0, 0],
                        ]
                    )
                ),
                0.1,
            ),
            SetpointTrackingCost(np.diag([0, 0, 0, 0.1]), np.array([0, 0, 0, 1.5])),
        ]
    )

    input_cost = QuadraticCost(np.diag([1e-3, 1]), np.zeros(2))

    per_horizon_ocp = OptimalControlProblem(
        dynamics, state_cost, input_cost, prediction_horizon
    )
    inner_solver = ILQRSolver(per_horizon_ocp)
    receding_horizon_strategy = RecedingHorizonStrategy(inner_solver)
    xs, us, infos = dynamics.rollout(x0, receding_horizon_strategy, simulation_horizon)
    return xs, us, infos, per_horizon_ocp
    # TODO: actually sanity-check the results


def test_receding_horizon_path_following_unicycle():
    _test_receding_horizon_path_following(dynamics=UnicycleDynamics(0.075))


# This test still fails
def test_receding_horizon_path_following_bicycle():
  _test_receding_horizon_path_following(dynamics=BicycleDynamics(0.075))


def visual_sanity_check(dynamics):
    xs, us, infos, per_horizon_ocp = _test_receding_horizon_path_following(dynamics)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.gca().set_aspect("equal", adjustable="box")
    dt = per_horizon_ocp.dynamics.dt

    def animate_frame(i):
        x = xs[i]
        info = infos[i]

        pred = info["predictions"]
        ax.clear()
        per_horizon_ocp.state_cost.visualize(ax)
        ax.plot(xs[:, 0], xs[:, 1], label="Closed-loop")
        per_horizon_ocp.dynamics.visualize_state(ax, x)
        ax.plot(pred[:, 0], pred[:, 1], label="Prediction")
        ax.legend()
        fig.canvas.draw()
        fig.canvas.flush_events()

    animation = FuncAnimation(
        fig, func=animate_frame, frames=range(len(infos)), interval=1
    )

    writer = writers["ffmpeg"](fps=1/dt)
    animation.save("test.mp4", writer, dpi=200)


if __name__ == "__main__":
    # visual_sanity_check(UnicycleDynamics(0.075))
    visual_sanity_check(BicycleDynamics(0.075))
