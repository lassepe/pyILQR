import numpy as np
from pyilqr.example_costs import TrackingCost


def test_tracking_cost():
    R = np.eye(1)
    Q = np.eye(2)

    cost = TrackingCost(R, Q, x_target = np.zeros(2))
    qcost = cost.quadratisized_along_trajectory([np.zeros(2)] , [np.zeros(1)])
    assert np.all(qcost.state_cost[0].Q == Q)
    assert np.all(qcost.state_cost[0].l == np.zeros(2))
    assert np.all(qcost.input_cost[0].Q == R)
    assert np.all(qcost.input_cost[0].l == np.zeros(2))

    cost = TrackingCost(R, Q, x_target = np.ones(2))
    qcost = cost.quadratisized_along_trajectory([np.zeros(2)] , [np.zeros(1)])
    assert np.all(qcost.state_cost[0].Q == Q)
    assert np.all(qcost.state_cost[0].l == -np.ones(2))
    assert np.all(qcost.input_cost[0].Q == R)
    assert np.all(qcost.input_cost[0].l == np.zeros(2))


if __name__ == '__main__':
    test_tracking_cost()
