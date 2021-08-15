import numpy as np
from pyilqr.example_costs import SetpointTrackingCost


def test_cost():
    Q = np.eye(2)

    cost = SetpointTrackingCost(Q, x_target=np.zeros(2))
    qcost = cost.quadratisized_along_trajectory([np.zeros(2)])
    assert np.all(qcost[0].Q == Q)
    assert np.all(qcost[0].l == np.zeros(2))

    qcost = cost.quadratisized_along_trajectory([np.ones(2)])
    assert np.all(qcost[0].Q == Q)
    assert np.all(qcost[0].l == np.ones(2))


if __name__ == "__main__":
    test_cost()
