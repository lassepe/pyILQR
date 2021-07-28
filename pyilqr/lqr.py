import numpy as np

from pyilqr.dynamics import LinearDynamics
from pyilqr.strategies import AffineStrategy, AffineStageStrategy
from pyilqr.costs import QuadraticCost


# TODO wrap this in a solver object that handles the memorey management
def solve_lqr(dynamics: LinearDynamics, costs: QuadraticCost):
    nx, nu = dynamics.dims
    H = dynamics.horizon

    # inialize the cost2go estimate
    _terminal_cost = costs.state_cost[H]
    Z, z = _terminal_cost.Q, _terminal_cost.l

    BZ = np.zeros((nu, nx))
    S = np.zeros((nu, nu))
    YP = np.zeros((nu, nx))
    Ya = np.zeros(nu)

    # TODO: could preallocate
    strategy = AffineStrategy([])

    # solve for the value function and feedback gains backward in time
    # TODO: think about off-by-one error in time
    for k in reversed(range(H)):
        A, B = dynamics.A(k), dynamics.B(k)
        Q, l = costs.Q(k), costs.l(k)
        R, r = costs.R(k), costs.r(k)

        # setup system of equations
        BZ = B.T @ Z
        S = R + BZ @ B
        YP = BZ @ A
        Ya = B.T @ z + r

        # compute strategy for this stage
        Sinv = np.linalg.inv(S)
        P = Sinv @ YP
        a = Sinv @ Ya
        strategy.stage_strategies.insert(0, AffineStageStrategy(P, a))


        # Update the cost2go
        F = A - B @ P
        b = -B @ a
        PR = P.T @ R
        z = F.T @ (z + Z @ b) + l + PR @ a - P.T @ r
        Z = F.T @ Z @ F + Q + PR @ P

    return strategy
