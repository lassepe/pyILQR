import numpy as np

from pyilqr.dynamics import LinearDynamics
from pyilqr.strategies import AffineStrategy, AffineStageStrategy
from pyilqr.costs import QuadraticCost

# Note: Could wrap this i an object that does the memory management, in particular:
# - store intermediate result matrices like S et al
# - preallocate the memory for the strategy so that it can just be updated inplace
def solve_lqr(dynamics: LinearDynamics, costs: QuadraticCost):
    H = dynamics.horizon
    # inialize the cost2go estimate
    _terminal_cost = costs.state_cost[H]
    Z, z = _terminal_cost.Q, _terminal_cost.l

    strategy = AffineStrategy([])

    # solve for the value function and feedback gains backward in time
    for k in reversed(range(H)):
        A, B = dynamics.A(k), dynamics.B(k)
        Q, l = costs.Q(k), costs.l(k)
        R, r = costs.R(k), costs.r(k)

        # setup system of equations
        BZ = B.T @ Z
        S = R + BZ @ B
        YP = BZ @ A
        Ya = B.T @ z + r

        # compute strategy for this stage; could be done more efficiently with householder QR
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
