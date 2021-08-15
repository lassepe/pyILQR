import numpy as np
from dataclasses import dataclass
from pyilqr.strategies import AffineStrategy, AffineStageStrategy
from pyilqr.ocp import LQRProblem

# Note: Could optimize this by making LQRSolver manage the memory
# - store intermediate result matrices like S et al
# - preallocate the memory for the strategy so that it can just be updated inplace
@dataclass
class LQRSolver:
    ocp: LQRProblem

    def solve(self):
        H = self.ocp.dynamics.horizon
        # inialize the cost2go estimate
        _terminal_cost = self.ocp.state_cost[H]
        Z, z = _terminal_cost.Q, _terminal_cost.l

        strategy = AffineStrategy([])

        # solve for the value function and feedback gains backward in time
        for k in reversed(range(H)):
            A, B = self.ocp.dynamics.A(k), self.ocp.dynamics.B(k)
            Q, l = self.ocp.state_cost[k].Q, self.ocp.state_cost[k].l
            R, r = self.ocp.input_cost[k].Q, self.ocp.input_cost[k].l

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
