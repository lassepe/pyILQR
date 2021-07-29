import numpy as np

from pyilqr.example_dyanmics import UnicycleDynamics


def test_unicycle():
    dyn = UnicycleDynamics(0.2)
    assert dyn.dt == 0.2
    assert all(dyn.dx(x=np.zeros(4), u=np.zeros(2)) == np.zeros(4))
