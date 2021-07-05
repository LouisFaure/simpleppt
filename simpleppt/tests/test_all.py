import simpleppt
import numpy as np
from simpleppt import __path__
import os


def test_all():

    data, emb = simpleppt.load_example()

    SP = simpleppt.ppt(data, Nodes=10, seed=1)

    SP.set_branches()

    simpleppt.project_ppt(SP, emb)

    assert np.allclose(np.argwhere(SP.B).ravel()[:5], [0, 7, 1, 6, 1])
    assert np.allclose(
        SP.F[0, :5], [20.18435385, 2.14613483, -5.09374701, -5.76036824, -6.95545319]
    )
    assert np.allclose(
        SP.d[0, :5], [0.0, 18.1882903, 27.16831346, 25.95303266, 27.19300147]
    )
    assert np.allclose(SP.forks, [8])
    assert np.allclose(SP.tips, [0, 2, 5])
