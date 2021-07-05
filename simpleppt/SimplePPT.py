from typing import Any, Union, Optional, Mapping, Iterable  # Meta
from typing import Mapping
import numpy as np
import igraph
from sklearn.metrics import pairwise_distances
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
import pandas as pd
import itertools


class SimplePPT:
    """A python object containing the data used for dynamical tracks analysis.

    Parameters
    ----------

    F
        coordinates of principal points in the learned space.
    R
        soft assignment of datapoints to principal points.
    B
        adjacency matrix of the principal points.
    L
        Laplacian matrix.
    d
        Pairwise distance matrix of principal points.
    score
        Score minimized during the tree learning.
    tips
        Node IDs of the tree that have degree 1.
    forks
        Node IDs of the tree that have a degree of more than 1.
    root
        Selected node ID as the root of the tree for distance calculations.
    pp_info
        Per node ID info of distance from the root, and segment assigment.
    pp_seg
        Per segment info with node ID extremities and distance."""

    def __init__(
        self,
        F: np.array,
        R: np.array,
        B: np.array,
        L: np.array,
        d: np.array,
        score: float,
        lam: float,
        sigma: float,
        nsteps: int,
        metric: str,
        tips: Optional[Union[Iterable, None]] = None,
        forks: Optional[Union[Iterable, None]] = None,
        root: Optional[Union[int, None]] = None,
        pp_info: Optional[Union[pd.DataFrame]] = None,
        pp_seg: Optional[Union[pd.DataFrame]] = None,
    ):

        self.F = F
        self.R = R
        self.B = B
        self.L = L
        self.d = d
        self.score = score
        self.lam = lam
        self.sigma = sigma
        self.nsteps = nsteps
        self.metric = metric
        self.tips = tips
        self.forks = forks

    def __repr__(self):
        dt, nd = self.R.shape
        descr = f"SimplePPT object of {nd} nodes approximating {dt} datapoints"
        return descr

    def set_tips_forks(self):
        """Obtains the tips and forks of the tree.

        Returns
        -------
        adds to SimplePPT object the following fields: :class:`simpleppt.SimplePPT`

            `.tips`
                Node IDs of the tree that have degree 1..
            `.forks`
                Node IDs of the tree that have a degree of more than 1.

        """
        g = igraph.Graph.Adjacency((self.B > 0).tolist(), mode="undirected")
        self.tips = np.argwhere(np.array(g.degree()) == 1).flatten()
        self.forks = np.argwhere(np.array(g.degree()) > 2).flatten()

    def set_branches(self, root=None):
        """Assign branches/segments to nodes.

        Returns
        -------
        adds to SimplePPT object the following fields: :class:`simpleppt.SimplePPT`

            `.pp_info`
                Per node ID info of distance from the root, and segment assigment.
            `.pp_seg`
                Per segment info with node ID extremities and distance.
        """
        root = self.tips[0] if root is None else root
        d = 1e-6 + pairwise_distances(self.F.T, self.F.T, metric=self.metric)

        to_g = self.B * d

        csr = csr_matrix(to_g)

        g = igraph.Graph.Adjacency((to_g > 0).tolist(), mode="undirected")
        g.es["weight"] = to_g[to_g.nonzero()]

        root_dist_matrix = shortest_path(csr, directed=False, indices=root)
        pp_info = pd.DataFrame(
            {
                "PP": g.vs.indices,
                "dist": root_dist_matrix,
                "seg": np.zeros(csr.shape[0]),
            }
        )

        nodes = np.argwhere(
            np.apply_along_axis(arr=(csr > 0).todense(), axis=0, func1d=np.sum) != 2
        ).flatten()
        nodes = np.unique(np.append(nodes, root))

        pp_seg = pd.DataFrame(columns=["n", "from", "to", "d"])
        for node1, node2 in itertools.combinations(nodes, 2):
            paths12 = g.get_shortest_paths(node1, node2)
            paths12 = np.array([val for sublist in paths12 for val in sublist])

            if np.sum(np.isin(nodes, paths12)) == 2:
                fromto = np.array([node1, node2])
                path_root = root_dist_matrix[[node1, node2]]
                fro = fromto[np.argmin(path_root)]
                to = fromto[np.argmax(path_root)]
                pp_info.loc[paths12, "seg"] = pp_seg.shape[0] + 1
                pp_seg = pp_seg.append(
                    pd.DataFrame(
                        {
                            "n": pp_seg.shape[0] + 1,
                            "from": fro,
                            "to": to,
                            "d": shortest_path(csr, directed=False, indices=fro)[to],
                        },
                        index=[pp_seg.shape[0] + 1],
                    )
                )

        pp_seg["n"] = pp_seg["n"].astype(int).astype(str)
        pp_seg["n"] = pp_seg["n"].astype(int).astype(str)

        pp_seg["from"] = pp_seg["from"].astype(int)
        pp_seg["to"] = pp_seg["to"].astype(int)

        pp_info["seg"] = pp_info["seg"].astype(int).astype(str)
        pp_info["seg"] = pp_info["seg"].astype(int).astype(str)

        self.pp_info = pp_info
        self.pp_seg = pp_seg
        self.root = root
