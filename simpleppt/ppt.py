from typing import Optional, Union
import numpy as np
from pandas import DataFrame
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
import igraph
from tqdm import tqdm
import sys
import math

from .utils import process_R_cpu, norm_R_cpu, cor_mat_cpu
from . import logging as logg
from . import settings
from .SimplePPT import SimplePPT


def ppt(
    X,
    W=None,
    Nodes: int = None,
    init: Optional[DataFrame] = None,
    sigma: Optional[Union[float, int]] = 0.1,
    lam: Optional[Union[float, int]] = 1,
    metric: str = "euclidean",
    nsteps: int = 50,
    err_cut: float = 5e-3,
    device: str = "cpu",
    gpu_tbp: int = 16,
    seed: Optional[int] = None,
    progress: bool = True,
):

    """\
    Generate a principal tree.

    Learn a simplified representation on any space, composed of nodes, approximating the
    position of the datapoints on a given space.

    Parameters
    ----------
    X
        n-dimensionnal matrix to be learned.
    W
        weight matrix, having the same dimensions as X.
    Nodes
        Number of nodes composing the principial tree.
    init
        Initialise the point positions.
    sigma
        Regularization parameter.
    lam
        Penalty for the tree length.
    metric
        The metric to use to compute distances in high dimensional space.
        For compatible metrics, check the documentation of
        sklearn.metrics.pairwise_distances if using cpu or
        cuml.metrics.pairwise_distances if using gpu.
    nsteps
        Number of steps for the optimisation process.
    err_cut
        Stop algorithm if proximity of principal points between iterations less than defined value.
    gpu_tpb
        Threads per block parameter for cuda computations.
    seed
        A numpy random seed.
    progress
        Show progressbar of the tree learning.

    Returns
    -------
    SimplePPT object with the following fields: :class:`simpleppt.SimplePPT`

        `.F`
            coordinates of principal points in the learned space.
        `.R`
            soft assignment of datapoints to principal points.
        `.B`
            adjacency matrix of the principal points.
        `.L`
            Laplacian matrix.
        `.d`
            Pairwise distance matrix of principal points.
        `.score`
            Score minimized during the tree learning.

    """

    logg.info(
        "inferring a principal tree",
        reset=True,
        end=" " if settings.verbosity > 2 else "\n",
    )
    logg.hint(
        "parameters used \n"
        "    "
        + str(Nodes)
        + " principal points, sigma = "
        + str(sigma)
        + ", lambda = "
        + str(lam)
        + ", metric = "
        + metric
    )
    X_t = X.T

    # if seed is not None:
    #    np.random.seed(seed)

    if device == "gpu":
        try:
            import rmm

            rmm.reinitialize(managed_memory=True)
            assert rmm.is_initialized()
            import cupy as cp
            from cuml.metrics import pairwise_distances
        except ModuleNotFoundError:
            raise Exception(
                "Some of the GPU dependencies are missing, use device='cpu' instead!"
            )
        from .utils import process_R_gpu, norm_R_gpu, cor_mat_gpu, mst_gpu, matmul

        X_gpu = cp.asarray(X_t, dtype=np.float64)
        W = cp.ones_like(X_gpu) if W is None else cp.asarray(W.T)

        if init is None:
            if seed is not None:
                np.random.seed(seed)
            F_mat_gpu = X_gpu[
                :, np.random.choice(X.shape[0], size=Nodes, replace=False)
            ]
        else:
            F_mat_gpu = cp.asarray(init.T)
            M = init.T.shape[0]

        iterator = tqdm(
            range(nsteps),
            file=sys.stdout,
            desc="    fitting",
            disable=progress == False,
        )
        for i in iterator:
            R = pairwise_distances(X_gpu.T, F_mat_gpu.T, metric=metric)

            threadsperblock = (gpu_tbp, gpu_tbp)
            blockspergrid_x = math.ceil(R.shape[0] / threadsperblock[0])
            blockspergrid_y = math.ceil(R.shape[1] / threadsperblock[1])
            blockspergrid = (blockspergrid_x, blockspergrid_y)

            process_R_gpu[blockspergrid, threadsperblock](R, sigma)
            Rsum = R.sum(axis=1)
            norm_R_gpu[blockspergrid, threadsperblock](R, Rsum)

            d = pairwise_distances(F_mat_gpu.T, metric=metric)
            mst = mst_gpu(d)
            mat = mst + mst.T - cp.diag(cp.diag(mst.A))
            B = (mat > 0).astype(int)

            D = cp.identity(B.shape[0]) * B.sum(axis=0)
            L = D - B
            M = L * lam + cp.identity(R.shape[1]) * R.sum(axis=0)
            old_F = F_mat_gpu

            dotprod = cp.zeros((X_gpu.shape[0], R.shape[1]))
            TPB = 16
            threadsperblock = (gpu_tbp, gpu_tbp)
            blockspergrid_x = math.ceil(dotprod.shape[0] / threadsperblock[0])
            blockspergrid_y = math.ceil(dotprod.shape[1] / threadsperblock[1])
            blockspergrid = (blockspergrid_x, blockspergrid_y)

            matmul[blockspergrid, threadsperblock]((X_gpu * W), R, dotprod)

            F_mat_gpu = cp.linalg.solve(M.T, dotprod.T).T

            err = cp.max(
                cp.sqrt((F_mat_gpu - old_F).sum(axis=0) ** 2)
                / cp.sqrt((F_mat_gpu ** 2).sum(axis=0))
            )
            if err < err_cut:
                iterator.close()
                logg.info("    converged")
                break

        if i == (nsteps - 1):
            logg.info("    inference not converged (error: " + str(err) + ")")

        score = cp.array(
            [
                cp.sum((1 - cor_mat_gpu(F_mat_gpu, X_gpu)) * R) / R.shape[0],
                sigma / R.shape[0] * cp.sum(R * cp.log(R)),
                lam / 2 * cp.sum(d * B),
            ]
        )

        score = cp.asnumpy(score)
        F = cp.asnumpy(F_mat_gpu)
        R = cp.asnumpy(R)
        B = cp.asnumpy(B)
        L = cp.asnumpy(L)
        d = cp.asnumpy(d)

    else:
        from sklearn.metrics import pairwise_distances
        from .utils import process_R_cpu, norm_R_cpu, cor_mat_cpu

        X_cpu = np.asarray(X_t, dtype=np.float64)
        W = np.ones_like(X_cpu) if W is None else W.T

        if init is None:
            if seed is not None:
                np.random.seed(seed)
            F_mat_cpu = X_cpu[
                :, np.random.choice(X.shape[0], size=Nodes, replace=False)
            ]
        else:
            F_mat_cpu = np.asarray(init.T)
            Nodes = init.T.shape[0]

        j = 1
        err = 100

        # while ((j <= nsteps) & (err > err_cut)):
        iterator = tqdm(
            range(nsteps),
            file=sys.stdout,
            desc="    fitting",
            disable=progress == False,
        )
        for i in iterator:
            R = pairwise_distances(X_cpu.T, F_mat_cpu.T, metric=metric)

            process_R_cpu(R, sigma)
            Rsum = R.sum(axis=1)
            norm_R_cpu(R, Rsum)

            d = pairwise_distances(F_mat_cpu.T, metric=metric)

            csr = csr_matrix(np.triu(d, k=-1))
            Tcsr = minimum_spanning_tree(csr)
            mat = Tcsr.toarray()
            mat = mat + mat.T - np.diag(np.diag(mat))
            B = (mat > 0).astype(int)

            D = (np.identity(B.shape[0])) * np.array(B.sum(axis=0))
            L = D - B
            M = L * lam + np.identity(R.shape[1]) * np.array(R.sum(axis=0))
            old_F = F_mat_cpu

            F_mat_cpu = np.linalg.solve(M.T, (np.dot(X_cpu * W, R)).T).T

            err = np.max(
                np.sqrt((F_mat_cpu - old_F).sum(axis=0) ** 2)
                / np.sqrt((F_mat_cpu ** 2).sum(axis=0))
            )

            err = err.item()
            if err < err_cut:
                iterator.close()
                logg.info("    converged")
                break

        if i == (nsteps - 1):
            logg.info("    not converged (error: " + str(err) + ")")

        score = [
            np.sum((1 - cor_mat_cpu(F_mat_cpu, X_cpu)) * R) / R.shape[0],
            sigma / R.shape[0] * np.sum(R * np.log(R)),
            lam / 2 * np.sum(d * B),
        ]

        F = F_mat_cpu

    SP = SimplePPT(F, R, B, L, d, score, lam, sigma, nsteps, metric)

    g = igraph.Graph.Adjacency((SP.B > 0).tolist(), mode="undirected")

    # remove lonely nodes
    co_nodes = np.argwhere(np.array(g.degree()) > 0).ravel()
    SP.R = SP.R[:, co_nodes]
    SP.F = SP.F[:, co_nodes]
    SP.B = SP.B[co_nodes, :][:, co_nodes]
    SP.L = SP.L[co_nodes, :][:, co_nodes]
    SP.d = SP.d[co_nodes, :][:, co_nodes]

    if len(co_nodes) < Nodes:
        logg.info("    " + str(Nodes - len(co_nodes)) + " lonely nodes removed")

    SP.set_tips_forks()

    if len(SP.tips) > 30:
        logg.info("    more than 30 tips detected!")
    logg.info("    finished", time=True, end=" " if settings.verbosity > 2 else "\n")

    return SP
