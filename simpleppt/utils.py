from numba import cuda, njit, prange
import math

from scipy import sparse
import numpy as np


@cuda.jit
def process_R_gpu(R, sigma):
    x, y = cuda.grid(2)
    if x < R.shape[0] and y < R.shape[1]:
        R[x, y] = math.exp(-R[x, y] / sigma)


@cuda.jit
def norm_R_gpu(R, Rsum):
    x, y = cuda.grid(2)
    if x < R.shape[0] and y < R.shape[1]:
        if Rsum[x] == 0:
            R[x, y] = 0
        else:
            R[x, y] = R[x, y] / Rsum[x]


@cuda.jit
def matmul(A, B, C):
    i, j = cuda.grid(2)
    if i < C.shape[0] and j < C.shape[1]:
        tmp = 0.0
        for k in range(A.shape[1]):
            tmp += A[i, k] * B[k, j]
        C[i, j] = tmp


@njit(parallel=True)
def process_R_cpu(R, sigma):
    Rshape = R.shape
    R = R.ravel()
    for x in prange(len(R)):
        R[x] = math.exp(-R[x] / sigma)
    R.reshape(Rshape)


@njit(parallel=True)
def norm_R_cpu(R, Rsum):
    for x in prange(len(Rsum)):
        for y in range(R.shape[1]):
            R[x, y] = R[x, y] / Rsum[x]
            if math.isnan(R[x, y]):
                R[x, y] = 0


def cor_mat_cpu(A, B):
    import numpy as np

    A1 = A - A.mean(axis=0)
    B1 = B - B.mean(axis=0)
    res = (B1.T.dot(A1)).T / np.sqrt(
        (A1**2).sum(axis=0).reshape(A1.shape[1], 1)
        @ (B1**2).sum(axis=0).reshape(1, B1.shape[1])
    )
    return res.T


def cor_mat_gpu(A, B):
    import cupy as cp

    A1 = A - A.mean(axis=0)
    B1 = B - B.mean(axis=0)
    res = (B1.T.dot(A1)).T / cp.sqrt(
        (A1**2).sum(axis=0).reshape(A1.shape[1], 1)
        @ (B1**2).sum(axis=0).reshape(1, B1.shape[1])
    )
    return res.T


def mst_gpu(d):
    import numpy as np
    import cudf
    import cupy as cp
    from cupyx.scipy.sparse import csr_matrix as csr_cupy
    from cupyx.scipy.sparse import coo_matrix
    from cugraph.tree.minimum_spanning_tree_wrapper import mst_double, mst_float
    import scipy

    csr_gpu = csr_cupy(d)
    offsets = cudf.Series(csr_gpu.indptr)
    indices = cudf.Series(csr_gpu.indices)

    num_verts = csr_gpu.shape[0]
    num_edges = len(csr_gpu.indices)
    weights = cudf.Series(csr_gpu.data)

    if weights.dtype == np.float32:
        mst = mst_float(num_verts, num_edges, offsets, indices, weights)

    else:
        mst = mst_double(num_verts, num_edges, offsets, indices, weights)

    mst = csr_cupy(
        coo_matrix(
            (mst.weight.values, (mst.src.values, mst.dst.values)),
            shape=(num_verts, num_verts),
        )
    ).get()
    return csr_cupy(scipy.sparse.triu(mst))
