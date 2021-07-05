import numpy as np
import igraph
import matplotlib.collections
from typing import Optional
import matplotlib.pyplot as plt


def project_ppt(
    SP,
    emb,
    size_nodes: float = None,
    plot_datapoints=True,
    alpha_seg=1,
    alpha_nodes=1,
    ax=None,
    show: Optional[bool] = None,
    **kwargs,
):

    """\
    Project principal graph onto embedding.

    Parameters
    ----------
    SP
        SimplePPT object.
    emb
        embedding to project the tree onto.
    size_nodes
        size of the projected prinicpal points.
    alpha_seg
        segment alpha
    alpha_nodes
        node alpha.
    ax
        Add plot to existing ax
    show
        show the plot.
    kwargs
        arguments to pass to scanpy functions plt.scatter

    Returns
    -------
    If `show==False` a :class:`~matplotlib.axes.Axes`
    """

    R = SP.R

    proj = (np.dot(emb.T, R) / R.sum(axis=0)).T

    B = SP.B

    if ax is None:
        fig, ax = plt.subplots()

    if plot_datapoints:
        ax.scatter(emb[:, 0], emb[:, 1], **kwargs)

    al = np.array(
        igraph.Graph.Adjacency((B > 0).tolist(), mode="undirected").get_edgelist()
    )
    segs = al.tolist()
    vertices = proj.tolist()
    lines = [[tuple(vertices[j]) for j in i] for i in segs]
    lc = matplotlib.collections.LineCollection(
        lines, colors="darkblue", alpha=alpha_seg, linewidths=2
    )
    ax.add_collection(lc)

    ax.scatter(proj[:, 0], proj[:, 1], s=size_nodes, c="k", alpha=alpha_nodes)

    bbox = dict(facecolor="white", alpha=0.6, edgecolor="white", pad=0.1)

    if show == False:
        return ax
    else:
        plt.show()
