import numpy as np
import igraph
import matplotlib.collections
from typing import Optional
import matplotlib.pyplot as plt


def project_ppt(
    ppt,
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
    adata
        Annotated data matrix.
    basis
        Name of the `obsm` basis to use.
    size_nodes
        size of the projected prinicpal points.
    color_cells
        cells color
    tips
        display tip ids.
    forks
        display fork ids.
    nodes
        display any node id.
    ax
        Add plot to existing ax
    show
        show the plot.
    save
        save the plot.
    kwargs
        arguments to pass to scanpy functions pl.embedding
    Returns
    -------
    If `show==False` a :class:`~matplotlib.axes.Axes`
    """

    R = ppt["R"]

    proj = (np.dot(emb.T, R) / R.sum(axis=0)).T

    B = ppt["B"]

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
