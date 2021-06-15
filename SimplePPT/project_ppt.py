import numpy as np
import pandas as pd
import igraph
from anndata import AnnData
import matplotlib.collections
from typing import Union, Optional, Sequence, Tuple, List
import plotly.graph_objects as go
import scanpy as sc
from cycler import Cycler

from pandas.api.types import is_categorical_dtype
from scanpy.plotting._utils import savefig_or_show
import types

from matplotlib.backend_bases import GraphicsContextBase, RendererBase
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from numba import njit
import math

def project_ppt(
    graph,
    emb,
    size_nodes: float = None,
    color_cells: Union[str, None] = None,
    tips: bool = True,
    forks: bool = True,
    nodes: Optional[List] = [],
    ax=None,
    show: Optional[bool] = None,
    save: Union[str, bool, None] = None,
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

    if "components" in kwargs:
        cmp = np.array(kwargs["components"]) - 1
        emb = emb[:, cmp]

    else:
        emb = emb[:, :2]

    R = graph["R"]

    proj = (np.dot(emb.T, R) / R.sum(axis=0)).T

    B = graph["B"]

    if ax is None:
        ax = sc.pl.embedding(
            adata, color=color_cells, basis=basis, show=False, **kwargs
        )
    else:
        sc.pl.embedding(
            adata, color=color_cells, basis=basis, ax=ax, show=False, **kwargs
        )

    al = np.array(
        igraph.Graph.Adjacency((B > 0).tolist(), mode="undirected").get_edgelist()
    )
    segs = al.tolist()
    vertices = proj.tolist()
    lines = [[tuple(vertices[j]) for j in i] for i in segs]
    lc = matplotlib.collections.LineCollection(lines, colors="k", linewidths=2)
    ax.add_collection(lc)

    ax.scatter(proj[:, 0], proj[:, 1], s=size_nodes, c="k")

    bbox = dict(facecolor="white", alpha=0.6, edgecolor="white", pad=0.1)

    if tips:
        for tip in graph["tips"]:
            ax.annotate(
                tip,
                (proj[tip, 0], proj[tip, 1]),
                ha="center",
                va="center",
                xytext=(-8, 8),
                textcoords="offset points",
                bbox=bbox,
            )
    if forks:
        for fork in graph["forks"]:
            ax.annotate(
                fork,
                (proj[fork, 0], proj[fork, 1]),
                ha="center",
                va="center",
                xytext=(-8, 8),
                textcoords="offset points",
                bbox=bbox,
            )
    if nodes:
        for node in nodes:
            ax.annotate(
                node,
                (proj[node, 0], proj[node, 1]),
                ha="center",
                va="center",
                xytext=(-8, 8),
                textcoords="offset points",
                bbox=bbox,
            )
    if show == False:
        return ax

    savefig_or_show("graph", show=show, save=save)