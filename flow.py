import maxflow
import numpy as np


def select_patch_mask(dis_np, comp_np, crop_np):
    assert comp_np.shape == crop_np.shape
    g = maxflow.GraphFloat()
    nodeids = g.add_grid_nodes(comp_np.shape[:2])
    structure = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]])
    weights = (comp_np - crop_np).abs().sum(axis=2)
    g.add_grid_edges(nodeids, weights=weights,
                     structure=structure, symmetric=True)
    structure = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]])
    g.add_grid_edges(nodeids, weights=weights,
                     structure=structure, symmetric=True)
    sourcecaps = (dis_np == 1) * 1e18
    sinkcaps = (dis_np == dis_np.max()) * 1e18
    g.add_grid_tedges(nodeids, sourcecaps=sourcecaps, sinkcaps=sinkcaps)
    f = g.maxflow()
    patch_mask = g.get_grid_segments(nodeids)
    return patch_mask, f
