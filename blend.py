import numpy as np
import cupy
from cupyx.scipy.sparse import csr_matrix
from cupyx.scipy.sparse.linalg import cg
from tqdm import tqdm

def blend(scene_np, mask_np, patch_np):
    n = mask_np.sum()
    b = np.zeros((n, 3), dtype=np.int32)
    patch_np = np.int32(patch_np)
    idx_np = np.zeros(mask_np.shape, dtype=np.int32)
    xs, ys = np.where(mask_np == 1)
    for idx, (x, y) in enumerate(zip(xs, ys)):
        idx_np[x, y] = idx
    dx = np.array([0, 1, 0, -1])
    dy = np.array([1, 0, -1, 0])
    xmax, ymax = mask_np.shape[:2]
    diag = np.zeros(n, dtype=np.int32)
    spxs = []
    spys = []
    spdata = []
    for idx, (x, y) in tqdm(enumerate(zip(xs, ys))):
        for nx, ny in zip(dx, dy):
            nx = nx + x
            ny = ny + y
            if nx < 0 or nx >= xmax:
                continue
            if ny < 0 or ny >= ymax:
                continue
            diag[idx] += 1
            b[idx] += patch_np[x, y] - patch_np[nx, ny]
            if mask_np[nx, ny] == 1:
                nidx = idx_np[nx, ny]
                spxs.append(idx)
                spys.append(nidx)
                spdata.append(-1)
            if mask_np[nx, ny] == 0:
                b[idx] += scene_np[nx, ny]
    spdata = np.concatenate((spdata, diag))
    spdata = np.concatenate((spdata, spdata, spdata))
    spxs = np.concatenate((np.array(spxs), np.arange(n)))
    spxs = np.concatenate((spxs, spxs + n, spxs + n + n))
    spys = np.concatenate((np.array(spys), np.arange(n)))
    spys = np.concatenate((spys, spys + n, spys + n + n))
    A = csr_matrix((cupy.array(spdata), (cupy.array(spxs), cupy.array(spys))), shape=(n * 3, n * 3), dtype=np.float32)
    (X, _) = cg(A, cupy.array(np.concatenate((b[:, 0], b[:, 1], b[:, 2])), dtype=np.float32))
    res = np.zeros(scene_np.shape, dtype=np.int32)
    for idx, (x, y) in enumerate(zip(xs, ys)):
        res[x, y, 0] = X[idx]
        res[x, y, 1] = X[idx + n]
        res[x, y, 2] = X[idx + n + n]
    return res
