import numpy as np
import cupy
from scipy.sparse import lil_matrix, diags
from cupyx.scipy.sparse import csr_matrix
from cupyx.scipy.sparse.linalg import spsolve
from tqdm import tqdm

def blend(scene_np, mask_np, patch_np):
    n = mask_np.sum()
    A = lil_matrix((n, n), dtype=np.int32)
    br = np.zeros(n, dtype=np.int32)
    bg = np.zeros(n, dtype=np.int32)
    bb = np.zeros(n, dtype=np.int32)
    patch_np = np.int32(patch_np)
    idx_np = np.zeros(mask_np.shape, dtype=np.int32)
    xs, ys = np.where(mask_np == 1)
    for idx, (x, y) in enumerate(zip(xs, ys)):
        idx_np[x, y] = idx
    dx = np.array([0, 1, 0, -1])
    dy = np.array([1, 0, -1, 0])
    xmax, ymax = mask_np.shape[:2]
    for idx, (x, y) in tqdm(enumerate(zip(xs, ys))):
        for nx, ny in zip(dx, dy):
            nx = nx + x
            ny = ny + y
            if nx < 0 or nx >= xmax:
                continue
            if ny < 0 or ny >= ymax:
                continue
            A[idx, idx] += 1
            br[idx] += patch_np[x, y, 0] - patch_np[nx, ny, 0]
            bg[idx] += patch_np[x, y, 1] - patch_np[nx, ny, 1]
            bb[idx] += patch_np[x, y, 2] - patch_np[nx, ny, 2]
            if mask_np[nx, ny] == 1:
                nidx = idx_np[nx, ny]
                A[idx, nidx] = -1
            if mask_np[nx, ny] == 0:
                br[idx] += scene_np[nx, ny, 0]
                bg[idx] += scene_np[nx, ny, 1]
                bb[idx] += scene_np[nx, ny, 2]
    A = A.tocsr()
    A = csr_matrix(A, dtype=np.float32)
    Xr, Xg, Xb = spsolve(A, cupy.array(br, dtype=np.float32)), spsolve(A, cupy.array(bg, dtype=np.float32)), spsolve(A, cupy.array(bb, dtype=np.float32))
    res = np.zeros(scene_np.shape, dtype=np.int32)
    for idx, (x, y) in enumerate(zip(xs, ys)):
        res[x, y, 0] = Xr[idx]
        res[x, y, 1] = Xg[idx]
        res[x, y, 2] = Xb[idx]
    return res
