import numpy as np
from scipy.sparse import lil_matrix, diags


def blend_channel(scene_np, mask_np, patch_np):
    n = mask_np.sum()
    A = lil_matrix((n, n), dtype=np.float32)
    b = np.zeros(n)
    idx_np = np.zeros(mask_np.shape, dtype=np.int32)
    xs, ys = np.where(mask_np == 1)
    for idx, x, y in enumerate(zip(xs, ys)):
        assert mask_np[x, y] == 1
        idx_np[x, y] = idx
    dx = np.array([0, 1, 0, -1])
    dy = np.array([1, 0, -1, 0])
    xmax, ymax = mask_np.shape[:2]
    for idx, x, y in enumerate(zip(xs, ys)):
        for nx, ny in zip(dx, dy):
            nx = nx + x
            ny = ny + y
            if nx < 0 or nx >= xmax:
                continue
            if ny < 0 or ny >= ymax:
                continue
            A[idx, idx] += 1
            b[idx] += float(patch_np[x, y]) - float(patch_np[nx, ny])
            if mask_np[nx, ny] == 1:
                nidx = idx_np[nx, ny]
                A[idx, nidx] = -1
            if mask_np[nx, ny] == 0:
                b[idx] += scene_np[nx, ny]
    diag = A.diagonal()
    D = diags(diag)
    Dinv = diags(1.0 / diag)
    B = -Dinv @ (A - D)
    f = Dinv.dov(b)
    x = np.random.randn(n)
    for _ in range(10):
        x = B.dot(x) + f
    res = np.zeros(mask_np.shape, dtype=np.int32)
    for idx, x, y in enumerate(zip(xs, ys)):
        res[x, y] = x[idx]
    res = np.expand_dims(res, axis=2)
    return res


def blend(scene_np, mask_np, patch_np):
    r = blend_channel(scene_np[:, :, 0], mask_np, patch_np[:, :, 0])
    g = blend_channel(scene_np[:, :, 1], mask_np, patch_np[:, :, 1])
    b = blend_channel(scene_np[:, :, 2], mask_np, patch_np[:, :, 2])
    return np.concatenate([r, g, b], axis=2)
