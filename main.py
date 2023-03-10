import argparse
import os
import numpy as np
import jittor as jt
from flow import select_patch_mask
from blend import blend
from PIL import Image

if jt.has_cuda:
    jt.flags.use_cuda = 1
print(jt.compiler.has_cuda)
print(jt.flags.use_cuda)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True)
    parser.add_argument("--mask", required=True)
    parser.add_argument("--comp", required=True)
    parser.add_argument("--save", required=True)
    parser.add_argument("--vicinity_px", default=80)
    parser.add_argument("--crop_scale_min", default=0.8)
    parser.add_argument("--crop_scale_max", default=1.2)
    args = parser.parse_args()
    return args


def vicinity_via_bfs(mask_im, vicinity_px):
    mask_np = np.asarray(mask_im.convert('1'))
    xs, ys = np.where(mask_np == False)  # noqa: E712
    xs, ys = xs.reshape(-1, 1), ys.reshape(-1, 1)
    q = np.concatenate((xs, ys), axis=1).tolist()
    x_max, y_max = mask_np.shape
    dx = [0, 1, 0, -1]
    dy = [1, 0, -1, 0]
    vici_np = np.zeros_like(mask_np, dtype=np.int32)
    while len(q) > 0:
        x, y = q.pop(0)
        if vici_np[x, y] >= vicinity_px:
            continue
        for nx, ny in zip(dx, dy):
            nx = nx + x
            ny = ny + y
            if nx < 0 or nx >= x_max:
                continue
            if ny < 0 or ny >= y_max:
                continue
            if not mask_np[nx, ny]:
                continue
            if vici_np[nx, ny] != 0:
                continue
            vici_np[nx, ny] = vici_np[x, y] + 1
            q.append(np.array((nx, ny)))
    dis_np = vici_np
    vici_np = (vici_np > 0) * 1
    return vici_np, dis_np


def crop(scene_im, vici_np, dis_np, mask_im):
    mask_np = np.asarray(mask_im.convert('1'))
    xs, ys = np.where(vici_np == 1)
    _xs, _ys = np.where(mask_np == False)
    xs, ys = np.concatenate((xs, _xs)), np.concatenate((ys, _ys))
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    vici_np = np.expand_dims(vici_np, axis=2)
    dis_np = np.expand_dims(dis_np, axis=2)
    crop_np = np.asarray(scene_im) * vici_np
    return (crop_np[x_min:x_max + 1, y_min:y_max + 1],
            vici_np[x_min:x_max + 1, y_min:y_max + 1],
            dis_np[x_min:x_max + 1, y_min:y_max + 1],
            x_min, x_max, y_min, y_max)


def conv(crop_np, crop_scale, comp_im, vici_np):
    nh, nw = crop_np.shape[1] * crop_scale, crop_np.shape[0] * crop_scale
    nh, nw = int(nh), int(nw)
    crop_im = Image.fromarray(np.uint8(crop_np)).resize((nh, nw))
    crop_np = np.asarray(crop_im)
    vici_np = vici_np.repeat(3, axis=2)
    vici_im = Image.fromarray(np.uint8(vici_np)).resize((nh, nw))
    vici_np = np.asarray(vici_im)
    comp_np = np.asarray(comp_im)
    [h1, w1, _] = comp_np.shape
    [h2, w2, _] = crop_np.shape
    if h2 >= h1 or w2 >= w1:
        return -1, -1, 1e18
    comp_conv_jt = jt.array(comp_np, dtype=jt.float32).reindex(
            [h1 - h2 + 1, w1 - w2 + 1, h2, w2, 3],
            ['i0 + i2', 'i1 + i3', 'i4'])
    crop_conv_jt = jt.array(crop_np, dtype=jt.float32).broadcast_var(
            comp_conv_jt)
    vici_conv_jt = jt.array(vici_np, dtype=jt.float32).broadcast_var(
            comp_conv_jt)
    error_jt = (comp_conv_jt * vici_conv_jt - crop_conv_jt) ** 2
    error_jt = error_jt.sum([2, 3, 4])
    error_np = error_jt.fetch_sync()
    (x, y) = np.unravel_index(error_np.argmin(), error_np.shape)
    return x, y, error_np[x, y]


def select_comp(crop_scale_min, crop_scale_max, comp_im, crop_np, vici_np):
    best_err = 1e18
    bx, by, bs = None, None, None
    for crop_scale in np.arange(crop_scale_min, crop_scale_max, 0.1):
        x, y, err = conv(crop_np, crop_scale, comp_im, vici_np)
        if err < best_err:
            bx, by, bs = x, y, crop_scale
            best_err = err
    nh, nw = crop_np.shape[1] * bs, crop_np.shape[0] * bs
    nh, nw = int(nh), int(nw)
    comp_np = np.asarray(comp_im)
    comp_np = comp_np[bx:bx + nw, by:by + nh]
    comp_im = Image.fromarray(np.uint8(comp_np)).resize(
            (crop_np.shape[1], crop_np.shape[0]))
    return np.asarray(comp_im)


def main(args):
    assert os.path.exists(args.scene)
    assert os.path.exists(args.mask)
    assert os.path.exists(args.comp)
    scene_im = Image.open(args.scene)
    mask_im = Image.open(args.mask)
    comp_im = Image.open(args.comp)
    scene_np = np.asarray(scene_im)
    vici_np, dis_np = vicinity_via_bfs(mask_im, args.vicinity_px)
    crop_np, vici_np, dis_np, \
        x_min, x_max, y_min, y_max = crop(scene_im, vici_np, dis_np, mask_im)
    comp_np = select_comp(
            args.crop_scale_min, args.crop_scale_max,
            comp_im, crop_np, vici_np)
    patch_mask_np = select_patch_mask(dis_np, comp_np, crop_np)
    patch_np = comp_np * (1 - np.expand_dims(patch_mask_np, axis=2))
    global_mask_np = np.zeros(scene_np.shape[:2], dtype=np.int32)
    global_mask_np[x_min:x_max + 1, y_min:y_max + 1] = 1 - patch_mask_np
    global_patch_np = np.zeros(scene_np.shape, dtype=np.int32)
    global_patch_np[x_min:x_max + 1, y_min:y_max + 1] = comp_np
    output = blend(scene_np, global_mask_np, global_patch_np)
    mask = np.expand_dims(global_mask_np, axis=2)
    output = scene_np * (1 - mask) + output * mask
    return output


if __name__ == '__main__':
    args = parse_args()
    output = main(args)
    output = Image.fromarray(np.uint8(np.clip(output, 0, 255)))
    output.save(args.save)