import argparse
import os
import numpy as np
import jittor as jt
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True)
    parser.add_argument("--mask", required=True)
    parser.add_argument("--comp", required=True)
    parser.add_argument("--vicinity_px", default=80)
    parser.add_argument("--crop_scale_min", default=0.5)
    parser.add_argument("--crop_scale_max", default=2.0)
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
    vici_np = (vici_np > 0) * 1
    return vici_np


def crop(scene_im, vici_np):
    xs, ys = np.where(vici_np == 1)
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    vici_np = np.expand_dims(vici_np, axis=2)
    crop_np = np.asarray(scene_im) * vici_np
    return crop_np[x_min:x_max + 1, y_min:y_max + 1]


def conv(crop_np, crop_scale, comp_im):
    nh, nw = crop_np.shape[1] * crop_scale, crop_np.shape[0] * crop_scale
    nh, nw = int(nh), int(nw)
    crop_im = Image.fromarray(np.uint8(crop_np)).resize((nh, nw))
    crop_np = np.asarray(crop_im)
    comp_np = np.asarray(comp_im)
    [h1, w1, _] = comp_np.shape
    [h2, w2, _] = crop_np.shape
    comp_conv_jt = jt.array(comp_np).reindex(
            [h1 - h2 + 1, w1 - w2 + 1, h2, w2, 3],
            ['i0 + i2', 'i1 + i3', 'i4'])
    crop_conv_jt = jt.array(crop_np).broadcast_var(comp_conv_jt)
    conv_jt = -2 * comp_conv_jt * crop_conv_jt
    conv_jt += comp_conv_jt * comp_conv_jt
    conv_jt += crop_conv_jt * crop_conv_jt
    conv_jt = conv_jt.sum([2, 3, 4])
    return np.asarray(conv_jt)


def main(args):
    assert os.path.exists(args.scene)
    assert os.path.exists(args.mask)
    assert os.path.exists(args.comp)
    scene_im = Image.open(args.scene)
    mask_im = Image.open(args.mask)
    comp_im = Image.open(args.comp)
    vici_np = vicinity_via_bfs(mask_im, args.vicinity_px)
    crop_np = crop(scene_im, vici_np)
    for crop_scale in range(args.crop_scale_min, args.crop_scale_max, 0.1):
        x, y, err = conv(crop_np, crop_scale, comp_im)

if __name__ == '__main__':
    args = parse_args()
    main(args)
