import os
import subprocess
from tqdm import tqdm

dry_run = False

def run(batch_name):
    scene_path = os.path.join('./data', batch_name + '.jpg')
    assert os.path.exists(scene_path)
    mask_path = os.path.join('./data', batch_name + '_mask.jpg')
    assert os.path.exists(mask_path)
    save_dir = os.path.join('./output', batch_name)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    for idx in tqdm(range(20)):
        comp_path = os.path.join('./data', batch_name, 'result_img0{:02d}.jpg'.format(idx + 1))
        if not os.path.exists(comp_path):
            continue
        save_path = os.path.join('./output', batch_name, 'out_{:02d}.jpg'.format(idx + 1))
        subprocess.Popen(["python", "main.py", "--scene", scene_path, "--comp", comp_path, "--mask", mask_path, "--save", save_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout.read()
        if dry_run:
            break

if not os.path.exists('output'):
    os.mkdir('output')

inputs = ['input1', 'input2', 'input3', 'input4', 'input5']

for i in tqdm(inputs):
    run(i)
    if dry_run:
        break