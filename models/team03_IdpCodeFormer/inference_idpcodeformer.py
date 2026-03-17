import os
import cv2
import argparse
import glob
import torch
import numpy as np

from .networks.codeformer_nets import CodeFormer
from .utils import img2tensor, tensor2img, imwrite, download_checkpoint
from torchvision.transforms.functional import normalize


def main(model_dir, input_path=None, output_path=None, device=None):
    # ------------------------ set up parameters ------------------------    
    print(f"Model dir: {model_dir}")
    print(f"Input path: {input_path}")
    print(f"Output path: {output_path}")

    target = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'weights')
    if os.path.exists(target):
        print(f"{target} already exists. Removing it...")
        os.remove(target)
    os.symlink(os.path.abspath(model_dir), target)
    # target = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'weights')
    # if os.path.exists(target) or os.path.islink(target):
    #     print(f"{target} already exists. Removing it...")
    #     os.unlink(target)
    # os.symlink(os.path.abspath(model_dir), target, target_is_directory=True) 
    print(f"Create a symbolic link from {os.path.abspath(model_dir)} to {target}. ")

    # ------------------------ checkpoint ------------------------
    ckpt_filename = 'idpcodeformer.pth'
    model_path = download_checkpoint(model_dir, ckpt_filename)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f'Checkpoint not found: {model_path}')
 
    # ------------------------ Collect image paths ------------------------
    if input_path.endswith(('jpg', 'jpeg', 'png', 'JPG', 'JPEG', 'PNG')):
        input_img_list = [input_path]
    else:
        if input_path.endswith('/'):
            input_path = input_path[:-1]
        input_img_list = sorted(glob.glob(os.path.join(input_path, '*.[jpJP][pnPN]*[gG]')))
 
    if len(input_img_list) == 0:
        raise FileNotFoundError(f'No image found in: {input_path}')
 
    result_root = output_path if output_path is not None else f'results/{os.path.basename(input_path)}_{w}'
    os.makedirs(result_root, exist_ok=True)
 
    # ------------------------ Load the model ------------------------
    net = CodeFormer(
        dim_embd=512, codebook_size=1024, n_head=8, n_layers=9,
        connect_list=['32', '64', '128', '256']
    ).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    net.load_state_dict(checkpoint)
    net.eval()
 
    # ------------------------ Process and predict ------------------------
    test_img_num = len(input_img_list)
    for i, img_path in enumerate(input_img_list):
        img_name = os.path.basename(img_path)
        basename, _ = os.path.splitext(img_name)
        print(f'[{i+1}/{test_img_num}] Processing: {img_name}')
 
        # Read input image
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
 
        # Pre-process
        img_t = img2tensor(img / 255., bgr2rgb=True, float32=True)
        normalize(img_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
        img_t = img_t.unsqueeze(0).to(device)
 
        # Infer
        try:
            with torch.no_grad():
                output = net(img_t, w=0.5, adain=True)[0]
                restored = tensor2img(output, rgb2bgr=True, min_max=(-1, 1))
            del output
            torch.cuda.empty_cache()
        except Exception as error:
            print(f'\tFailed inference: {error}')
            restored = tensor2img(img_t, rgb2bgr=True, min_max=(-1, 1))
 
        # Save images
        save_path = os.path.join(result_root, f'{basename}.png')
        imwrite(restored.astype('uint8'), save_path)
 
    print(f'\nAll results are saved in {result_root}')