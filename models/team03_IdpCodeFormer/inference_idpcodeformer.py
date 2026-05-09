import os
import cv2
import argparse
import glob
import torch
import time
import numpy as np

from .networks.codeformer_nets import CodeFormer
from .utils import img2tensor, tensor2img, imwrite
from torchvision.transforms.functional import normalize


def main(model_dir, input_path=None, output_path=None, device=None):
    # ------------------------ checkpoint ------------------------
    ckpt_filename = 'idpcodeformer.pth'
    model_path = os.path.join(model_dir, ckpt_filename)
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
 
    result_root = output_path if output_path is not None else f'results/{os.path.basename(input_path)}'
    os.makedirs(result_root, exist_ok=True)
 
    # ------------------------ Load the model ------------------------
    net = CodeFormer(
        dim_embd=512, codebook_size=1024, n_head=8, n_layers=9,
        connect_list=['32', '64', '128', '256']
    ).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    if 'params_ema' in checkpoint:
        checkpoint = checkpoint['params_ema']
    net.load_state_dict(checkpoint)
    net.eval()

    # Set w_scale
    w_scale = 0.5
    use_adain = False
 
    # Get model FLOPS and inference time
    from utils.model_summary import get_model_flops
    def codeformer_input(input_res):
        return {'x': torch.FloatTensor(1, *input_res).to(device), 'w': w_scale, 'adain': use_adain}

    flops = get_model_flops(net, input_res=(3, 512, 512), print_per_layer_stat=False, input_constructor=codeformer_input)
    print(f"FLOPs: {flops / 1e9:.2f} G")

    # ------------------------ Process and predict ------------------------
    total_time = 0
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
                torch.cuda.synchronize()
                t_start = time.time()
                output = net(img_t, w=w_scale, adain=use_adain)[0]
                torch.cuda.synchronize()
                total_time += time.time() - t_start
                restored = tensor2img(output, rgb2bgr=True, min_max=(-1, 1))
            del output
            torch.cuda.empty_cache()
        except Exception as error:
            print(f'\tFailed inference: {error}')
            restored = tensor2img(img_t, rgb2bgr=True, min_max=(-1, 1))
 
        # Save images
        save_path = os.path.join(result_root, f'{basename}.png')
        imwrite(restored.astype('uint8'), save_path)
 
    avg_time = total_time / test_img_num
    print(f'\nAvg inference time: {avg_time * 1000:.2f} ms ({1/avg_time:.2f} FPS)')
    print(f'\nAll results are saved in {result_root}')