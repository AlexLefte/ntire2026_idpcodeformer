import os
import cv2
import gdown
import numpy as np
import torch


def img2tensor(img, bgr2rgb=True, float32=True):
    """Numpy array (H x W x C) to tensor (C x H x W)."""
    if img.shape[2] == 3 and bgr2rgb:
        if img.dtype == 'float64':
            img = img.astype('float32')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(img.transpose(2, 0, 1))
    if float32:
        img = img.float()
    return img


def tensor2img(tensor, rgb2bgr=True, out_type=np.uint8, min_max=(-1, 1)):
    """Tensor (C x H x W) to numpy array (H x W x C)."""
    _tensor = tensor.squeeze(0).float().detach().cpu().clamp_(*min_max)
    _tensor = (_tensor - min_max[0]) / (min_max[1] - min_max[0])

    img_np = _tensor.numpy().transpose(1, 2, 0)
    if rgb2bgr:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    if out_type == np.uint8:
        img_np = (img_np * 255.0).round()

    return img_np.astype(out_type)


def imwrite(img, file_path):
    """Write image."""
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
    ok = cv2.imwrite(file_path, img)
    if not ok:
        raise IOError(f'Failed to write image: {file_path}')
    

def download_checkpoint(model_dir, ckpt_filename):
    """Download checkpoint if not present locally."""
    model_path = os.path.join(model_dir, ckpt_filename)
    if os.path.exists(model_path):
        return model_path
    
    print("did not return")
    # Read .txt file and dowmload from GDrive
    txt_path = os.path.join(model_dir, 'team03_IdpCodeFormer.txt')
    with open(txt_path, 'r') as f:
        url = f.read().strip()
    
    print(f'Downloading checkpoint from {url}...')
    gdown.download(url, model_path, fuzzy=True)
    return model_path