# NTIRE 2025 – Face Restoration Challenge
## Team: #3 MaDENN

---

## Environment Setup

We use the **same environment** as the provided test baseline.

**Requirements:**
- Python 3.8
- CUDA 11.7
- PyTorch 1.13.1

### 1. Clone the repository

```bash
git clone https://github.com/AlexLefte/ntire2026_idpcodeformer.git
cd ntire2026_idpcodeformer
```

### 2. Create (if needed) and activate conda environment

```bash
conda create -n idpcodeformer python=3.8 -y
conda activate idpcodeformer
```

### 3. Install PyTorch

```bash
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 \
    --extra-index-url https://download.pytorch.org/whl/cu117
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Download Pretrained Model

The pretrained model should be placed under `model_zoo/team03_IdpCodeFormer/`.  
Run the following command from the root of the repository:

```bash
wget -i model_zoo/team03_IdpCodeFormer/team03_IdpCodeFormer.txt -P model_zoo/team03_IdpCodeFormer/
```

Alternatively, download manually from:
```
https://drive.google.com/file/d/1mfHregud3cWycOHmnjG3DzU-KDhPwWaq/view?usp=sharing
```
and place the file under `model_zoo/team03_IdpCodeFormer/`.

---

## Inference

```bash
CUDA_VISIBLE_DEVICES=0 python test.py --test_dir [path to test data dir] --save_dir [path to your save dir] --model_id 3
```

---

