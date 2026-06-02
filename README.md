# ZS-HPD

Official PyTorch implementation of **ZS-HPD**, a zero-shot image denoising method for synthetic and real-world noise. This repository contains the code used for the CVPR 2026 submission/release.

The method optimizes a lightweight image-specific denoising network at test time. No pretrained checkpoint is required.

## Features

- Zero-shot denoising for synthetic noise and real noisy images
- Per-image optimization with a compact CNN
- Content-aware downsampling pairs for self-supervised training
- Random pixel-bank sampling for pseudo noisy pair construction
- PSNR/SSIM logging and optional image saving

## Repository Structure

```text
ZS-HPD/
|-- denoise_syn.py        # synthetic-noise denoising entry
|-- denoise_real.py       # real-world denoising entry
|-- model.py              # denoising network and loss functions
|-- tools.py              # downsampling and pixel-bank utilities
|-- utils.py              # noise generation, logging, SSIM utilities
|-- config.toml           # config for synthetic-noise experiments
|-- config_real.toml      # config for real-world experiments
`-- datasets/             # dataset root, not included by default
```

## Environment

The code was developed with Python 3.9. A typical setup is:

```bash
conda create -n zs-hpd python=3.9 -y
conda activate zs-hpd

pip install torch torchvision
pip install numpy pillow toml natsort kornia
```

Install the CUDA-enabled PyTorch build that matches your machine if GPU acceleration is needed. See the official PyTorch installation page for the correct `torch` and `torchvision` command.

## Data Preparation

### Synthetic Noise

For synthetic-noise experiments, place clean images in one folder:

```text
datasets/
`-- Kodak24/
    |-- image_001.png
    |-- image_002.png
    `-- ...
```

Set the folder in `config.toml`:

```toml
[paths]
dataset_path = "./datasets/Kodak24"
```

Noise is generated on the fly according to the `[noise]` section.

### Real-World Noise

For real-world experiments, place paired noisy/ground-truth images in two folders:

```text
datasets/
`-- Realworld/
    `-- SIDD/
        |-- GT/
        |   |-- image_001.png
        |   `-- ...
        `-- Noisy/
            |-- image_001.png
            `-- ...
```

Set the folders in `config_real.toml`:

```toml
[paths]
dataset_gt_path = "./datasets/Realworld/SIDD/GT"
dataset_noisy_path = "./datasets/Realworld/SIDD/Noisy"
```

Images are loaded in natural sorted order, so the GT and noisy folders should contain matching image sequences.

Supported image formats are `.png`, `.jpg`, `.jpeg`, `.tif`, and `.tiff`. Images are center-cropped to `256 x 256` before denoising.

## Running

### Synthetic Noise

```bash
python denoise_syn.py --config config.toml
```

By default, `config.toml` uses Gaussian noise and writes results to:

```text
result/<experiment_name>/
|-- noisy/
|-- denoised/
`-- <experiment_name>_<timestamp>.log
```

### Real-World Noise

```bash
python denoise_real.py --config config_real.toml
```

By default, `config_real.toml` writes results to:

```text
real_results/<experiment_name>/
|-- noisy/
|-- denoised/
`-- <experiment_name>_<timestamp>.log
```

If you are using PowerShell and `conda activate` is unavailable, you can also run with:

```bash
conda run -n zs-hpd python denoise_syn.py --config config.toml
conda run -n zs-hpd python denoise_real.py --config config_real.toml
```

## Configuration

Important options in `config.toml` and `config_real.toml`:

| Section | Option | Description |
| --- | --- | --- |
| `paths` | `name` | Experiment name. Existing result folders are archived automatically. |
| `paths` | `results_path` | Output directory. |
| `paths` | `is_save` | Whether to save noisy and denoised images. |
| `device` | `type` | `auto`, `cuda`, or `cpu`. |
| `noise` | `type` | Synthetic noise type: `gaussian`, `poisson`, `saltpepper`, `bernoulli`, `impulse`, or `salt and gauss`. |
| `noise` | `level` | Noise level for synthetic experiments. |
| `training` | `max_epoch` | Number of optimization steps per image. |
| `training` | `lr` | Learning rate. |
| `training` | `patch_size` | Patch size for pixel-bank construction. |
| `training` | `M` | Number of sampled candidates used for pseudo image construction. |
| `training` | `sigma` | Random candidate sampling scale. |
| `training` | `similarity_metric` | Patch matching metric, e.g. `l1` or `l2`. |
| `epoch` | `run_epoch` | Number of repeated runs. |

## Outputs

Each run logs:

- per-image processing time
- initial noisy PSNR
- final denoised PSNR
- final SSIM
- average PSNR/SSIM summary

When `is_save = true`, denoised images are written under the experiment output directory.

## Notes

- The model is trained from scratch for every test image.
- Full runs can be slow on CPU; CUDA is recommended.
- The current scripts use batch size 1.
- For real-world evaluation, noisy and GT folders must be aligned.

## Citation

If you find this project useful, please cite our paper:

```bibtex
@inproceedings{zshpd2026,
  title     = {Zero-Shot Image Denoising via Hybrid Prior-Guided Pseudo Sample Generation},
  author    = {Xiaole Zhao, Qingsong Pang, Xiaobo Zhang, Xun Xu, Xun Gong, Yan Yang, Tianrui Li},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year      = {2026}
}
```

The citation entry will be updated after the paper metadata is released.
