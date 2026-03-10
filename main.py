import argparse
import logging
import numpy as np
import os
import time
import toml
import torch.nn.functional as F
import torch
import torch.nn as nn
import math
import torch.optim as optim
import torchvision.transforms as transforms
from natsort import natsorted
from os import path as osp
from torch.optim.swa_utils import AveragedModel, SWALR
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image

from test import random_noisy,content_aware_downsample_larger_region,build_random_blank

from model import network, zsn2n_cross_loss, up_loss
from utils import add_noise, dict2str, mkdir_and_rename, get_time_str, ssim_torch 

from torch.optim.lr_scheduler import MultiStepLR
def mse(gt: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    loss = torch.nn.MSELoss()
    return loss(gt, pred)

# --- Configuration Loading ---
def load_config(config_path):
    """Loads configuration from a TOML file."""
    try:
        with open(config_path, 'r') as f:
            config = toml.load(f)
        # Basic validation
        required_sections = ['paths', 'device', 'noise', 'training']
        for section in required_sections:
            if section not in config:
                raise ValueError(
                    f"Missing required section '{section}' in config file.")
        return config
    except FileNotFoundError:
        logging.error(f"Configuration file not found at: {config_path}")
        return None
    except toml.TomlDecodeError as e:
        logging.error(f"Error decoding TOML file {config_path}: {e}")
        return None
    except ValueError as e:
        logging.error(f"Configuration error: {e}")
        return None
    except Exception as e:
        logging.error(
            f"An unexpected error occurred while loading config: {e}")
        return None
    
class DifferentiableRadius(nn.Module):
    def __init__(self, init_ratio=0.2, min_ratio=0.1, max_ratio=0.5):
        """
        init_ratio: 初始半径比例 (0-1)
        min_ratio: 最小半径比例 (防止消失)
        max_ratio: 最大半径比例
        """
        super().__init__()
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        
        # 初始化参数确保在合理范围内
        initial_value = torch.tensor(init_ratio)
        self.log_ratio = nn.Parameter(torch.log(initial_value / (1 - initial_value + 1e-8)))
    
    @property
    def radius_ratio(self):
        # 使用 sigmoid + 缩放确保值在 [min_ratio, max_ratio] 范围
        sigmoid_ratio = torch.sigmoid(self.log_ratio)
        return self.min_ratio + (self.max_ratio - self.min_ratio) * sigmoid_ratio
    
    def forward(self):
        return self.radius_ratio

# --- Argument Parser (Now only for config file path) ---
def parse_cli_args():
    """Parses command-line arguments, primarily for the config file path."""
    parser = argparse.ArgumentParser(
        description="Load config and train Noise2Void-like model.")
    parser.add_argument(
        '--config',
        type=str,
        default='config.toml',  # Default config file name
        help='Path to the TOML configuration file (default: config.toml)')
    return parser.parse_args()


# --- Dataset Definition ---
class ImageDataset(Dataset):

    def __init__(self, root_dir):
        self.root_dir = root_dir
        all_files = os.listdir(root_dir)
        self.supported_formats = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')
        image_files_unsorted = [
            f for f in all_files if osp.isfile(osp.join(root_dir, f))
            and f.lower().endswith(self.supported_formats)
        ]
        self.image_files = natsorted(image_files_unsorted)
        if not self.image_files:
            raise FileNotFoundError(
                f"No supported image files ({', '.join(self.supported_formats)}) found in {root_dir}"
            )
        self.transform = transforms.ToTensor()
        self.CenterCrop = transforms.CenterCrop(size = (256,256))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = osp.join(self.root_dir, img_name)
        try:
            img = Image.open(img_path)
            img_tensor = self.transform(img)
            img_tensor = self.CenterCrop(img_tensor)
            return img_tensor, img_name
        except Exception as e:
            if logging.getLogger().hasHandlers():
                logging.error(
                    f"Error loading or processing image {img_path}: {e}")
            else:
                print(f"Error loading or processing image {img_path}: {e}")
            return None, None
            
class NoiseDataset(Dataset):

    def __init__(self, noisy_img):
        super().__init__()
        self.noisy_img = noisy_img

    def __len__(self):
        return 1

    def __getitem__(self, index):
        return self.noisy_img
# --- Collate Function ---
def collate_fn(batch):
    filtered_batch = list(filter(lambda x: x[0] is not None, batch))
    assert len(filtered_batch) <= 1, "Batch size should be 1 for this dataset."
    if not filtered_batch:
        return None, None  # Return None pair to indicate failure to the main loop
    tensor, filename = filtered_batch[0]
    # Add the batch dimension: (C, H, W) -> (1, C, H, W)
    tensor_batch = tensor.unsqueeze(0)
    return tensor_batch, filename


    
# --- Training & Testing Functions ---
def train_pipeline(model, optimizer, D1, D2,noisy_img, lam_1,up_D1,up_D2):
    model.train()
    loss_zs = zsn2n_cross_loss(model, D1, D2, noisy_img, lam_1)
    loss_up = up_loss(model ,up_D1,up_D2 , lam_1) 
    
    loss = loss_zs + loss_up
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

# def train_pipeline(model, optimizer, D1, D2,noisy_img, lam_1,lam_2,up_D1,up_D2):
#     model.train()

#     loss_zs = zsn2n_cross_loss(model, D1, D2, noisy_img, lam_1,lam_2)
#     loss_up = up_loss(model ,up_D1,up_D2 , lam_1) 
    
#     loss =   loss_up + loss_zs
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
#     return loss.item()
    
# def train_pipeline(model, optimizer,D1,D2,D3,D4,mask1,mask2,up_D3,up_D4, noisy_img):
#     model.train()
#     loss = frequency_domain_loss_2(model,D1,D2,D3,D4,mask1,mask2,up_D3,up_D4, noisy_img)
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
#     return loss.item()


# def bsn_train(model ,optimizer, noisy , D1 ,D2  ,up_D1 ,up_D2 , step ,k):
#     model.train()
#     D1_s2s ,D1_mask = get_gridded_data_neighbor_replace_v2(D1,step )
#     D2_s2s , D2_mask = get_gridded_data_neighbor_replace_v2(D2,step )
#     blit_up_D1,up_D1_mask = get_gridded_data_neighbor_replace_v2(up_D1 ,step)
#     D1_pred = model(D1 )
#     D2_pred = model(D2)
#     up_pred = model(up_D1)
    
#     # global_D1 = model(D1 )
#     # global_D2 = model(D2 , 'global')
#     # global_up_D1 = model(up_D1 , 'global')
#     loss_1 =1/2 * ( mse(D1_pred  ,D2 ) + mse(D2_pred  ,D1 ))
#     loss_up =1/2 * (mse(up_pred , up_D2 ))
    
#     # global_res =(loss_1 + loss_up) + 1/2 * (mse(global_D1 , D2) + mse(global_D2 , D1))
#     # global_up_res = (mse(global_up_D1 , up_D2))
    
#     loss = (loss_1+loss_up )
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
#     return loss.item()
    
def test_psnr(pred, gt):
    with torch.no_grad():
        mse_val = mse(pred, gt).item()
        psnr_val = 10 * np.log10(1 /
                                 mse_val) if mse_val > 1e-10 else float('inf')
        ssim = ssim_torch(gt, pred).item()
        return psnr_val,ssim


def denoised(model, noisy_img):
    model.eval()
    with torch.no_grad():
        denoised = noisy_img - model(noisy_img)

    return denoised
    
def test_pipeline(model, noisy_img, clean_img):
    model.eval()
    with torch.no_grad():
        pred_noise =  model(noisy_img)
        denoised_img = torch.clamp(noisy_img -pred_noise, 0, 1)
        mse_val = mse(clean_img, denoised_img).item()
        psnr_val = 10 * np.log10(1 /
                                 mse_val) if mse_val > 1e-10 else float('inf')
        ssim = ssim_torch(clean_img, denoised_img).item()
    return psnr_val, denoised_img, ssim ,pred_noise

# def test_pipeline(model, noisy_img, clean_img):
#     model.eval()
#     with torch.no_grad():
#         pred_noise = model(noisy_img )
#         denoised_img = torch.clamp( pred_noise, 0, 1)
#         mse_val = mse(clean_img, denoised_img).item()
#         psnr_val = 10 * np.log10(1 /
#                                  mse_val) if mse_val > 1e-10 else float('inf')
#         ssim = ssim_torch(clean_img, denoised_img).item()
#     return psnr_val, denoised_img, ssim , noisy_img - pred_noise

# --- Main Training Orchestration ---
def main(config, config_path):
    """ Main function using configuration dictionary """
    # --- Setup Logging ---
    experiment_name = config['paths']['name']
    results_path = config['paths']['results_path']
    is_save = config['paths']['is_save']

    mkdir_and_rename(osp.join(results_path, experiment_name))
    if is_save:
        os.makedirs(osp.join(results_path, experiment_name, 'noisy'),
                    exist_ok=True)
        os.makedirs(osp.join(results_path, experiment_name, 'denoised'),
                    exist_ok=True)
        # os.makedirs(osp.join(results_path, experiment_name, 'denoised_first'),
        #             exist_ok=True)
        # os.makedirs(osp.join(results_path, experiment_name, 'denoised_sec'),
        #             exist_ok=True)
        os.makedirs(osp.join(results_path, experiment_name, 'pred_noisy'),
                    exist_ok=True)
        logging.info(f"Images will be saved to: {results_path}")
    else:
        logging.info(
            "No images will be saved. Set 'is_save' to True in config to enable saving."
        )
    # run = wandb.init(
    #     # Set the wandb entity where your project will be logged (generally your team name).
    #     entity="yuyumao99-southwest-jiaotong-university",
    #     name=time.strftime('%m%d%H%M%S'),
    #     # Set the wandb project where this run will be logged.
    #     project="ZS-N2N",
    #     # Track hyperparameters and run metadata.
    #     config={
    #         "learning_rate": 0.0005,
    #         "architecture": "ZS-N2N",
    #         "dataset": "./datasets\Gaussion\Kodak24",
    #         "epochs": 100,
    #     },
    # )
    log_file_path = osp.join(results_path, experiment_name,
                             f"{experiment_name}_{get_time_str()}.log")
    os.makedirs(osp.dirname(log_file_path), exist_ok=True)

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[
                            logging.FileHandler(log_file_path, mode='w'),
                            logging.StreamHandler()
                        ])

    logging.info(f"Successfully loaded configuration from: {config_path}")
    logging.info(f"Configuration:{dict2str(config)}")

    # --- Load Dataset ---
    dataset_path = config['paths']['dataset_path']
    try:
        dataset = ImageDataset(root_dir=dataset_path)
        dataloader = DataLoader(dataset,
                                batch_size=1,
                                shuffle=False,
                                collate_fn=collate_fn)
        logging.info(
            f"Found {len(dataset.image_files)} images in {dataset_path}")
        logging.info(f"Will attempt to process {len(dataloader)} images.")
    except FileNotFoundError as e:
        logging.error(f"Dataset error: {e}")
        return
    except Exception as e:
        logging.error(
            f"An unexpected error occurred during dataset loading: {e}")
        return

    # --- Device Selection ---
    device_type_req = config['device']['type'].lower()
    if device_type_req == 'auto':
        device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
    elif device_type_req == 'cuda':
        if not torch.cuda.is_available():
            logging.warning(
                "CUDA specified in config but not available. Falling back to CPU."
            )
            device_type = 'cpu'
        else:
            device_type = 'cuda'
    else:
        device_type = 'cpu'
    device = torch.device(device_type)

    logging.info(f"Starting experiment: {experiment_name}")
    logging.info(f"Using device: {device}")

    # --- Extract noise and training params from config ---
    noise_type = config['noise']['type']
    noise_level = config['noise']['level']
    max_epoch = config['training']['max_epoch']
    lr = config['training']['lr']
    step_size = config['training']['step_size']
    gamma = config['training']['gamma']
    M = config['training']['M']
    patch_size = config['training']['patch_size']

    train_log = {}
    train_log['lr'] = lr
    train_log['epoch'] = max_epoch
    # --- Log noise type ONCE before the loop ---
    if noise_type == 'gaussian':
        logging.info(
            f"Noise type: Gaussian, Level: {noise_level} (0-255 scale, scaling handled by add_noise)\n"
        )
    elif noise_type == 'poisson':
        logging.info(f"Noise type: Poisson, Level/Intensity: {noise_level}\n")
    else:
        logging.warning(
            f"Unknown noise type specified: {noise_type}. Proceeding with level {noise_level}.\n"
        )

    # Prepare results storage
    results = {}

    total_training_time = 0
    all_initial_psnrs = []
    all_final_psnrs = []
    all_final_ssim = []
    processed_image_count = 0
    processed_filenames = []
    
    # --- Loop over dataset ---
    logging.info("Beginning processing of images...")
    for i, batch_data in enumerate(dataloader):

        clean_img, filename = batch_data
        if clean_img is None:
            # filename might be None too if the entire batch failed collation
            item_identifier = dataset.image_files[i] if i < len(
                dataset.image_files) else f"item index {i}"
            logging.warning(
                f"Skipping {item_identifier} due to image loading/processing error in dataset or collation."
            )
            continue

        clean_img = clean_img.to(device)
        

        logging.info(
            f"--- Processing image {processed_image_count + 1}/{len(dataset.image_files)}: {filename} ---"
        )

        # Add noise
 
        noisy_img = add_noise(clean_img, noise_type, noise_level)
        noisy_img = noisy_img.to(device)
        
        num_random_candidates_per_anchor = config['training']['num_random_candidates_per_anchor']
        anchor_processing_batch_size = config['training']['anchor_processing_batch_size']
        similarity_metric = config['training']['similarity_metric']
        sigma = config['training']['sigma']
        image_start_time = time.time()
        
        D1 , D2 ,D3 ,D4 = content_aware_downsample_larger_region(noisy_img)

        if noise_level > 25 and noise_type == 'gaussian' :
            sigma = 15
            M = 8
        elif noise_level == 25  and noise_type == 'gaussian':
            sigma = 10
            M = 8
        elif noise_level < 25 and noise_type == 'gaussian' :
            sigma = 10
            M = 4
        pixel_banks = build_random_blank(noisy_img , patch_size , M , 
                                         num_random_candidates_per_anchor =num_random_candidates_per_anchor,
                                         similarity_metric = similarity_metric,
                                        anchor_processing_batch_size = anchor_processing_batch_size,
                                        sigma = sigma ,
                                         device = noisy_img.device)


        # Calculate PSNR before denoising
        with torch.no_grad():
            mse_noisy = mse(clean_img, noisy_img).item()
            psnr_noisy = 10 * np.log10(
                1 / mse_noisy) if mse_noisy > 1e-10 else float('inf')
        
        # Initialize model and optimizer for each image
        B,n_chan ,H ,W = clean_img.shape
        
        model = network(n_chan ).to(device)
        optimizer = optim.AdamW((model.parameters()), lr=lr)
        scheduler = MultiStepLR(optimizer, milestones = [500,1000] , gamma = 0.5)
 
        noisy_clone = noisy_img.clone()
        noisy_ = noisy_clone.squeeze(0).permute(1 , 2 , 0).cpu().numpy()

        for epoch in range(max_epoch):
            up_D1 ,up_D2 = random_noisy(noisy_img ,pixel_banks ,M = M)
            if epoch % 2 == 0:
                train_pipeline(model, optimizer, D1, D2, noisy_img, epoch + 1,  up_D1, up_D2)
            else :
                train_pipeline(model, optimizer, D3, D4, noisy_img, epoch + 1,  up_D1, up_D2)
            if (epoch + 1) % 1000 == 0:
                with torch.no_grad():
                    loss_up = up_loss(model ,up_D1,up_D2 ,epoch+1 ).item()
                    loss_toal = zsn2n_cross_loss(model, D1, D2,noisy_img, epoch+1).item() +loss_up
                logging.info(f"Image {filename}, Epoch {epoch+1}/{max_epoch}, ,zs_loss: {loss_toal:.4f}  ,up_loss:{loss_up:.4f} ,lam_1 :{lam_1}")

            scheduler.step()  # 主训练阶段的学习率调度

        # Test the trained model
        final_psnr, denoised_img, ssim ,pred_noisy= test_pipeline(model, noisy_img,
                                                       clean_img)

        train_log['psnr'] = final_psnr
        image_end_time = time.time()
        image_train_time = image_end_time - image_start_time
        total_training_time += image_train_time

        all_initial_psnrs.append(psnr_noisy)
        all_final_psnrs.append(final_psnr)
        all_final_ssim.append(ssim)
        results[filename] = {
            'training_time_sec': image_train_time,
            'initial_psnr': psnr_noisy,
            'final_psnr': final_psnr,
            'ssim': ssim
        }
        processed_image_count += 1
        processed_filenames.append(filename)

        logging.info(
            f"Finished processing {filename}. Time: {image_train_time:.2f} sec, "
            f"Initial PSNR: {psnr_noisy:.4f} dB, Final PSNR: {final_psnr:.4f} dB, "
            f"ssim : {ssim:.4f}, "
            f"param : {sum(p.numel() for p in model.parameters() if p.requires_grad)},"
            # f"first_improve:{(sec_psnr - first_psnr):.2f} sec_improve:{(final_psnr - sec_psnr):.2f}"
        )

        # Save the noisy and denoised images if required
        if is_save:
            base, ext = osp.splitext(filename)
            if ext == '.tif':
                ext = '.png'
            noisy_save_name = f"noisy_{base}{ext}"
            denoised_save_name = f"{base}{ext}"
            noisy_save_path = osp.join(results_path, experiment_name, 'noisy',
                                       noisy_save_name)

            denoised_save_path = osp.join(results_path, experiment_name,
                                          'denoised', denoised_save_name)
 
            try:
                save_image(noisy_img.squeeze(0).cpu(), noisy_save_path)
                save_image(denoised_img.squeeze(0).cpu(), denoised_save_path)
 
                
                logging.debug(
                    f"Noisy image saved to: {noisy_save_path}, Denoised image saved to: {denoised_save_path}"
                )
            except Exception as e:
                logging.error(f"Failed to save image: {e}")

        # Explicitly delete model and optimizer to free memory before next iteration
        del model, optimizer, scheduler, denoised_img, noisy_img, clean_img
        if device_type == 'cuda':
            torch.cuda.empty_cache()

    logging.info("Finished processing all images.\n")

    # Final Summary
    logging.info("--- Experiment Summary ---")
    logging.info(f"Experiment Name: {experiment_name}")
    logging.info(
        f"Processed {processed_image_count} images successfully out of {len(dataset.image_files)} found."
    )
    if all_final_psnrs:
        average_initial_psnr = np.mean(all_initial_psnrs)
        average_final_psnr = np.mean(all_final_psnrs)
        average_final_ssim = np.mean(all_final_ssim)
        logging.info(
            f"Average Initial PSNR (noisy): {average_initial_psnr:.4f} dB")
        logging.info(
            f"Average Final PSNR (denoised): {average_final_psnr:.4f} dB")
        logging.info(
            f"Average PSNR Improvement: {average_final_psnr - average_initial_psnr:.4f} dB"
        )
        logging.info(f"Average ssim : {average_final_ssim:.4f} dB")
    else:
        logging.info("No images were processed successfully.")

    logging.info(
        f"Total processing time for successful images: {total_training_time:.2f} seconds"
    )
    if processed_image_count > 0:
        logging.info(
            f"Average processing time per image: {total_training_time / processed_image_count:.2f} seconds"
        )



    original_files = set(dataset.image_files)
    processed_files = set(processed_filenames)
    skipped_or_failed_files = natsorted(list(original_files - processed_files))
    if skipped_or_failed_files:
        logging.info("Files skipped or failed during processing:")
        for fname in skipped_or_failed_files:
            logging.info(f"  - {fname}")
    train_log['average_final_psnr'] = average_final_psnr

    return average_final_psnr, average_final_ssim


if __name__ == "__main__":
    average_final_psnr, average_final_ssim = [], []
    cli_args = parse_cli_args()
    config = load_config(cli_args.config)
    if config:
        for i in range(config['epoch']['run_epoch']):
            average_psnr, average_ssim = main(config, cli_args.config)
            average_final_psnr.append(average_psnr)
            average_final_ssim.append(average_ssim)
    else:
        print(
            "Exiting due to configuration loading errors. Check logs or console output."
        )
        exit(1)
    avg_psnr = np.mean(average_final_psnr)
    max_psnr = np.max(average_final_psnr)
    min_psnr = np.min(average_final_psnr)
    avg_ssim = np.mean(average_final_ssim)
    max_ssim = np.max(average_final_ssim)
    min_ssim = np.min(average_final_ssim)
    variance = np.var(average_final_psnr)
    logging.info(f"Average Final PSNR (denoised): {avg_psnr:.4f} dB, Max Final PSNR (denoised): {max_psnr:.4f} dB ,Min Final PSNR (denoised): {min_psnr:.4f} dB")
    logging.info(f"Average Final SSIM (denoised): {avg_ssim:.4f} var: {variance:.6f}, Max Final SSIM (denoised): {max_ssim:.4f} ,Min Final SSIM (denoised): {min_ssim:.4f}" )
