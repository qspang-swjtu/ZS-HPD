import argparse
import logging
import numpy as np
import os
import time
import toml
import torch
import torch.optim as optim
import torchvision.transforms as transforms
from natsort import natsorted
from os import path as osp
from torch.optim.swa_utils import AveragedModel
from PIL import Image
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image

from tools import random_noisy,content_aware_downsample_larger_region,build_random_blank
from model import network, zsn2n_cross_loss, up_loss
from utils import  dict2str, mkdir_and_rename, get_time_str, ssim_torch


# --- Configuration Loading ---
def load_config(config_path):
    """Loads configuration from a TOML file."""
    try:
        with open(config_path, 'r') as f:
            config = toml.load(f)
        # Basic validation
        required_sections = ['paths', 'device', 'training']
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


def mse(gt: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    loss = torch.nn.MSELoss()
    return loss(gt, pred)


# --- Argument Parser (Now only for config file path) ---
def parse_cli_args():
    """Parses command-line arguments, primarily for the config file path."""
    parser = argparse.ArgumentParser(
        description="Load config and train Noise2Void-like model.")
    parser.add_argument(
        '--config',
        type=str,
        default='config_real.toml',  # Default config file name
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
        self.CenterCrop = transforms.CenterCrop(size= (256,256))

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
def train_pipeline(model, optimizer, D1, D2, up_D1, up_D2):
    model.train()

    loss_zs = zsn2n_cross_loss(model, D1, D2)
    loss_up = up_loss(model, up_D1, up_D2)
    
    loss =   loss_up + 1/6 * loss_zs 
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()
def test_pipeline(model, noisy_img, clean_img):
    model.eval()
    with torch.no_grad():
        pred_noise = model(noisy_img)
        denoised_img = torch.clamp(noisy_img- pred_noise, 0, 1)
        mse_val = mse(clean_img, denoised_img).item()
        psnr_val = 10 * np.log10(1 /
                                 mse_val) if mse_val > 1e-10 else float('inf')
        ssim = ssim_torch(clean_img, denoised_img).item()
    return psnr_val, denoised_img, ssim


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
        logging.info(f"Images will be saved to: {results_path}")
    else:
        logging.info(
            "No images will be saved. Set 'is_save' to True in config to enable saving."
        )

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
    dataset_gt_path = config['paths']['dataset_gt_path']
    dataset_noisy_path = config['paths']['dataset_noisy_path']
    try:
        dataset_gt = ImageDataset(root_dir=dataset_gt_path)
        dataset_noisy = ImageDataset(dataset_noisy_path)
        dataloader_gt = DataLoader(dataset_gt,
                                   batch_size=1,
                                   shuffle=False,
                                   collate_fn=collate_fn)
        dataloader_noisy = DataLoader(dataset_noisy,
                                      batch_size=1,
                                      shuffle=False,
                                      collate_fn=collate_fn)

        logging.info(
            f"Found {len(dataset_gt.image_files)} images in {dataset_noisy_path}"
        )
        logging.info(
            f"Will attempt to process {len(dataloader_noisy)} images.")
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

    max_epoch = config['training']['max_epoch']
    lr = config['training']['lr']
    M = config['training']['M']
    patch_size = config['training']['patch_size']
    num_random_candidates_per_anchor = config['training']['num_random_candidates_per_anchor']
    anchor_processing_batch_size = config['training']['anchor_processing_batch_size']
    similarity_metric = config['training']['similarity_metric']
    sigma = config['training']['sigma']
    total_training_time = 0
    all_initial_psnrs = []
    all_final_psnrs = []
    processed_image_count = 0
    processed_filenames = []
    all_final_ssim = []

    # --- Loop over dataset ---
    logging.info("Beginning processing of images...")
    for (clean_batch, noisy_batch) in zip(dataloader_gt, dataloader_noisy):
        clean_img, clean_filename = clean_batch
        noisy_img, noisy_filename = noisy_batch
        if clean_img is None:
            # filename might be None too if the entire batch failed collation
            item_identifier = clean_filename or noisy_filename
            logging.warning(
                f"Skipping {item_identifier} due to image loading/processing error in dataset or collation."
            )
            continue

        clean_img = clean_img.to(device)
        image_start_time = time.time()

        logging.info(
            f"--- Processing image {processed_image_count + 1}/{len(dataset_noisy.image_files)}: {noisy_filename} ---"
        )

        # Add noise
        noisy_img = noisy_img.to(device)
        D1 , D2 ,D3 ,D4 = content_aware_downsample_larger_region(noisy_img)
        n_chan = clean_img.shape[1]
        model = network(n_chan).to(device)
        
        
        swa_start_epoch = 1500
        swa_lr = lr * 0.1
        swa_model = AveragedModel(model)

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
        
        
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = MultiStepLR(optimizer , milestones = [500,1000] , gamma = 0.5)
        for epoch in range(max_epoch):
            up_D1 ,up_D2 = random_noisy(noisy_img ,pixel_banks , M = M)
            if epoch % 2 == 0:
                train_pipeline(model, optimizer, D1, D2, up_D1, up_D2)
            else :
                train_pipeline(model, optimizer, D3, D4, up_D1, up_D2)
            
            if (epoch+1) == 700:
                first_psnr, _, first_ssim = test_pipeline(model, noisy_img, clean_img)
                logging.info(f"第一阶段psrn {first_psnr:.4f}, ssim:{first_ssim:.4f}")
            if (epoch+1) == 1200:
                sec_psnr, _, sec_ssim = test_pipeline(model, noisy_img, clean_img)
                logging.info(f"第二阶段psrn {sec_psnr:.4f}, ssim:{sec_ssim:.4f}")
            if (epoch+1) >= swa_start_epoch:
                if (epoch+1) == swa_start_epoch:
                    
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = swa_lr
                swa_model.update_parameters(model)
                    
            else:
                scheduler.step()
        # Test the trained model
        final_psnr,denoised_img,ssim = test_pipeline(model, noisy_img, clean_img)
        image_end_time = time.time()
        image_train_time = image_end_time - image_start_time
        total_training_time += image_train_time

        all_initial_psnrs.append(psnr_noisy)
        all_final_psnrs.append(final_psnr)
        all_final_ssim.append(ssim)
        processed_image_count += 1
        processed_filenames.append(noisy_filename)

        logging.info(
            f"Finished processing {noisy_filename}. Time: {image_train_time:.2f} sec, "
            f"Initial PSNR: {psnr_noisy:.4f} dB, Final PSNR: {final_psnr:.4f} dB ,"
             f"first_improve:{(sec_psnr - first_psnr):.2f} sec_improve:{(final_psnr - sec_psnr):.2f}"
        )
                  

        # Save the noisy and denoised images if required
        if is_save:
            base, ext = osp.splitext(noisy_filename)
            noisy_save_name = f"noisy_{base}{ext}"
            denoised_save_name = f"{base}_{final_psnr:.2f}_{ssim:.4f}{ext}"
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
        f"Processed {processed_image_count} images successfully out of {len(dataset_noisy.image_files)} found."
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
            f"Average PSNR Improvement: {average_final_psnr - average_initial_psnr:.4f} dB",
        )
        logging.info(f"Average ssim : {average_final_ssim:.4f} ")
    else:
        logging.info("No images were processed successfully.")

    logging.info(
        f"Total processing time for successful images: {total_training_time:.2f} seconds"
    )
    if processed_image_count > 0:
        logging.info(
            f"Average processing time per image: {total_training_time / processed_image_count:.2f} seconds"
        )

    original_files = set(dataset_noisy.image_files)
    processed_files = set(processed_filenames)
    skipped_or_failed_files = natsorted(list(original_files - processed_files))
    if skipped_or_failed_files:
        logging.info("Files skipped or failed during processing:")
        for fname in skipped_or_failed_files:
            logging.info(f"  - {fname}")
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

    print(
        f"Average Final PSNR (denoised): {np.mean(average_final_psnr):.4f} dB",
        f"Max Final PSNR (denoised): {np.max(average_final_psnr):.4f} dB")
    print(f"Average Final SSIM (denoised): {np.mean(average_final_ssim):.4f} ",
          f"Max Final SSIM (denoised): {np.max(average_final_ssim):.4f} ")
