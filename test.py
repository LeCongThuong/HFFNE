#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision import transforms
import pandas as pd
from tqdm import tqdm
import numpy as np

# Import your models. Adjust these imports if your module structure differs.
from models import NormalEncoder, NormalDecoder, UNet


def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Inference script for generating refined normal predictions."
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="./ckpts/RefinerModel.pt",
        help="Path to checkpoint for NormalEncoder/NormalDecoder.",
    )
    parser.add_argument(
        "--ckpt_s1",
        type=str,
        default="./ckpts/ExemplarModel.pkl",
        help="Path to checkpoint for the pre-trained UNet model.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/mnt/hmi/thuong/Photoface_dist/PhotofaceDBLib/",
        help="Directory containing input images.",
    )
    parser.add_argument(
        "--dest_dir",
        type=str,
        default="./sample_results/",
        help="Directory where predictions will be saved.",
    )
    parser.add_argument(
        "--csv_file",
        type=str,
        default="/mnt/hmi/thuong/Photoface_dist/PhotofaceDBNormalTrainValTest2/dataset_0/test.csv",
        help="CSV file listing test image paths.",
    )
    return parser.parse_args()


def load_models(args, device):
    """
    Load and initialize the models:
      - NormalEncoder and NormalDecoder for refining normal predictions.
      - PreTrainModel (UNet) for generating initial normal predictions.
    """
    # Initialize models and move them to the specified device
    normal_encoder = NormalEncoder(norm_dim=512).to(device)
    normal_decoder = NormalDecoder(bilinear=False).to(device)
    pretrain_model = UNet(channel=1).to(device)

    # Load pre-trained weights for the UNet model
    ckpt_model = torch.load(args.ckpt_s1, map_location=lambda storage, loc: storage)
    pretrain_model.load_state_dict(ckpt_model)

    # If a checkpoint for the refiner exists, load the encoder and decoder weights
    if args.ckpt is not None:
        ckpt = torch.load(args.ckpt, map_location=lambda storage, loc: storage)
        try:
            ckpt_name = os.path.basename(args.ckpt)
            args.start_iter = int(os.path.splitext(ckpt_name)[0])
        except ValueError:
            pass
        normal_encoder.load_state_dict(ckpt["NormalEncoder"])
        normal_decoder.load_state_dict(ckpt["NormalDecoder"])

    # Set all models to evaluation mode
    pretrain_model.eval()
    normal_encoder.eval()
    normal_decoder.eval()

    return normal_encoder, normal_decoder, pretrain_model


def get_image_transform():
    """
    Create a transformation pipeline to convert a PIL image into a tensor.
    """
    return transforms.Compose([transforms.ToTensor()])


def process_image(image_path, transform, device):
    """
    Open an image in grayscale, convert it to a tensor, and move it to the given device.
    
    Args:
        image_path (str): Path to the image file.
        transform (callable): Transformation to apply to the image.
        device (torch.device): Device to move the tensor onto.
    
    Returns:
        torch.Tensor: The processed image tensor.
    """
    image = Image.open(image_path).convert("L")
    tensor_img = transform(image).unsqueeze(0).to(device)
    return tensor_img


def run_inference(normal_encoder, normal_decoder, pretrain_model, args, device):
    """
    Iterate over all test samples, run the inference pipeline, and save predictions.

    For each image:
      1. Load and process the image.
      2. Pass it through the pre-trained model to obtain an initial prediction.
      3. Refine the prediction using NormalEncoder and NormalDecoder.
      4. Save the output as a NumPy array, preserving the directory structure.
    """
    transform = get_image_transform()
    sample_list = pd.read_csv(args.csv_file, header=None)
    num_samples = len(sample_list)

    # Ensure the destination directory exists
    Path(args.dest_dir).mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for i in tqdm(range(num_samples), desc="Running Inference"):
            # Construct the full path for the current image
            img_relative_path = sample_list.iloc[i, 0]
            img_path = os.path.join(args.data_dir, img_relative_path)

            # Process the image
            tensor_img = process_image(img_path, transform, device)

            # Run through the pre-trained model and refine the prediction
            co_norm = F.normalize(pretrain_model(tensor_img))
            normal_feat = normal_encoder(co_norm)
            fine_normal = F.normalize(normal_decoder((tensor_img, normal_feat)))

            # Convert output from (N, C, H, W) to (H, W, C) and then to a NumPy array
            output_np = fine_normal.permute(0, 2, 3, 1).cpu().numpy()[0]

            # Prepare the destination path, replacing "crop.jpg" with "predict.npy"
            dest_file = img_relative_path.replace("crop.jpg", "predict.npy")
            dest_path = os.path.join(args.dest_dir, dest_file)
            Path(Path(dest_path).parent).mkdir(parents=True, exist_ok=True)
            np.save(dest_path, output_np)


def main():
    # Parse arguments and set the device
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Enable cuDNN benchmarking for faster runtime (if input sizes are constant)
    torch.backends.cudnn.benchmark = True

    # Load models
    normal_encoder, normal_decoder, pretrain_model = load_models(args, device)

    # Run inference on the test dataset and save predictions
    run_inference(normal_encoder, normal_decoder, pretrain_model, args, device)


if __name__ == "__main__":
    main()
