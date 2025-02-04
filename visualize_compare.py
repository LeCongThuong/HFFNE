import argparse
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


def N_SFS2CM(normal: np.ndarray, use_conversion: bool = True) -> np.ndarray:
    """
    Convert normal vectors from the SFS coordinate system to the CM coordinate system.

    For an input normal map with shape (H, W, 3):
      - Output channel 0 is taken from the input channel 1.
      - Output channel 1 is the negative of the input channel 0.
      - Output channel 2 remains unchanged.

    Args:
        normal (np.ndarray): Input normal map of shape (H, W, 3).
        use_conversion (bool): Whether to apply the conversion. Defaults to True.

    Returns:
        np.ndarray: Converted normal map of shape (H, W, 3) if use_conversion is True;
                    otherwise, returns the original normal map.
    """
    if not use_conversion:
        return normal

    converted = np.empty_like(normal)
    converted[:, :, 0] = normal[:, :, 1]
    converted[:, :, 1] = -normal[:, :, 0]
    converted[:, :, 2] = normal[:, :, 2]
    return converted


def load_data(gt_dir: str, pred_dir: str, pred_path: str, use_conversion: bool):
    """
    Load the ground truth and predicted data files based on a given predicted file path.
    
    The ground truth normal file is derived by replacing '_predict.npy' with '_sn.npy' in pred_path.
    Similarly, image and mask paths are derived from the normal file name.

    Args:
        gt_dir (str): Directory containing ground truth files.
        pred_dir (str): Directory containing predicted result files.
        pred_path (str): Relative file path of the predicted npy file.
        use_conversion (bool): Whether to apply N_SFS2CM conversion to the ground truth normals.

    Returns:
        tuple: (np_img, np_gt, np_pred, np_mask)
            - np_img (np.ndarray): Input image.
            - np_gt (np.ndarray): (Possibly converted) ground truth normal map.
            - np_pred (np.ndarray): Predicted normal map.
            - np_mask (np.ndarray): Mask image.
    """
    # Derive file paths
    gt_normal_path = pred_path.replace("_predict.npy", "_sn.npy")
    img_path = gt_normal_path.replace("_sn.npy", "_crop.jpg")
    mask_path = gt_normal_path.replace("_sn.npy", "_mask.png")

    # Load data
    np_gt = np.load(os.path.join(gt_dir, gt_normal_path))
    np_img = cv2.imread(os.path.join(gt_dir, img_path))
    np_pred = np.load(os.path.join(pred_dir, pred_path))
    np_mask = cv2.imread(os.path.join(gt_dir, mask_path))

    # Convert ground truth normals if desired
    np_gt = N_SFS2CM(np_gt, use_conversion=use_conversion)
    return np_img, np_gt, np_pred, np_mask


def plot_results(np_img: np.ndarray, np_gt: np.ndarray, np_pred: np.ndarray, np_mask: np.ndarray):
    """
    Display the input image, ground truth normals, predicted normals, and mask.

    The normal maps (which are assumed to be in the range [-1, 1]) are converted
    to the [0, 255] range for display purposes.

    Args:
        np_img (np.ndarray): Input image.
        np_gt (np.ndarray): Ground truth normal map.
        np_pred (np.ndarray): Predicted normal map.
        np_mask (np.ndarray): Mask image.
    """
    # Convert BGR to RGB for proper display with matplotlib
    np_img_rgb = cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB)

    # Map normals from [-1, 1] to [0, 255] and cast to uint8 for display
    gt_display = (((np_gt + 1) / 2) * 255).astype(np.uint8)
    pred_display = (((np_pred + 1) / 2) * 255).astype(np.uint8)

    # Create subplots
    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(16, 9))
    axes[0].imshow(np_img_rgb)
    axes[0].set_title("Input Image")
    axes[0].axis("off")

    axes[1].imshow(gt_display)
    axes[1].set_title("GT Normals")
    axes[1].axis("off")

    axes[2].imshow(pred_display)
    axes[2].set_title("Prediction")
    axes[2].axis("off")

    axes[3].imshow(np_mask)
    axes[3].set_title("Mask")
    axes[3].axis("off")

    plt.tight_layout()
    plt.show()


def parse_args():
    """
    Parse command-line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Visualize Photoface reconstruction results with optional normal conversion."
    )
    parser.add_argument(
        "--gt_dir",
        type=str,
        default="/mnt/hmi/thuong/Photoface_dist/PhotofaceDBLib/",
        help="Directory for ground truth files (e.g., images, normal maps, masks).",
    )
    parser.add_argument(
        "--pred_dir",
        type=str,
        required=True,
        help="Directory for predicted results (npy files).",
    )
    parser.add_argument(
        "--pred_path",
        type=str,
        required=True,
        help="Relative file path for the predicted npy file (e.g., '1057/2009-08-05_14-28-18/im2_predict.npy').",
    )
    parser.add_argument(
        "--no_conversion",
        action="store_true",
        help="Disable the N_SFS2CM conversion. By default, conversion is applied.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    use_conversion = not args.no_conversion

    # Load data using provided directories and file paths
    np_img, np_gt, np_pred, np_mask = load_data(
        gt_dir=args.gt_dir, pred_dir=args.pred_dir, pred_path=args.pred_path, use_conversion=use_conversion
    )

    # Display the results
    plot_results(np_img, np_gt, np_pred, np_mask)


if __name__ == "__main__":
    main()

#python3 visualize_compare.py --pred_dir "/mnt/hmi/thuong/HFFNE/sample_results/" --pred_path "1034/2009-08-03_12-39-50/im0_predict.npy"