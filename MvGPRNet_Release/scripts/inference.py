"""MvGPRNet single inference run."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import logging
import os
from typing import Dict, Optional

import numpy as np
import torch
from scipy.io import savemat
from torch.utils.data import DataLoader

from datasets import MvGPRDatasetTest
from models import MvGPRNet
from utils import (
    calculate_dice,
    calculate_mae,
    calculate_mse,
    verify_model_n_views,
)


def inference_single_run(
    n_views: int,
    model_path: str,
    projection_folder: str,
    label_folder: str,
    seed: int,
    save_results: bool = True,
    save_dir: Optional[str] = None,
) -> Optional[Dict[str, Dict[str, float]]]:
    """Single inference run with given seed.

    Args:
        n_views: Number of views.
        model_path: Path to model checkpoint (fixed path).
        projection_folder: Path to projection data folder (fixed path).
        label_folder: Path to label folder.
        seed: Random seed (must match training seed; explicitly passed).
        save_results: Whether to save 3D reconstruction results.
        save_dir: Directory to save results (required if save_results=True).

    Returns:
        Dict mapping file_name to {mse, mae, dice}, or None on failure.
    """
    logger = logging.getLogger(__name__)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if model_path is None or not os.path.exists(model_path):
        logger.error("Model weights not found: %s", model_path)
        return None

    if projection_folder is None or not os.path.exists(projection_folder):
        logger.error("Projection folder not found: %s", projection_folder)
        return None

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = MvGPRNet(n_views=n_views).to(device)

    if not verify_model_n_views(model_path, n_views):
        logger.error(
            "Model n_views mismatch for %s (expected %d)", model_path, n_views
        )
        return None

    try:
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict):
            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        logger.info("Loaded model: %s", model_path)
    except Exception as e:
        logger.error("Failed to load model %s: %s", model_path, e)
        return None

    dataset = MvGPRDatasetTest(
        label_folder=label_folder,
        projection_folder=projection_folder,
        n_views=n_views,
    )
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    metrics_dict: Dict[str, Dict[str, float]] = {}

    with torch.no_grad():
        for imaging_projs, label_projs, files_name in data_loader:
            imaging_projs = imaging_projs.to(device)
            label_projs = label_projs.to(device)
            volume_pred = model(imaging_projs)
            volume_pred_np = volume_pred.squeeze().cpu().numpy()
            if volume_pred_np.ndim == 4:
                volume_pred_np = volume_pred_np.squeeze(0)

            file_name = files_name[0]
            if save_results and save_dir is not None:
                type_info = (
                    file_name.split("_")[2]
                    if len(file_name.split("_")) > 2
                    else "unknown"
                )
                result_save_dir = os.path.join(save_dir, "Result", type_info)
                os.makedirs(result_save_dir, exist_ok=True)
                data_3d = volume_pred_np
                mat_path = os.path.join(result_save_dir, f"{file_name}.mat")
                savemat(mat_path, {"data_3d": data_3d})

            mse = calculate_mse(volume_pred.squeeze(), label_projs.squeeze())
            mae = calculate_mae(volume_pred.squeeze(), label_projs.squeeze())
            dice = calculate_dice(volume_pred.squeeze(), label_projs.squeeze())
            metrics_dict[file_name] = {"mse": mse, "mae": mae, "dice": dice}

    return metrics_dict


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="MvGPRNet single inference run."
    )
    parser.add_argument("--n_views", type=int, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--projection_folder", type=str, required=True)
    parser.add_argument("--label_folder", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--save_results", action="store_true", default=True)

    args = parser.parse_args()

    results = inference_single_run(
        n_views=args.n_views,
        model_path=args.model_path,
        projection_folder=args.projection_folder,
        label_folder=args.label_folder,
        seed=args.seed,
        save_results=args.save_results,
        save_dir=args.save_dir,
    )

    if results:
        logging.info("Inference completed for %d samples", len(results))
