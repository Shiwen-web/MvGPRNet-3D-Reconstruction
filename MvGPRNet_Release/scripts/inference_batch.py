#!/usr/bin/env python3
"""Launcher for batch inference. Run from project root or scripts/."""
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import inference_batch from inference folder
sys.path.insert(0, str(PROJECT_ROOT / "inference"))
import inference_batch as _batch_mod
inference_batch = _batch_mod.inference_batch

if __name__ == "__main__":
    import argparse
    import json
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="MvGPRNet batch inference: process folder and save as .mat."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to JSON config file. Avoids long command lines and line continuation issues.",
    )
    parser.add_argument("--n_views", type=int, default=None)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--projection_folder", type=str, default=None)
    parser.add_argument("--label_folder", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./output-mvnet")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_mat", action="store_true", default=True)

    args = parser.parse_args()

    if args.config is not None:
        config_path = Path(args.config)
        if not config_path.is_absolute():
            config_path = (Path.cwd() / config_path).resolve()
        with open(config_path, "r") as f:
            cfg = json.load(f)
        n_views = cfg.get("n_views", args.n_views)
        model_path = cfg.get("model_path", args.model_path)
        projection_folder = cfg.get("projection_folder", args.projection_folder)
        label_folder = cfg.get("label_folder", args.label_folder)
        output_dir = cfg.get("output_dir", args.output_dir)
        seed = cfg.get("seed", args.seed)
    else:
        n_views = args.n_views
        model_path = args.model_path
        projection_folder = args.projection_folder
        label_folder = args.label_folder
        output_dir = args.output_dir
        seed = args.seed

    if n_views is None or model_path is None or projection_folder is None or label_folder is None:
        parser.error(
            "When not using --config, the following are required: --n_views, --model_path, "
            "--projection_folder, --label_folder. Tip: Use --config ../inference/inference_batch_config.json"
        )

    results = inference_batch(
        n_views=n_views,
        model_path=model_path,
        projection_folder=projection_folder,
        label_folder=label_folder,
        output_dir=output_dir,
        seed=seed,
        save_mat=args.save_mat,
    )

    if results:
        logging.info("Inference completed for %d samples", len(results))
