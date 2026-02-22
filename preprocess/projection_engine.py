"""
Multi-view Maximum Intensity Projection (MIP) for 3D volumetric masks.

Generates 2D projections from 3D masks via spherical-view sampling and GPU-accelerated
affine transforms. Supports golden-angle, uniform-grid, and per-sample random sampling.
"""

from __future__ import annotations

import logging
import math
import os
import pickle
import time
from typing import List, Literal, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy.io import loadmat

logger = logging.getLogger(__name__)

SamplingMethod = Literal["golden_angle", "uniform_grid", "random"]
SphericalCoords = List[Tuple[float, float]]


def _spherical_golden_angle(n_views: int) -> SphericalCoords:
    """
    Golden-angle spiral sampling on the unit sphere.

    Uses φ = π(3 - √5) to achieve quasi-uniform coverage without periodic overlap.
    θ (polar): [0, π]; φ (azimuthal): [0, 2π).

    Returns:
        List of (θ, φ) in radians.
    """
    phi_golden = np.pi * (3.0 - np.sqrt(5.0))
    return [
        (np.arccos(1.0 - (i / max(n_views - 1, 1)) * 2.0), phi_golden * i)
        for i in range(n_views)
    ]


def _spherical_uniform_grid(n_views: int) -> SphericalCoords:
    """
    Uniform latitude-longitude grid on the sphere.

    θ: polar angle [0, π]; φ: azimuth [0, 2π).
    Grid size ≈ √n_views × √n_views.

    Returns:
        List of (θ, φ) in radians.
    """
    n_theta = max(1, int(math.sqrt(n_views)))
    n_phi = max(1, int(np.ceil(n_views / n_theta)))
    coords: SphericalCoords = []
    for i in range(n_theta):
        theta = (i * math.pi / (n_theta - 1)) if n_theta > 1 else math.pi / 2
        for j in range(n_phi):
            phi = j * 2 * math.pi / n_phi
            if len(coords) < n_views:
                coords.append((theta, phi))
            else:
                break
        if len(coords) >= n_views:
            break
    while len(coords) < n_views:
        coords.append(coords[-1] if coords else (math.pi / 2, 0.0))
    return coords[:n_views]


def _spherical_random(n_views: int, seed: Optional[int] = None) -> SphericalCoords:
    """
    Uniform random sampling on the sphere via inverse transform.

    θ ~ arccos(U[-1,1]) for uniform solid angle; φ ~ U[0, 2π).

    Returns:
        List of (θ, φ) in radians.
    """
    rng = np.random.default_rng(seed)
    coords: SphericalCoords = []
    for _ in range(n_views):
        u = rng.uniform(-1.0, 1.0)
        theta = np.arccos(np.clip(u, -1.0, 1.0))
        phi = rng.uniform(0.0, 2.0 * np.pi)
        coords.append((theta, phi))
    return coords


class ProjectionEngine:
    """
    GPU-accelerated multi-view MIP engine for 3D masks.

    Applies two sequential rotations (Z then X, Euler ZYX convention) to align
    the view direction with the projection axis, then computes max-intensity
    projection along that axis.
    """

    def __init__(
        self,
        n_views: int = 16,
        sampling_method: SamplingMethod = "golden_angle",
        random_seed: Optional[int] = None,
        device: Optional[torch.device] = None,
        threshold: float = 0.4,
    ) -> None:
        """
        Args:
            n_views: Number of projection views per volume.
            sampling_method: 'golden_angle' | 'uniform_grid' | 'random'.
            random_seed: Used only for 'random'; enables reproducibility.
            device: CUDA device or CPU.
            threshold: Binarization threshold for mask values (< threshold → 0).
        """
        self.n_views = n_views
        self.sampling_method = sampling_method
        self.random_seed = random_seed
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.threshold = threshold

    def _get_view_angles(self, sample_idx: int) -> SphericalCoords:
        """Per-sample view angles. Random mode uses sample_idx for per-sample uniqueness."""
        if self.sampling_method == "golden_angle":
            return _spherical_golden_angle(self.n_views)
        if self.sampling_method == "uniform_grid":
            return _spherical_uniform_grid(self.n_views)
        seed = (self.random_seed + sample_idx) if self.random_seed is not None else None
        return _spherical_random(self.n_views, seed=seed)

    def _rotation_matrix_z(self, theta: torch.Tensor) -> torch.Tensor:
        """Rotation about Z-axis by θ (radians). Shape: (3, 3)."""
        c, s = torch.cos(theta), torch.sin(theta)
        return torch.stack(
            [
                torch.stack([c, -s, torch.zeros_like(c)]),
                torch.stack([s, c, torch.zeros_like(c)]),
                torch.stack([torch.zeros_like(c), torch.zeros_like(c), torch.ones_like(c)]),
            ],
            dim=-1,
        )

    def _rotation_matrix_x(self, phi: torch.Tensor) -> torch.Tensor:
        """Rotation about X-axis by φ (radians). Shape: (3, 3). Euler ZYX: second rotation."""
        c, s = torch.cos(phi), torch.sin(phi)
        return torch.stack(
            [
                torch.stack([torch.ones_like(c), torch.zeros_like(c), torch.zeros_like(c)]),
                torch.stack([torch.zeros_like(c), c, -s]),
                torch.stack([torch.zeros_like(c), s, c]),
            ],
            dim=-1,
        )

    def _affine_from_rotation(self, rotation_matrix: torch.Tensor) -> torch.Tensor:
        """Build 3x4 affine matrix (R^T | 0) for grid_sample. Shape: (1, 3, 4)."""
        rotation_inverse = rotation_matrix.transpose(-2, -1)
        translation_column = torch.zeros(3, 1, dtype=rotation_matrix.dtype, device=rotation_matrix.device)
        return torch.cat([rotation_inverse, translation_column], dim=-1).unsqueeze(0)

    def _project_single_view(
        self,
        volume: torch.Tensor,
        theta: float,
        phi: float,
    ) -> np.ndarray:
        """
        MIP for one view direction given by spherical (θ, φ).

        Euler ZYX: first Rz(θ), then Rx(φ). Projection along transformed axis (dim=3).
        """
        theta_tensor = torch.tensor(theta, dtype=torch.float32, device=self.device)
        phi_tensor = torch.tensor(phi, dtype=torch.float32, device=self.device)

        rotation_z = self._rotation_matrix_z(theta_tensor)
        rotation_x = self._rotation_matrix_x(phi_tensor)
        rotation_matrix = rotation_x @ rotation_z

        affine = self._affine_from_rotation(rotation_matrix)
        grid = F.affine_grid(affine, volume.size(), align_corners=False)
        rotated = F.grid_sample(volume, grid, mode="nearest", align_corners=False)
        projection_2d = rotated.max(dim=3)[0].squeeze(0).squeeze(0)
        return projection_2d.cpu().numpy()

    def process_volume(self, mask_data: np.ndarray, sample_idx: int = 0) -> np.ndarray:
        """
        Compute MIP projections for one 3D mask.

        Args:
            mask_data: (H, W, D) or (Y, X, Z) from .mat.
            sample_idx: Index for per-sample random sampling (random mode only).

        Returns:
            (n_views, H, W) float array.
        """
        mask_data = np.where(mask_data < self.threshold, 0.0, mask_data)
        volume = torch.from_numpy(mask_data.astype(np.float32)).to(self.device)
        volume = volume.permute(2, 1, 0).unsqueeze(0).unsqueeze(0)

        projections: List[np.ndarray] = []
        for theta, phi in self._get_view_angles(sample_idx):
            projection_2d = self._project_single_view(volume, theta, phi)
            projections.append(projection_2d)

        projection_stack = np.stack(projections, axis=0)
        del volume, projections
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        return projection_stack

    def run(
        self,
        mask_folder: str,
        projection_folder: str,
        *,
        auto_naming: bool = False,
    ) -> str:
        """
        Process all .mat masks in mask_folder and save projections as .pkl.

        Args:
            mask_folder: Directory containing .mat files with 'mask_data' key.
            projection_folder: Output directory for .pkl files.
            auto_naming: If True, append sampling suffix to projection_folder.

        Returns:
            Resolved projection_folder path.
        """
        if auto_naming:
            base_path = os.path.dirname(mask_folder.rstrip("/")) or mask_folder
            method_suffix = self.sampling_method
            if self.sampling_method == "random" and self.random_seed is not None:
                method_suffix = f"{method_suffix}_seed{self.random_seed}"
            projection_folder = os.path.join(base_path, f"projection_{self.n_views}_{method_suffix}")

        os.makedirs(projection_folder, exist_ok=True)
        logger.info("Device: %s | Method: %s | n_views: %d", self.device, self.sampling_method, self.n_views)

        mask_files = sorted(f for f in os.listdir(mask_folder) if f.endswith(".mat"))
        if not mask_files:
            raise FileNotFoundError(f"No .mat files in {mask_folder}")

        total_start = time.perf_counter()
        sample_times: List[float] = []

        try:
            from tqdm import tqdm
            progress_bar = tqdm(enumerate(mask_files), total=len(mask_files), desc="MIP", unit="vol")
        except ImportError:
            progress_bar = enumerate(mask_files)

        for idx, mask_file in progress_bar:
            mask_path = os.path.join(mask_folder, mask_file)
            try:
                mat_content = loadmat(mask_path)
                if "mask_data" not in mat_content:
                    raise KeyError(f"'mask_data' not found in {mask_file}")
                mask_data = mat_content["mask_data"]
            except Exception as e:
                logger.exception("Failed to load %s: %s", mask_path, e)
                raise

            sample_start = time.perf_counter()
            try:
                projections = self.process_volume(mask_data, sample_idx=idx)
            except torch.cuda.OutOfMemoryError:
                logger.exception("OOM processing %s", mask_file)
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
                raise

            elapsed = time.perf_counter() - sample_start
            sample_times.append(elapsed)

            output_path = os.path.join(projection_folder, mask_file.replace(".mat", ".pkl"))
            with open(output_path, "wb") as output_file:
                pickle.dump(projections, output_file)

            if hasattr(progress_bar, "set_postfix"):
                progress_bar.set_postfix(file=mask_file[:20], time=f"{elapsed:.1f}s")
            logger.info("[%d/%d] %s | %.2fs", idx + 1, len(mask_files), mask_file, elapsed)
            if logger.isEnabledFor(logging.DEBUG):
                view_angles = self._get_view_angles(idx)
                for view_idx, (theta_rad, phi_rad) in enumerate(view_angles[:3]):
                    logger.debug("  view %d: θ=%.4f φ=%.4f", view_idx, theta_rad, phi_rad)

        total_elapsed = time.perf_counter() - total_start
        logger.info(
            "Done. n=%d | total=%.2fs | avg=%.2fs | min=%.2fs max=%.2fs",
            len(mask_files),
            total_elapsed,
            np.mean(sample_times),
            min(sample_times),
            max(sample_times),
        )
        return projection_folder


def generate_projections(
    mask_folder: str,
    projection_folder: Optional[str] = None,
    n_views: int = 16,
    sampling_method: SamplingMethod = "golden_angle",
    random_seed: Optional[int] = None,
    auto_naming: bool = True,
) -> str:
    """
    Convenience wrapper for ProjectionEngine.run().

    Args:
        mask_folder: Input directory of .mat masks.
        projection_folder: Output directory; if None and auto_naming, derived from mask_folder.
        n_views: Number of views per volume.
        sampling_method: 'golden_angle' | 'uniform_grid' | 'random'.
        random_seed: For 'random' only; enables reproducibility.
        auto_naming: If True, build projection_folder from mask_folder and sampling_method.

    Returns:
        Resolved projection_folder path.
    """
    engine = ProjectionEngine(
        n_views=n_views,
        sampling_method=sampling_method,
        random_seed=random_seed,
        threshold=0.4,
    )
    return engine.run(
        mask_folder,
        projection_folder or ".",
        auto_naming=auto_naming or projection_folder is None,
    )


def generate_projections_batch(
    mask_folder: str,
    base_output_dir: Optional[str] = None,
    n_views: int = 16,
    sampling_methods: Optional[List[SamplingMethod]] = None,
    random_seed: int = 42,
) -> None:
    """
    Generate projections for multiple sampling methods into separate output dirs.

    Args:
        mask_folder: Input mask directory.
        base_output_dir: Base path for outputs; defaults to mask_folder parent.
        n_views: Views per volume.
        sampling_methods: Methods to run; default ['golden_angle','uniform_grid','random'].
        random_seed: Used for 'random' method.
    """
    base_output_dir = base_output_dir or os.path.dirname(os.path.abspath(mask_folder.rstrip("/")))
    sampling_methods = sampling_methods or ["golden_angle", "uniform_grid", "random"]

    logger.info("Batch projection | input=%s | base=%s | methods=%s", mask_folder, base_output_dir, sampling_methods)

    for method in sampling_methods:
        method_suffix = f"{method}_seed{random_seed}" if method == "random" else method
        projection_output_dir = os.path.join(base_output_dir, f"projection_{n_views}_{method_suffix}")
        engine = ProjectionEngine(
            n_views=n_views,
            sampling_method=method,
            random_seed=random_seed if method == "random" else None,
        )
        engine.run(mask_folder, projection_output_dir, auto_naming=False)
        logger.info("Completed %s -> %s", method, projection_output_dir)


def _parse_args() -> "argparse.Namespace":
    import argparse

    parser = argparse.ArgumentParser(
        prog="projections_three",
        description="Multi-view MIP projection for 3D masks. Supports golden-angle, uniform-grid, and per-sample random spherical sampling.",
    )
    parser.add_argument(
        "--mask-folder",
        type=str,
        required=True,
        help="Directory containing .mat files with 'mask_data' key.",
    )
    parser.add_argument(
        "--projection-folder",
        type=str,
        default=None,
        help="Output directory. If omitted, derived from mask-folder and sampling method.",
    )
    parser.add_argument(
        "--n-views",
        type=int,
        default=16,
        help="Number of projection views per volume.",
    )
    parser.add_argument(
        "--sampling-method",
        type=str,
        choices=["golden_angle", "uniform_grid", "random"],
        default="golden_angle",
        help="Spherical sampling strategy. 'random' uses per-sample independent angles.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for 'random' sampling (enables reproducibility).",
    )
    parser.add_argument(
        "--batch-comparison",
        action="store_true",
        help="Run all three sampling methods and save to separate output dirs.",
    )
    parser.add_argument(
        "--base-output-dir",
        type=str,
        default=None,
        help="Base output directory for --batch-comparison. Default: parent of mask-folder.",
    )
    parser.add_argument(
        "--no-auto-naming",
        action="store_true",
        help="Disable automatic output folder naming.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    try:
        from tqdm import tqdm
        _HAS_TQDM = True
    except ImportError:
        _HAS_TQDM = False

    if args.batch_comparison:
        generate_projections_batch(
            mask_folder=args.mask_folder,
            base_output_dir=args.base_output_dir,
            n_views=args.n_views,
            random_seed=args.random_seed,
        )
    else:
        if _HAS_TQDM:
            logger.info("Processing masks (tqdm in engine)...")
        generate_projections(
            mask_folder=args.mask_folder,
            projection_folder=args.projection_folder,
            n_views=args.n_views,
            sampling_method=args.sampling_method,
            random_seed=args.random_seed if args.sampling_method == "random" else None,
            auto_naming=not args.no_auto_naming,
        )
