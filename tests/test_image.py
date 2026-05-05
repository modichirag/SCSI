"""Smoke test: train a tiny SCSI U-Net on MNIST + random_mask for a handful
of steps, then save an (original / corrupted / generated) panel via the
existing image callback. Mirrors the wiring in `scsi.py`.

Run from repo root:
    python tests/test_image.py
"""
import os
import sys
from functools import partial

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(REPO_ROOT, "src"))

from custom_datasets import CorruptedDataset, ImagesOnly, get_dataset
from callbacks import save_image
import forward_maps as fwd_maps
from interpolant_utils import SCSInterpolant
from networks import ConditionalDhariwalUNet
from trainer_si import Trainer


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("DEVICE:", device)

    data_root = os.path.join(REPO_ROOT, "tests", "data")
    results_folder = os.path.join(REPO_ROOT, "tests", "results", "image")
    os.makedirs(results_folder, exist_ok=True)

    # Dataset (downloads to tests/data/mnist on first run).
    dataset, D, nc = get_dataset("mnist", data_root, seed=42)
    image_dataset = ImagesOnly(dataset)

    # Corruption: 50% pixels masked, tiny noise.
    corruption = "random_mask"
    corruption_levels = [0.5, 0.0]
    fwd_func = fwd_maps.corruption_dict[corruption](*corruption_levels)
    corrupt_fn = partial(fwd_func, cond_y=False, embed=False)
    use_latents, latent_dim = fwd_maps.parse_latents(corruption, D, C=nc, cond_y=False)

    # Tiny U-Net (8 channels keeps it well under a million params).
    model = ConditionalDhariwalUNet(
        D, nc, nc,
        latent_dim=latent_dim,
        model_channels=8,
        max_pos_embedding=2,
    ).to(device)

    interpolant = SCSInterpolant(
        corrupt_fn, use_latents=use_latents, n_steps=40, alpha=0.9, resamples=2, gamma_scale=0.0,
    ).to(device)

    corrupt_dataset = CorruptedDataset(image_dataset, corrupt_fn, tied_rng=True, base_seed=42)

    train_steps = 50
    trainer = Trainer(
        model=model,
        interpolant=interpolant,
        dataset=corrupt_dataset,
        train_batch_size=16,
        gradient_accumulate_every=1,
        train_lr=1e-4,
        lr_scheduler=None,
        train_num_steps=train_steps,
        save_and_sample_every=train_steps,  # one mid-train viz + 'fin' viz
        results_folder=results_folder,
        num_workers=0,
        callback_fn=save_image,
    )
    trainer.train()

    pngs = [f for f in os.listdir(results_folder) if f.startswith("denoising_") and f.endswith(".png")]
    assert pngs, f"No denoising_*.png written to {results_folder}"
    print(f"PASS: wrote {sorted(pngs)} to {results_folder}")


if __name__ == "__main__":
    main()
