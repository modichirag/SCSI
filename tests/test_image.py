"""Smoke test: train a tiny SCSI U-Net on MNIST + random_mask for a handful
of steps, then save an (original / corrupted / generated) panel via the
existing image callback. Mirrors the wiring in `scsi_image.py`.

Run from repo root:
    python tests/test_image.py
"""
import json
import os
import sys
from argparse import Namespace
from functools import partial

import torch
from torch.utils.data import Subset

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(REPO_ROOT, "src"))

from custom_datasets import CorruptedDataset, ImagesOnly, get_dataset
from callbacks import save_image
import forward_maps as fwd_maps
from interpolant_utils import SCSInterpolant
from networks import ConditionalDhariwalUNet
from trainer_si import Trainer
from utils import make_serializable


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("DEVICE:", device)

    data_root = os.path.join(REPO_ROOT, "tests", "data")
    results_folder = os.path.join(REPO_ROOT, "tests", "results", "image")
    os.makedirs(results_folder, exist_ok=True)
    args = Namespace(
        dataset="mnist",
        data_root=data_root,
        results_folder=results_folder,
        corruption="random_mask",
        corruption_levels=[0.5, 0.0],
        cond_y=False,
        embed=False,
        train_steps=6,
        save_and_sample_every=3,
        train_batch_size=8,
        gradient_accumulate_every=1,
        train_lr=3e-4,
        model_channels=8,
        max_pos_embedding=2,
        ode_steps=4,
        alpha=0.9,
        resamples=1,
        gamma_scale=0.0,
        tied_rng=True,
        dataset_seed=42,
        train_dataset_size=128,
    )

    # Dataset (downloads to tests/data/mnist on first run).
    dataset, D, nc = get_dataset(args.dataset, args.data_root, seed=args.dataset_seed)
    image_dataset = ImagesOnly(Subset(dataset, range(args.train_dataset_size)))

    # Corruption: 50% pixels masked, tiny noise.
    fwd_func = fwd_maps.corruption_dict[args.corruption](*args.corruption_levels)
    corrupt_fn = partial(fwd_func, cond_y=args.cond_y, embed=args.embed)
    use_latents, latent_dim = fwd_maps.parse_latents(args.corruption, D, C=nc, cond_y=args.cond_y)
    with open(os.path.join(args.results_folder, "args.json"), "w") as f:
        json.dump(make_serializable(vars(args)), f, indent=4)

    # Tiny U-Net (16 channels keeps it well under a million params).
    model = ConditionalDhariwalUNet(
        D, nc, nc,
        latent_dim=latent_dim,
        model_channels=args.model_channels,
        max_pos_embedding=args.max_pos_embedding,
    ).to(device)

    interpolant = SCSInterpolant(
        corrupt_fn, use_latents=use_latents, n_steps=args.ode_steps, alpha=args.alpha,
        resamples=args.resamples, gamma_scale=args.gamma_scale,
    ).to(device)

    corrupt_dataset = CorruptedDataset(image_dataset, corrupt_fn, tied_rng=args.tied_rng, base_seed=args.dataset_seed)
    trainer = Trainer(
        model=model,
        interpolant=interpolant,
        dataset=corrupt_dataset,
        train_batch_size=args.train_batch_size,
        gradient_accumulate_every=args.gradient_accumulate_every,
        train_lr=args.train_lr,
        lr_scheduler=None,
        train_num_steps=args.train_steps,
        save_and_sample_every=args.save_and_sample_every,  # one mid-train viz + 'fin' viz
        results_folder=args.results_folder,
        num_workers=0,
        callback_fn=save_image,
    )
    trainer.train()

    pngs = sorted(f for f in os.listdir(results_folder) if f.startswith("denoising_") and f.endswith(".png"))
    assert "denoising_3.png" in pngs, f"Expected step-based callback file denoising_3.png in {results_folder}, found {pngs}"
    assert "denoising_6.png" in pngs, f"Expected step-based callback file denoising_6.png in {results_folder}, found {pngs}"
    assert "denoising_fin.png" in pngs, f"Expected final callback file denoising_fin.png in {results_folder}, found {pngs}"
    assert os.path.exists(os.path.join(results_folder, "args.json")), f"Expected args.json in {results_folder}"
    print(f"PASS: wrote {pngs} to {results_folder}")


if __name__ == "__main__":
    main()
