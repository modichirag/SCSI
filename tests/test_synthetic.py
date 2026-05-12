"""Smoke test: train a tiny SCSI MLP on two_moons + gaussian_noise for a few
steps, then save a (clean / corrupted / generated) panel via the existing
2-D callback. Mirrors the wiring in `scsi_synthetic.py`.

Run from repo root:
    python tests/test_synthetic.py
"""
import json
import os
import sys
from argparse import Namespace

import torch
from torch.utils.data import Subset

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(REPO_ROOT, "src"))

from custom_datasets import CorruptedDataset, get_dataset
from callbacks import save_fig_2dsynt_vec
from interpolant_utils import SCSInterpolant
from mlps import FeedForwardwithEMB
import forward_maps as fwd_maps
from trainer_si import Trainer
from utils import make_serializable


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("DEVICE:", device)

    data_root = os.path.join(REPO_ROOT, "tests", "data")
    results_folder = os.path.join(REPO_ROOT, "tests", "results", "synthetic")
    os.makedirs(results_folder, exist_ok=True)
    args = Namespace(
        dataset="two_moons",
        data_root=data_root,
        results_folder=results_folder,
        corruption="gaussian_noise",
        corruption_levels=[0.5],
        train_steps=10,
        save_and_sample_every=5,
        train_batch_size=64,
        gradient_accumulate_every=1,
        update_transport_every=1,
        train_lr=1e-3,
        ode_steps=4,
        alpha=0.9,
        resamples=1,
        gamma_scale=0.0,
        dataset_seed=42,
        clean_data_steps=0,
        tied_rng=False,
        t_emb_dim=16,
        hidden_dims=[32],
        train_dataset_size=512,
        validation_dataset_size=4096,
    )

    # Dataset + corruption (additive Gaussian noise, no latents).
    dim_in = 2
    fwd_func = fwd_maps.corruption_dict[args.corruption](*args.corruption_levels)
    use_latents = False
    with open(os.path.join(args.results_folder, "args.json"), "w") as f:
        json.dump(make_serializable(vars(args)), f, indent=4)

    clean_dataset, _, _ = get_dataset(args.dataset, args.data_root, seed=args.dataset_seed)
    dataset = CorruptedDataset(Subset(clean_dataset, range(args.train_dataset_size)), fwd_func, tied_rng=args.tied_rng)

    # Validation panel for the callback.
    interpolant = SCSInterpolant(
        fwd_func, use_latents=use_latents, n_steps=args.ode_steps, alpha=args.alpha,
        resamples=args.resamples, gamma_scale=args.gamma_scale,
    ).to(device)
    clean_valid = clean_dataset.array[:args.validation_dataset_size].to(device)
    corrupted_valid = interpolant.push_fwd(clean_valid)
    validation_data = (clean_valid, corrupted_valid, None)

    # Tiny model.
    model = FeedForwardwithEMB(dim_in, args.t_emb_dim, args.hidden_dims, latent_dim=None).to(device)

    trainer = Trainer(
        model=model,
        interpolant=interpolant,
        dataset=dataset,
        train_batch_size=args.train_batch_size,
        gradient_accumulate_every=args.gradient_accumulate_every,
        update_transport_every=args.update_transport_every,
        train_lr=args.train_lr,
        lr_scheduler=None,
        train_num_steps=args.train_steps,
        save_and_sample_every=args.save_and_sample_every,  # one mid-train viz + 'fin' viz
        results_folder=args.results_folder,
        num_workers=0,
        clean_data_steps=args.clean_data_steps,
        callback_fn=save_fig_2dsynt_vec,
        validation_data=validation_data,
    )
    trainer.train()

    # Verify step-based callback filenames.
    pngs = sorted(f for f in os.listdir(results_folder) if f.startswith("denoising_") and f.endswith(".png"))
    assert "denoising_5.png" in pngs, f"Expected step-based callback file denoising_5.png in {results_folder}, found {pngs}"
    assert "denoising_10.png" in pngs, f"Expected step-based callback file denoising_10.png in {results_folder}, found {pngs}"
    assert "denoising_fin.png" in pngs, f"Expected final callback file denoising_fin.png in {results_folder}, found {pngs}"
    assert os.path.exists(os.path.join(results_folder, "args.json")), f"Expected args.json in {results_folder}"
    print(f"PASS: wrote {pngs} to {results_folder}")


if __name__ == "__main__":
    main()
