"""Smoke test: train a tiny SCSI MLP on two_moons + gaussian_noise for a few
steps, then save a (clean / corrupted / generated) panel via the existing
2-D callback. Mirrors the wiring in `mlp_interpolants.py`.

Run from repo root:
    python tests/test_synthetic.py
"""
import os
import sys

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(REPO_ROOT, "src"))

from custom_datasets import CorruptedDataset, get_dataset
from callbacks import save_fig_2dsynt_vec
from interpolant_utils import SCSInterpolant
from mlps import FeedForwardwithEMB
import forward_maps as fwd_maps
from trainer_si import Trainer


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("DEVICE:", device)

    data_root = os.path.join(REPO_ROOT, "tests", "data")
    results_folder = os.path.join(REPO_ROOT, "tests", "results", "synthetic")
    os.makedirs(results_folder, exist_ok=True)

    # Dataset + corruption (additive Gaussian noise, no latents).
    dim_in = 2
    epsilon = 0.5
    fwd_func = fwd_maps.corruption_dict["gaussian_noise"](epsilon)
    use_latents = False

    clean_dataset, _, _ = get_dataset("two_moons", data_root, seed=42)
    dataset = CorruptedDataset(clean_dataset, fwd_func, tied_rng=False)

    # Validation panel for the callback.
    interpolant = SCSInterpolant(
        fwd_func, use_latents=use_latents, n_steps=40, alpha=0.9, resamples=2, gamma_scale=0.0,
    ).to(device)
    clean_valid = clean_dataset.array.to(device)
    corrupted_valid = interpolant.push_fwd(clean_valid)
    validation_data = (clean_valid, corrupted_valid, None)

    # Tiny model.
    model = FeedForwardwithEMB(dim_in, 32, [128] * 2, latent_dim=None).to(device)

    train_steps = 2000
    trainer = Trainer(
        model=model,
        interpolant=interpolant,
        dataset=dataset,
        train_batch_size=2048,
        gradient_accumulate_every=1,
        update_transport_every=1,
        train_lr=1e-3,
        lr_scheduler=None,
        train_num_steps=train_steps,
        save_and_sample_every=train_steps,  # one mid-train viz + 'fin' viz
        results_folder=results_folder,
        num_workers=0,
        clean_data_steps=0,
        callback_fn=save_fig_2dsynt_vec,
        validation_data=validation_data,
    )
    trainer.train()

    # Verify a viz file landed.
    pngs = [f for f in os.listdir(results_folder) if f.startswith("denoising_") and f.endswith(".png")]
    assert pngs, f"No denoising_*.png written to {results_folder}"
    print(f"PASS: wrote {sorted(pngs)} to {results_folder}")


if __name__ == "__main__":
    main()
