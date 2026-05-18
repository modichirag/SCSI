# SCSI — Self-Consistent Stochastic Interpolants

Official implementation of **Generative Modeling from Black-box Corruptions via Self-Consistent Stochastic Interpolants** (Chirag Modi, Jiequn Han, Eric Vanden-Eijnden, Joan Bruna — ICLR 2026). [[link]](https://arxiv.org/abs/2512.10857)

Research code for training generative models (stochastic interpolants) from **corrupted** observations — i.e. learning an generative model for clean-data distribution when only degraded samples are available. Corruptions include Gaussian noise, random/block masking, Gaussian/motion blur, and random projections. Tested on CIFAR-10, MNIST, and the 2-D synthetic distribution.

## Installation

```bash
pip install -r requirements.txt
```

Python 3.9+ and a recent PyTorch (CUDA for serious training; CPU is fine for the 2-D MLP smoke tests) are expected.

## Quickstart: 2-D synthetic example

A small MLP trained on `two_moons` with additive Gaussian noise. Runs on CPU in a couple of minutes and exercises the full training loop.

```bash
python scsi_synthetic.py \
    --dataset two_moons \
    --corruption gaussian_noise \
    --corruption_levels 0.5 \
    --n_samples 6000 \
    --train_steps 5000 \
    --learning_rate 3e-3 \
    --suffix test
```

Other settings use driver defaults, including `batch_size=2000`, `fc_width=256`, `fc_depth=3`, `resamples=1`, `alpha=0.9`, `ode_steps=40`, and `t_emb_dim=32`.

Outputs (loss curve, intermediate restored snapshots, final model) land under `./results/two_moons/gaussian_noise/0.50-test/`. `tests/test_synthetic.py` runs a much smaller configuration as a wiring check.

## Quickstart: MNIST under various corruptions

A tiny U-Net trained on MNIST. Each setting runs in a couple of minutes on a single A100 GPU at smoke-test scale (`train_steps=100`) and already produces visually reasonable restorations — handy for exploring how the method behaves under different forward models without committing to a paper-tier run.

```bash
python -u scsi_image.py \
    --dataset mnist \
    --corruption random_mask \
    --corruption_levels 0.5 0.0 1.0 \
    --train_steps 100 \
    --channels 32 \
    --ode_steps 64 \
    --alpha 0.9 \
    --resamples 2 \
    --learning_rate 3e-4 \
    --save_every 10 \
    --suffix test
```

For `random_mask`, the three positional levels are `(mask_ratio, epsilon, noise_mask)`: 50% of pixels are masked, kept pixels carry no extra observation noise (`epsilon=0.0`), and masked pixels are replaced by standard Gaussian noise (`noise_mask=1.0`). This matches the noiseless paper row (Table 1, σ_n=10⁻⁶); the noisier row uses `0.5 0.1 1.0`.

To try a different forward model, swap the `--corruption` / `--corruption_levels` pair — all other flags can stay the same:

| `--corruption`         | `--corruption_levels` | Argument semantics                       |
|---|---|---|
| `random_mask`          | `0.5 0.0 1.0`         | `(mask_ratio, epsilon, noise_mask)`      |
| `gaussian_blur`        | `1.0 0.1`             | `(sigma_R, sigma_n)`                     |
| `gaussian_blur_pnoise` | `1.0 0.1`             | `(sigma_R, lambda_n)` — Poisson rate     |
| `random_motion`        | `5.0 0.1`             | `(kernel_size, sigma_n)`                 |

Outputs land under `./results/mnist/<corruption>/<levels>-test/`. `tests/test_image.py` runs a much smaller configuration as a wiring check.

## A real example: CIFAR-10 with random masking

```bash
python -u scsi_image.py \
    --dataset cifar10 --corruption random_mask \
    --corruption_levels 0.5 0.0 1.0 \
    --train_steps 50000 --channels 32 --ode_steps 64 \
    --alpha 0.9 --resamples 2 --learning_rate 5e-4 --lr_scheduler
```

CIFAR-10 downloads to `$SCSI_DATA` (default `./data/cifar10/`) on first call. Outputs go to `$SCSI_RESULTS/<dataset>/<corruption>/<run-slug>/` (with a `-mv` token appended to the slug when `--multiview` is passed).

Distributed (single-node, multi-GPU) training uses `torchrun`:

```bash
torchrun --standalone --nproc_per_node=4 scsi_distributed.py [args...]
```

## Paths

Two environment variables (also exposed as CLI flags) control where data and outputs live:

| Variable | CLI flag | Default | Purpose |
|---|---|---|---|
| `SCSI_DATA` | `--data_root` | `./data` | Dataset caches (MNIST, CIFAR-10, synthetic) |
| `SCSI_RESULTS` | `--results_root` | `./results` | Training outputs, `model-best.pt`, logs |

## Datasets

All drivers pull data via `get_dataset(name, data_root, seed=42)` from `src/custom_datasets.py`. Supported names:

- `mnist`, `cifar10` — downloaded via `torchvision` on first call.
- `two_moons` — synthetic; cached under `$SCSI_DATA/two_moons/seed_<seed>/`.

## Driver map

Top-level drivers add `<repo>/src` to `sys.path` via a `__file__`-relative `sys.path.append`, so they can be invoked from any working directory — no install step needed.

| Script | Purpose |
|---|---|
| `scsi_synthetic.py` | 2-D synthetic MLP experiments (`two_moons`) |
| `scsi_image.py` | Single-GPU training from corrupted images |
| `scsi_distributed.py` | DDP variant via `torchrun` |
| `train.py`, `sample.py`, `fid_eval.py` | Plain EDM diffusion baseline (clean data only) |
| `fid_eval_interpolants.py`, `fid_eval_dps.py` | FID evaluation of trained interpolants / DPS baseline |
| `lpips_eval_interpolants.py`, `lpips_eval_dps.py` | LPIPS / PSNR / SSIM evaluation |
| `clean_interpolants.py` | Produces "cleaned" samples from a trained interpolant |

## Repository layout

```
src/                          # Library code — not an installable package yet
  custom_datasets.py          # Dataset registry, CorruptedDataset
  forward_maps.py             # Corruption factories (corruption_dict)
  interpolant_utils.py        # SCSInterpolant and variants
  trainer_si.py               # Production training loop (EMA, DDP, checkpointing)
  trainer.py                  # Older/simpler loop used by train.py (EDM baseline)
  networks.py, karras_unet*.py, mlps.py
  paths.py                    # SCSI_DATA / SCSI_RESULTS resolution
```

Top-level `*.py` files are thin argparse + config wrappers around `src/`.

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{modi2025generative,
  title={Generative Modeling from Black-box Corruptions via Self-Consistent Stochastic Interpolants},
  author={Modi, Chirag and Han, Jiequn and Vanden-Eijnden, Eric and Bruna, Joan},
  booktitle={International Conference on Learning Representations (ICLR)},
  url={https://arxiv.org/abs/2512.10857},
  year={2026}
}
```

## License

Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0) — see [LICENSE](LICENSE). This matches the license of the vendored NVIDIA EDM code under `src/` (`networks.py`, `generate.py`, `dnnlib.py`, `torch_utils/`).
