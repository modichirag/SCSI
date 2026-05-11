# SCSI — Self-Consistent Stochastic Interpolants

Official implementation of **Generative Modeling from Black-box Corruptions via Self-Consistent Stochastic Interpolants** (Chirag Modi, Jiequn Han, Eric Vanden-Eijnden, Joan Bruna — ICLR 2026). [[link]](https://arxiv.org/abs/2512.10857)

Research code for training generative models (diffusion / stochastic interpolants) from **corrupted** observations — i.e. learning a clean-data prior when only degraded samples are available. Corruptions include Gaussian noise, random/block masking, Gaussian/motion blur, JPEG compression, and random projections. Targets CIFAR-10, MNIST, CelebA, SDSS DR16 quasar spectra (1-D), and 2-D synthetic distributions (`two_moons`, `checkerboard`).

## Installation

```bash
pip install -r requirements.txt
```

Python 3.9+ and a recent PyTorch (CUDA for serious training; CPU is fine for the 2-D MLP smoke tests) are expected.

## Quickstart: 2-D synthetic smoke test

Runs on CPU in a couple of minutes and exercises the full training loop.

```bash
python mlp_interpolants.py \
    --dataset two_moons --corruption projection_coeff \
    --corruption_levels 1 0.1 \
    --train_steps 2000 --batch_size 500 \
    --suffix smoke
```

Outputs (loss curve, intermediate denoising snapshots, final model) land under `./results/two_moons-projection_coeff-1.00-0.10-smoke/`.

## A real example: CIFAR-10 with random masking

```bash
python -u scsi.py \
    --dataset cifar10 --corruption random_mask \
    --corruption_levels 0.5 0.0 1.0 \
    --train_steps 50000 --channels 32 --ode_steps 64 \
    --alpha 0.9 --resamples 2 --learning_rate 5e-4 --lr_scheduler
```

CIFAR-10 downloads to `$SCSI_DATA` (default `./data/cifar10/`) on first call. Outputs go to `$SCSI_RESULTS/singleview/...` (or `multiview/...` with `--multiview`).

Distributed (single-node, multi-GPU) training uses `torchrun`:

```bash
torchrun --standalone --nproc_per_node=4 scsi_distributed.py [args...]
```

## Paths

Two environment variables (also exposed as CLI flags) control where data and outputs live:

| Variable | CLI flag | Default | Purpose |
|---|---|---|---|
| `SCSI_DATA` | `--data_root` | `./data` | Dataset caches (MNIST, CIFAR-10, CelebA, synthetic, QSO) |
| `SCSI_RESULTS` | `--results_root` | `./results` | Training outputs, `model-best.pt`, logs |

## Datasets

All drivers pull data via `get_dataset(name, data_root, seed=42)` from `src/custom_datasets.py`. Supported names:

- `mnist`, `cifar10` — downloaded via `torchvision` on first call.
- `celebA` — **not auto-downloaded**; place `img_align_celeba/` under `$SCSI_DATA/celebA/`.
- `two_moons`, `checkerboard` — synthetic; cached under `$SCSI_DATA/<name>/seed_<seed>/`.
- `qso` — SDSS DR16 quasar spectra; downloaded via `src/qso_download.py` on first call (small default: 1000 spectra in redshift range 2.75–3.25; configurable via `--max_spectra`, `--z_min`, `--z_max` on `qsos.py`). Requires `astropy`.

To pre-download the QSO cache:

```bash
python scripts/download_qso.py --data-root ./data --max-spectra 1000
```

## Driver map

Top-level drivers add `<repo>/src` to `sys.path` via a `__file__`-relative `sys.path.append`, so they can be invoked from any working directory — no install step needed.

| Script | Purpose |
|---|---|
| `mlp_interpolants.py` | 2-D synthetic MLP experiments (`two_moons`, `checkerboard`) |
| `scsi.py` | Single-GPU training from corrupted images |
| `scsi_distributed.py` | DDP variant via `torchrun` |
| `awgn.py` | Specialization for the additive-Gaussian-noise case |
| `qsos.py` | 1-D quasar-spectra variant; uses `KarrasUnet1D` |
| `train.py`, `sample.py`, `fid_eval.py` | Plain EDM diffusion baseline (clean data only) |
| `fid_eval_interpolants.py`, `fid_eval_awgn.py`, `fid_eval_dps.py` | FID evaluation of trained interpolants / DPS baseline |
| `lpips_eval_interpolants.py`, `lpips_eval_dps.py` | LPIPS / PSNR / SSIM evaluation |
| `clean_interpolants.py` | Produces "cleaned" samples from a trained interpolant |

The SLURM scripts `job-cifar10.sh`, `job-dist.sh`, `job-sample.sh` are canonical invocation examples.

## Repository layout

```
src/                          # Library code — not an installable package yet
  custom_datasets.py          # Dataset registry, CorruptedDataset, QSODataset
  forward_maps.py             # Corruption factories (corruption_dict)
  interpolant_utils.py        # SCSInterpolant and variants
  trainer_si.py               # Production training loop (EMA, DDP, checkpointing)
  trainer.py                  # Older/simpler loop used by train.py (EDM baseline)
  networks.py, karras_unet*.py, mlps.py
  paths.py                    # SCSI_DATA / SCSI_RESULTS resolution
  qso_download.py             # SDSS DR16 spectra downloader
scripts/                      # CLI wrappers (currently just QSO download)
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
