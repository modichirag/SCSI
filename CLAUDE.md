# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository purpose

Research code for training generative models (diffusion / stochastic interpolants) from **corrupted** observations — i.e. learning a clean-data prior when only degraded samples are available. Corruptions include Gaussian noise, random/block masking, Gaussian/motion blur, JPEG compression, and random projections. Targets CIFAR-10, MNIST, CelebA, quasar spectra (1-D), and 2-D synthetic distributions (two_moons, checkerboard).

## Running scripts

Top-level driver scripts prepend `<repo>/src` to `sys.path` via a `__file__`-relative `sys.path.append` — they can be run from any working directory. No install step is required; clone and run.

```bash
python -u scsi_image.py --dataset cifar10 --corruption random_mask \
    --corruption_level 0.5 0.0 1.0 --train_steps 50000 --channels 32 \
    --ode_steps 64 --alpha 0.9 --resamples 2 --learning_rate 5e-4 --lr_scheduler
```

Distributed (single-node, multi-GPU) training uses `torchrun`:

```bash
torchrun --standalone --nproc_per_node=4 scsi_distributed.py [args...]
```

There is no `requirements.txt`, `pyproject.toml`, or test suite. Dependencies are the Flatiron cluster `torchlatest` env used in the `job-*.sh` scripts: `torch`, `torchvision`, `ema_pytorch`, `transformers` (for `get_cosine_schedule_with_warmup`), `tqdm`, `numpy`, `matplotlib`, plus `lpips` / `torchmetrics`-style FID for eval scripts.

## Cluster / output paths

Paths are controlled by two env vars / CLI flags, resolved in `src/paths.py`:
- `SCSI_DATA` / `--data_root` — dataset cache root (default `./data`).
- `SCSI_RESULTS` / `--results_root` — training output root (default `./results`).

Multiview-aware drivers append `singleview/` or `multiview/` under `results_root` based on `--multiview`; plain drivers (`train.py`, `sample.py`, `fid_eval.py`, `fid_eval_stage.py`) write directly under `results_root`. Each run saves `args.json`, `model-best.pt`, and loss/FID artifacts into a folder whose name is auto-built from `{dataset}-{corruption}-{levels}-{options}-{suffix}` (see `scsi_image.py` for the naming convention). The DPS eval drivers (`fid_eval_dps.py`, `lpips_eval_dps.py`) also load their baseline EDM checkpoint from `{results_root}/{modelfolder}/model-{model}.pt`.

The SLURM scripts (`job-cifar10.sh`, `job-dist.sh`, `job-sample.sh`) are the canonical examples of invocation — copy flags from there rather than inventing new ones.

## Driver script → purpose map

Top-level scripts are **thin argparse + config wrappers**, not libraries. The real logic is in `src/`. Each driver roughly wires together a dataset, a corruption function, a model, an interpolant/loss class, and a `Trainer`:

- `train.py`, `sample.py`, `fid_eval.py` — plain EDM diffusion baseline (clean data only), using `EDMPrecond` + `VELoss` + `edm_sampler`.
- `scsi_image.py` — main single-GPU entry for learning from corrupted images via stochastic interpolants.
- `scsi_distributed.py` — same, but DDP via `torchrun`.
- `awgn.py` — specialization using `SCSInterpolantAWGN` for the additive-Gaussian-noise case.
- `qsos.py` — 1-D quasar-spectra application; uses `KarrasUnet1D`.
- `scsi_synthetic.py` — low-dim MLP experiments on 2-D synthetic distributions (`checker`, `moon`, `gmm`) using `FeedForwardwithEMB` + `Trainer`. Pulls distributions from `src/distribution.py` (streaming, not the registry); migrating to `get_dataset` for `two_moons` / `checkerboard` is pending.
- `clean_interpolants.py` — produces "cleaned" samples from a trained interpolant (used as input to warm-start runs).
- `fid_eval_*.py`, `lpips_eval_*.py`, `fid_eval_stage.py` — evaluation drivers; `_dps` variants benchmark diffusion posterior sampling baselines via `src/dps.py`.

New experiments typically start by copying one of these drivers and editing the corruption / dataset / model args.

## Core architecture in `src/`

The pipeline has four pluggable pieces; a driver picks one of each and hands them to `Trainer`.

**1. Dataset (`custom_datasets.py`).** Canonical API is `get_dataset(name, data_root, seed=42, **fetch_kwargs) -> (dataset, D, nc)`, backed by the `DATASETS` registry. Supported names: `mnist`, `cifar10`, `celebA`, `two_moons`, `checkerboard`, `qso`. Each entry has a `fetch(data_root, seed=...)` that downloads/generates into `data_root/<name>/` on first call (synthetic caches under `.../seed_<seed>/`), then loads from disk. `fetch_kwargs` are forwarded to the fetch function — used by `qsos.py` to pass `max_spectra` and `z_range`. QSO downloading lives in `src/qso_download.py` with a CLI wrapper at `scripts/download_qso.py`; `QSODataset` exposes both per-spectrum flux (as items) and the shared `loglam` grid (as an attribute). `CorruptedDataset` wraps a clean dataset and applies a forward map on-the-fly; `tied_rng` (controlled by `--multiview`) decides whether each epoch sees the same corruption realization per sample or a fresh one. `ImagesOnly` strips labels; `NumpyArrayDataset` / `CombinedNumpyDataset` wrap pre-saved arrays.

**2. Forward map (`forward_maps.py`).** `corruption_dict` registers every corruption as a **factory** that takes its levels and returns a callable `fwd(x, return_latents=..., cond_y=..., embed=...)`. `parse_latents(corruption, D, C, cond_y)` returns `(use_latents, latent_dim)` so the model knows its conditioning shape. When adding a corruption, register it here *and* extend `parse_latents`.

**3. Model (`networks.py`, `karras_unet.py`, `karras_unet_1d.py`, `mlps.py`).** `networks.py` is vendored EDM code (NVIDIA); the key class used everywhere is `ConditionalDhariwalUNet`, which accepts `latent_dim` so the corruption's auxiliary output (mask, blurred image, k-space indices, …) can be injected as conditioning. `EDMPrecond` is the baseline wrapper used by `train.py`. 1-D variants live in `karras_unet_1d.py`. MLP experiments use `SimpleFeedForward`/`FeedForwardwithEMB` in `mlps.py` (project-authored, kept separate from the NVIDIA-licensed `networks.py`) and `MLPResNet` / `MLPVelocityField` in `networks.py` / `interpolant_utils.py`.

**4. Interpolant + training loop (`interpolant_utils.py`, `trainer_si.py`).** `SCSInterpolant` (and variants `…Combined`, `…AWGN`, `…Follmer`) own the stochastic-interpolant loss, the transport step that produces pseudo-clean targets `x0` from corrupted `x`, and the ODE/SDE samplers (`euler`, `heun`). The key flags are:

- `--alpha` — probability of using newly-transported pseudo-clean data vs. prior estimate.
- `--resamples` — number of resampling iterations per loss eval.
- `--transport_steps` / `--n_transports` — how often the "frozen" transport model is refreshed and how many transports are stacked per batch.
- `--gamma_scale`, `--diffusion_coeff`, `--smodel`, `--combinedsde` — switch between deterministic (ODE) and stochastic (SDE) interpolants; `--smodel` adds a second score network `s_model` trained alongside `b`.
- `--cleansteps` — after this many steps, training switches to (or mixes in) genuinely clean data (only meaningful when clean data is actually available, e.g. warm-start).

`trainer_si.py`'s `Trainer` is the production loop: EMA via `ema_pytorch`, optional `get_cosine_schedule_with_warmup`, DDP support (`get_worker_info()` reads `RANK`/`WORLD_SIZE`), gradient accumulation (`gradient_accumulate_every = batch_size * n_transports / mini_batch_size`), `model-best.pt` / `model-latest.pt` checkpointing with keys `model`, `ema`, optionally `s_model`, `s_ema`. `trainer.py` is the older/simpler loop still used by the plain-EDM `train.py`.

Sampling for the EDM baseline goes through `generate.py::edm_sampler` (plus `edm_fine` preset for SDE). FID is computed via `src/fid_evaluation.py`. `src/torch_utils/` (distributed, persistence, training_stats, misc) is vendored from the EDM repo and used by `networks.py`.

## Conventions worth knowing

- When loading a saved checkpoint, drivers do `EMA(b).load_state_dict(remove_all_prefix(data['ema']))` then copy `ema.ema_model.state_dict()` into the raw model — the `remove_all_prefix` / `remove_orig_mod_prefix` helpers in `utils.py` strip DDP (`module.`) and `torch.compile` (`_orig_mod.`) prefixes. Mirror this pattern rather than calling `load_state_dict` directly on saved weights.
- Run folder names are load-bearing: eval scripts reconstruct them from the same flags, so changing the naming logic in one driver requires matching changes in the corresponding `fid_eval_*` / `lpips_eval_*` / `clean_interpolants.py`.
- `args.json` is written once per run — rerunning with different args to the same folder silently overwrites it.
- There are no tests and no linter config. Iteration is via SLURM job scripts + notebooks in `notebooks/`.

@claude.local.md
