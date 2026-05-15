import sys, os
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision('high')
import json
from functools import partial

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
from utils import count_parameters, make_serializable
from mlps import FeedForwardwithEMB
from custom_datasets import get_dataset, CorruptedDataset
from interpolant_utils import SCSInterpolant
from callbacks import save_fig_2dsynt_projection
from trainer_si import Trainer
from paths import default_data_root, default_results_root, build_run_slug
import forward_maps as fwd_maps
import argparse

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("DEVICE : ", device)

parser = argparse.ArgumentParser(description="")
parser.add_argument("--dataset", type=str, default="two_moons", help="dataset (two_moons)")
parser.add_argument("--data_root", type=str, default=default_data_root(), help="root dir for dataset caches (default: $SCSI_DATA or ./data)")
parser.add_argument("--dataset_seed", type=int, default=42, help="seed for synthetic dataset generation")
parser.add_argument("--n_samples", type=int, default=10000, help="number of synthetic samples to generate/cache")
parser.add_argument("--corruption", type=str, default="gaussian_noise", help="corruption")
parser.add_argument("--corruption_levels", type=float, nargs='+', default=[0.5], help="corruption level")
parser.add_argument("--fc_width", type=int, default=256, help="width of the feedforward network")
parser.add_argument("--fc_depth", type=int, default=3, help="depth of the feedforward network")
parser.add_argument("--t_emb_dim", type=int, default=32, help="time embedding dim for FeedForwardwithEMB")
parser.add_argument("--alpha", type=float, default=0.9, help="prob of newly-transported pseudo-clean vs prior estimate")
parser.add_argument("--resamples", type=int, default=1, help="resamples per loss eval in SCSInterpolant")
parser.add_argument("--gamma_scale", type=float, default=0.0, help="gaussian noise level in the interpolant")
parser.add_argument("--train_steps", type=int, default=40000, help="number of channels in model")
parser.add_argument("--batch_size", type=int, default=2000, help="batch size")
parser.add_argument("--learning_rate", type=float, default=1e-3, help="learning rate")
parser.add_argument("--update_transport_every", type=int, default=1, help="continued training count")
parser.add_argument("--prefix", type=str, default='', help="prefix for folder name")
parser.add_argument("--suffix", type=str, default='', help="suffix for folder name")
parser.add_argument("--lr_scheduler", action='store_true', help="use scheduler if provided, else not")
parser.add_argument("--clean_data_steps", type=int, default=-1, help="number of clean data steps to use in training")
parser.add_argument("--ode_steps", type=int, default=40, help="ode steps")
parser.add_argument("--save_and_sample_every", type=int, default=1000, help="save and sample every n steps")
parser.add_argument("--model_path", type=str, default='latest', help="which model to load")
parser.add_argument("--resume_count", type=int, default=1, help="continued training count")
parser.add_argument("--results_root", type=str, default=default_results_root(), help="root dir for training outputs (default: $SCSI_RESULTS or ./results)")

# Parse arguments
args = parser.parse_args()
# args = parser.parse_args(['--corruption_levels', '2', '0.01',
#                           '--suffix', 'test'])

print(args)
train_num_steps = args.train_steps
save_and_sample_every = args.save_and_sample_every
batch_size = args.batch_size
update_transport_every = args.update_transport_every
lr = args.learning_rate
lr_scheduler = args.lr_scheduler

# Parse corruption arguments
corruption = args.corruption # to fix
corruption_levels = args.corruption_levels
try:
    fwd_func = fwd_maps.corruption_dict[corruption](*corruption_levels)
except Exception as e:
    print("Exception in loading corruption function : ", e)
    sys.exit()
folder = build_run_slug(args)
results_folder = f"{args.results_root}/{folder}/"
os.makedirs(results_folder, exist_ok=True)
print(f"Results will be saved in folder: {results_folder}")
use_latents, latent_dim = fwd_maps.parse_latents(corruption, None)
args_dict = make_serializable(vars(args) if isinstance(args, argparse.Namespace) else args)
with open(f"{results_folder}/args.json", "w") as f:
    json.dump(args_dict, f, indent=4)

# Initialize model and train
interpolant = SCSInterpolant(
    fwd_func,
    use_latents=use_latents,
    n_steps=args.ode_steps,
    alpha=args.alpha,
    resamples=args.resamples,
    diffusion_coeff=0.0,
    gamma_scale=args.gamma_scale
).to(device)
if args.dataset == "two_moons":
    dim_in = 2
    clean_dataset, _, _ = get_dataset(args.dataset, args.data_root, seed=args.dataset_seed, n_samples=args.n_samples)
    dataset = CorruptedDataset(clean_dataset, fwd_func, tied_rng=False)
    dataloader = None
    save_fig_fn = partial(save_fig_2dsynt_projection, corruption_name=args.corruption)
    clean_data_valid = clean_dataset.array.to(device)
else:
    raise ValueError(f"Unknown dataset: {args.dataset}")
corrupted_valid, latents_valid = interpolant.push_fwd(clean_data_valid, return_latents=True)
latents_valid = latents_valid if use_latents else None
if args.corruption.startswith("projection") and use_latents:
    latent_dim = dim_in * int(args.corruption_levels[0])
else:
    latent_dim = None
valid_data_plot = (clean_data_valid, corrupted_valid, latents_valid)

b =  FeedForwardwithEMB(dim_in, args.t_emb_dim, [args.fc_width]*args.fc_depth, latent_dim=latent_dim).to(device)
print("Parameter count : ", count_parameters(b))

trainer = Trainer(model=b,
        interpolant=interpolant,
        dataloader=dataloader,
        dataset=dataset,
        train_batch_size = batch_size,
        update_transport_every = update_transport_every,
        gradient_accumulate_every = 1,
        train_lr = lr,
        lr_scheduler = lr_scheduler,
        train_num_steps = train_num_steps,
        save_and_sample_every= save_and_sample_every,
        results_folder=results_folder,
        clean_data_steps=args.clean_data_steps,
        callback_fn=save_fig_fn,
        validation_data=valid_data_plot,
        num_workers=0,
        )

losses = trainer.train()
