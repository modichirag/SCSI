import torch
import sys, os
import json
import argparse
from torch.utils.data import DataLoader, Dataset
from ema_pytorch import EMA

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
from networks import EDMPrecond
from custom_datasets import get_dataset, ImagesOnly
from fid_evaluation import FIDEvaluation
from utils import cycle
from generate import edm_sampler
from paths import default_data_root, default_results_root

# Create an ArgumentParser object
parser = argparse.ArgumentParser(description="")
parser.add_argument("--folder", type=str, help="Path")
parser.add_argument("--dataset", type=str, help="dataset")
parser.add_argument("--data_root", type=str, default=default_data_root(), help="root dir for dataset caches (default: $SCSI_DATA or ./data)")
parser.add_argument("--results_root", type=str, default=default_results_root(), help="root dir for training outputs (default: $SCSI_RESULTS or ./results)")
parser.add_argument("--channels", type=int, default=64, help="number of channels in model")
parser.add_argument("--model", type=str, default='best', help="which saved model in folder")
parser.add_argument("--n_samples", type=int, default=50_000, help="Samples to evalaute FID")
parser.add_argument("--batch_size", type=int, default=256, help="batch size")


# Parse arguments
args = parser.parse_args()
print(args)
folder = f"{args.results_root}/{args.folder}/"
dataset, D, nc = get_dataset(args.dataset, args.data_root)
image_dataset = ImagesOnly(dataset)
model_channels = args.channels #192
print(D, nc)

#Other parameters
batch_size = args.batch_size
n_samples = args.n_samples
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("DEVICE : ", device)


print("Setup model and dataloader")
model = EDMPrecond(D, nc, model_channels=model_channels).to(device)
image_dataset = ImagesOnly(dataset)
dl = DataLoader(image_dataset, batch_size=batch_size, shuffle = True, pin_memory = True, num_workers = 1) 
dl = cycle(dl)
fid_scorer = FIDEvaluation(
    batch_size=batch_size,
    dl=dl,
    channels=nc,
    accelerator=None, #self.accelerator,
    stats_dir=folder,
    device=device,
    num_fid_samples=n_samples,
    inception_block_idx=2048
)
sampling_scheme = lambda net, latents: edm_sampler(net, latents)

#Load model

data = torch.load(f'{folder}/model-{args.model}.pt', map_location=device, weights_only=True)
model.load_state_dict(data['model'])
print("Model loaded")
score = fid_scorer.fid_score(model, sampling_scheme=sampling_scheme, force_calc=True)
print(f"FID score of loaded best model : {score}")

#Load EMA model
ema = EMA(model).to(device)
ema.load_state_dict(data['ema'])
print("Model loaded")
score_ema = fid_scorer.fid_score(ema.ema_model, sampling_scheme=sampling_scheme)
print(f"FID score of corresponding ema model: {score_ema}")

to_save = {'FID_best': score, 'FID_best_ema':score_ema}
n = int(n_samples/1e3)
save_name = f"{folder}/fid_{n}k-{args.model}.json"
with open(save_name, 'w') as file:
        json.dump(to_save, file, indent=4)
