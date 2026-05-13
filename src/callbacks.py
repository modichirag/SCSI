
import torch
import os
import matplotlib.pyplot as plt
import numpy as np
from utils import grab, push_to_device

## Expected signature of callback_fn
# callback_fn(milestone, b, interpolant,
#               dataloader, validation_data, losses, device, results_folder)

def get_samples(b, interpolant, dataloader, device, validation_data, s=None):
    if validation_data is None:
        data, obs, latents = next(dataloader)
    else:
        data, obs, latents = validation_data
    if interpolant.use_latents:
        data, obs, latents = push_to_device(data, obs, latents, device=device)
    else:
        data, obs = push_to_device(data, obs, device=device)
        latents = None
    restored = interpolant.transport(b, obs, latents, s=s)
    return data, obs, latents, restored



def save_image(idx, b, interpolant, dataloader, device, results_folder, losses, validation_data, s=None, **_kwargs):

    data, obs, latents, restored = get_samples(b, interpolant, dataloader, device, validation_data, s=s)
    to_show = [data, obs, restored]

    fig, axar = plt.subplots(len(to_show), 8, figsize=(8, 3), sharex=True, sharey=True)
    vmax, vmin = data.max()*1.1, data.min()*0.5
    for i in range(len(to_show)):
        for j in range(8):
            ax = axar[i, j]
            im = ax.imshow(grab(to_show[i][j]).transpose(1, 2, 0), vmax=vmax, vmin=vmin)
    axar[0, 0].set_ylabel('Original')
    axar[1, 0].set_ylabel(f'Corrupted')
    axar[2, 0].set_ylabel(f'Restored')
    for axis in axar.flatten():
        axis.set_xticks([])
        axis.set_yticks([])
    plt.subplots_adjust(wspace=0.0, hspace=0.0)
    plt.savefig(f'{results_folder}/denoising_{idx}.png', dpi=300)
    plt.close()


def save_fig_2dsynt_vec(idx, b, interpolant, dataloader, device, results_folder, losses, validation_data, s=None, **_kwargs):
    clean, corrupted, latents, restored = get_samples(b, interpolant, dataloader, device, validation_data, s=s)
    push_fwd_func = interpolant.push_fwd
    c = '#62508f' # plot color
    fig, axes = plt.subplots(1,4, figsize=(20, 5))

    clean = grab(clean)
    corrupted = grab(corrupted)
    restored = grab(restored)

    axes[0].scatter(clean[:,0], clean[:,1], alpha = 0.03, c = c)
    axes[0].set_title(r"Clean samples", fontsize = 18)
    axes[0].set_xlim(-6,6), axes[0].set_ylim(-6,6)
    axes[0].set_xticks([-4,0,4]), axes[0].set_yticks([-4,0,4])

    axes[1].scatter(corrupted[:,0], corrupted[:,1], alpha = 0.03, c = c)
    axes[1].set_title(r"Corrupted samples", fontsize = 18)
    axes[1].set_xlim(-6,6), axes[2].set_ylim(-6,6)
    axes[1].set_xticks([-4,0,4]), axes[2].set_yticks([]);

    axes[2].scatter(restored[:,0], restored[:,1], alpha = 0.03, c = c)
    axes[2].set_title(r"Restored samples ", fontsize = 18)
    axes[2].set_xlim(-6,6), axes[1].set_ylim(-6,6)
    axes[2].set_xticks([-4,0,4]), axes[1].set_yticks([])

    restored_corrupted = push_fwd_func(torch.from_numpy(restored)).numpy()
    axes[3].scatter(restored_corrupted[:,0], restored_corrupted[:,1], alpha = 0.03, c = c)
    axes[3].set_title(r"Restored corrupted samples ", fontsize = 18)
    axes[3].set_xlim(-6,6), axes[3].set_ylim(-6,6)
    axes[3].set_xticks([-4,0,4]), axes[3].set_yticks([])

    plt.subplots_adjust(wspace=0.0, hspace=0.0)  # Reduce spacing
    # plt.tight_layout()
    plt.savefig(f'{results_folder}/denoising_{idx}.png', dpi=300)
    plt.close()


def save_fig_2dsynt_coeff(idx, b, interpolant, dataloader, device, results_folder, losses, validation_data, s=None, **_kwargs):
    clean, corrupted, latents, restored = get_samples(b, interpolant, dataloader, device, validation_data, s=s)
    push_fwd_func = interpolant.push_fwd

    c = '#62508f' # plot color
    push_fwd_func = interpolant.push_fwd
    assert latents is not None, "Latents should be provided for this function"
    latents = latents.squeeze()
    assert latents.shape[-1] == 2, "Latents should be 2D for this function"
    angles_rad = grab(torch.atan2(latents[:, 1], latents[:, 0]))
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    # if push_fwd_func is None:
    #     fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    # else:
    #     fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    clean = grab(clean)
    corrupted = grab(corrupted)
    restored = grab(restored)

    axes[0].scatter(clean[:,0], clean[:,1], alpha = 0.03, c = c)
    axes[0].set_title(r"Clean samples", fontsize = 18)
    axes[0].set_xlim(-6,6), axes[0].set_ylim(-6,6)
    axes[0].set_xticks([-4,0,4]), axes[0].set_yticks([-4,0,4])

    axes[1].scatter(corrupted[:,0], angles_rad, alpha = 0.03, c = c)
    axes[1].set_title(r"Corrupted samples", fontsize = 18)
    axes[1].set_xlim(-6,6), axes[2].set_ylim(-6,6)
    axes[1].set_xticks([-4,0,4]), axes[2].set_yticks([]);

    axes[2].scatter(restored[:,0], restored[:,1], alpha = 0.03, c = c)
    axes[2].set_title(r"Restored samples ", fontsize = 18)
    axes[2].set_xlim(-6,6), axes[1].set_ylim(-6,6)
    axes[2].set_xticks([-4,0,4]), axes[1].set_yticks([])

    restored_corrupted, latents_new = push_fwd_func(torch.from_numpy(restored), return_latents=True)
    restored_corrupted = grab(restored_corrupted)
    latents_new = latents_new.squeeze()
    angles_rad_new = grab(torch.atan2(latents_new[:, 1], latents_new[:, 0]))
    axes[3].scatter(restored_corrupted[:,0], angles_rad_new, alpha = 0.03, c = c)
    axes[3].set_title(r"Restored corrupted samples ", fontsize = 18)
    axes[3].set_xlim(-6,6), axes[3].set_ylim(-6,6)
    axes[3].set_xticks([-4,0,4]), axes[3].set_yticks([])

    plt.subplots_adjust(wspace=0.0, hspace=0.0)  # Reduce spacing
    plt.savefig(f'{results_folder}/denoising_{idx}.png', dpi=300)
    plt.close()


def save_losses_fig(losses, results_folder):
    steps = np.arange(len(losses))
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    axs[0].semilogy(steps, losses, marker='.', linestyle='-', markersize=4, alpha=0.7)
    axs[0].set_xlabel("Steps")
    axs[0].set_ylabel("Loss (log scale)")
    axs[0].set_title("Loss Curve (Semi-Log Y Scale)")
    axs[0].grid(True, which="both", ls="--", alpha=0.5)
    axs[1].loglog(steps, losses, marker='.', linestyle='-', markersize=4, alpha=0.7, color='orangered')
    axs[1].set_xlabel("Steps (log scale)")
    axs[1].set_ylabel("Loss (log scale)")
    axs[1].set_title("Loss Curve (Log-Log Scale)")
    axs[1].grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    print(os.path.join(results_folder, 'losses.png'))
    plt.savefig(os.path.join(results_folder, 'losses.png'), dpi=300, bbox_inches='tight')
    plt.close()
