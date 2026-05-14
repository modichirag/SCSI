
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


def save_fig_2dsynt_projection(idx, b, interpolant, dataloader, device, results_folder, losses, validation_data, s=None, corruption_name=None, **_kwargs):
    # 4-panel scatter for 2D synthetic datasets (two_moons, checkerboard) under
    # projection_{coeff,vec}. All forward maps in this family return corrupted
    # samples of shape (N, dim_in=2), so panels Clean/Restored always scatter
    # directly. For Corrupted/Restored-corrupted: projection_vec and full-rank
    # coeff are scattered directly; projection_coeff with dim_out=1 overrides
    # the y-axis with the angle of A (atan2 of the unit direction), since the
    # raw second coord of y is just random Gaussian padding and uninformative.
    clean, corrupted, latents, restored = get_samples(b, interpolant, dataloader, device, validation_data, s=s)
    push_fwd_func = interpolant.push_fwd
    c = '#62508f'

    use_angle = (corruption_name == "projection_coeff"
                 and latents is not None and latents.shape[-2] == 1)

    def panel_xy(y, A):
        if use_angle:
            A_sq = A.squeeze(-2)
            angle = torch.atan2(A_sq[:, 1], A_sq[:, 0])
            return grab(y[:, 0]), grab(angle)
        y_np = grab(y)
        return y_np[:, 0], y_np[:, 1]

    clean = grab(clean)
    restored = grab(restored)
    cor_x, cor_y = panel_xy(corrupted, latents)

    restored_corrupted, latents_new = push_fwd_func(torch.from_numpy(restored), return_latents=True)
    rc_x, rc_y = panel_xy(restored_corrupted, latents_new)

    def style_ambient(ax):
        ax.set_xlim(-6, 6); ax.set_ylim(-6, 6)
        ax.set_xticks([-4, 0, 4]); ax.set_yticks([-4, 0, 4])

    def style_angle(ax):
        ax.set_xlim(-6, 6); ax.set_ylim(-3.3, 3.3)
        ax.set_xticks([-4, 0, 4]); ax.set_yticks([-3, 0, 3])

    style_corrupted = style_angle if use_angle else style_ambient

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].scatter(clean[:, 0], clean[:, 1], alpha=0.03, c=c)
    axes[0].set_title("Clean samples", fontsize=18); style_ambient(axes[0])

    axes[1].scatter(cor_x, cor_y, alpha=0.03, c=c)
    axes[1].set_title("Corrupted samples", fontsize=18); style_corrupted(axes[1])

    axes[2].scatter(restored[:, 0], restored[:, 1], alpha=0.03, c=c)
    axes[2].set_title("Restored samples", fontsize=18); style_ambient(axes[2])

    axes[3].scatter(rc_x, rc_y, alpha=0.03, c=c)
    axes[3].set_title("Restored corrupted samples", fontsize=18); style_corrupted(axes[3])

    plt.subplots_adjust(wspace=0.0, hspace=0.0)
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
