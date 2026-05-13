import numpy as np
import torch
from torch import nn
from torch.func import vmap

from networks import Linear, PositionalEmbedding


class SimpleFeedForward(nn.Module):
    def __init__(
        self, dim, hidden_sizes = [256, 256], activation=torch.nn.ReLU, latent_dim=None
    ):
        super().__init__()
        self.latent_dim = latent_dim if latent_dim is not None else 0
        layers = []
        prev_dim = dim + self.latent_dim + 1 # 1 for t
        for hidden_size in hidden_sizes:
            layers.append(torch.nn.Linear(prev_dim, hidden_size))
            layers.append(activation())
            prev_dim = hidden_size

        # final layer
        layers.append(torch.nn.Linear(prev_dim, dim))

        # Wrap all layers in a Sequential module
        self.net = torch.nn.Sequential(*layers)

    def _single_forward(self, x, t, latent):
        t = t.unsqueeze(-1)
        return self.net(torch.cat((x, t, latent)))

    def forward(self, x, t, latents=None):
        batch_size = x.shape[0]
        if latents is not None:
            if latents.shape[0] != batch_size:
                raise ValueError(f"Latents batch size {latents.shape[0]} does not match x/t batch size {batch_size}")
            if latents[0].numel() != self.latent_dim:
                raise ValueError(f"Latents feature dimension {latents[0].numel()} does not match model's feature_dim_latent {self.latent_dim}")
            latents = latents.reshape(batch_size, -1)
        else:
            latents = torch.zeros(x.shape[0], self.latent_dim, device=x.device, dtype=x.dtype)
        return vmap(self._single_forward, in_dims=(0, 0, 0), out_dims=(0))(x, t, latents)


class FeedForwardwithEMB(nn.Module):
    def __init__(
        self, dim, emb_channels=64, hidden_sizes = [256, 256], activation=torch.nn.SiLU, latent_dim=None
    ):
        super().__init__()
        self.latent_dim = latent_dim if latent_dim is not None else 0
        self.emb_channels = emb_channels
        layers = []
        prev_dim = dim + emb_channels # emb for time
        if self.latent_dim > 0:
            prev_dim += emb_channels # emb for latent
        for hidden_size in hidden_sizes:
            layers.append(torch.nn.Linear(prev_dim, hidden_size))
            layers.append(activation())
            prev_dim = hidden_size
        # final layer
        layers.append(torch.nn.Linear(prev_dim, dim))
        # Wrap all layers in a Sequential module
        self.final_net = torch.nn.Sequential(*layers)
        self.map_t = PositionalEmbedding(num_channels=emb_channels, max_positions=1)

        if self.latent_dim > 0:
            self.map_latents = torch.nn.Sequential(
                Linear(in_features=latent_dim, out_features=emb_channels, bias=False, init_mode='kaiming_normal', init_weight=np.sqrt(latent_dim)),
                torch.nn.SiLU(),
                Linear(in_features=emb_channels, out_features=emb_channels, bias=False, init_mode='kaiming_normal', init_weight=np.sqrt(latent_dim)),
                torch.nn.SiLU(),
                Linear(in_features=emb_channels, out_features=emb_channels, bias=False, init_mode='kaiming_normal', init_weight=np.sqrt(latent_dim))
            )

    def forward(self, x, t, latents=None):
        batch_size = x.shape[0]
        if latents is not None:
            if latents.shape[0] != batch_size:
                raise ValueError(f"Latents batch size {latents.shape[0]} does not match x/t batch size {batch_size}")
            if latents[0].numel() != self.latent_dim:
                raise ValueError(f"Latents feature dimension {latents[0].numel()} does not match model's feature_dim_latent {self.latent_dim}")
            latents = latents.reshape(batch_size, -1)
        t_emb = self.map_t(t)
        if self.latent_dim > 0:
            latent_emb = self.map_latents(latents)
            return self.final_net(torch.cat((x, t_emb, latent_emb), dim=-1))
        else:
            return self.final_net(torch.cat((x, t_emb), dim=-1))
