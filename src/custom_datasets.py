import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F
from PIL import Image
import os
from pathlib import Path

class ImagesOnly(Dataset):
        def __init__(self, base_dataset):
            self.base = base_dataset

        def __len__(self):
            return len(self.base)

        def __getitem__(self, idx):
            img, _ = self.base[idx]
            return img


class ImageOnlyFolder(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_files = sorted([
            f for f in os.listdir(image_dir)
            if f.endswith(('.png', '.jpg', '.jpeg'))
        ])
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        img = Image.open(img_path).convert('RGB')  # ensure 3 channels
        if self.transform:
            img = self.transform(img)
        return img


# 1) Define your transforms
mnist_transforms = transforms.Compose([
    transforms.Pad(2),                              # [0,255]→[0,1]
    transforms.ToTensor(),                              # [0,255]→[0,1]
    transforms.Normalize((0.1307,), (0.3081,))          # mean/std for MNIST
])

mnist_transforms_raw = transforms.Compose([
    transforms.Pad(2),                              # [0,255]→[0,1]
    transforms.ToTensor(),                              # [0,255]→[0,1]
])

mnist_inverse_transforms = transforms.Compose([
    transforms.Normalize(                              # mean/std for CIFAR-10
            mean=(0.),
            std=(1/0.3081)
    ),
    transforms.Normalize(                              # mean/std for CIFAR-10
            mean=(-0.1307),
            std=(1.)
    )
])

cifar10_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),                  # data aug only on train
    #transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize(                              # mean/std for CIFAR-10
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616)
    ),
])

cifar10_inverse_transforms = transforms.Compose([
    transforms.Normalize(                              # mean/std for CIFAR-10
            mean=(0., 0., 0.),
            std=(1/0.2470, 1/0.2435, 1/0.2616)
    ),
    transforms.Normalize(                              # mean/std for CIFAR-10
            mean=(-0.4914, -0.4822, -0.4465),
            std=(1., 1., 1.)
    ),
])

celebA_inverse_transforms = transforms.Compose([
    transforms.Normalize(                              # mean/std for CIFAR-10
            mean=(0., 0., 0.),
            std=(1/0.5, 1/0.5, 1/0.5)
    ),
    transforms.Normalize(                              # mean/std for CIFAR-10
            mean=(-0.5, -0.5, -0.5),
            std=(1., 1., 1.)
    )
])

cifar10_transforms_raw = transforms.Compose([
        # transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
    ])


celebA_transforms = transforms.Compose([
    transforms.CenterCrop(178),
    transforms.Resize(64),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),  # → [-1, 1]
])


class NumpyArrayDataset(Dataset):
    def __init__(self, data_array, transform=None):
        self.data = torch.from_numpy(data_array).float()  # or .long() for labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.transform is not None:
            return self.transform(self.data[idx])
        else:
            return self.data[idx]


class CombinedNumpyDataset(Dataset):
    def __init__(self, folder, transform=None):
        import os
        files = os.listdir(folder)
        file_list = [os.path.join(folder, f) for f in files if f.endswith('.npy')]
        self.data = [np.load(f) for f in file_list]  # load into memory
        self.cumsum = np.cumsum([len(arr) for arr in self.data])
        self.transform = transform

    def __len__(self):
        return self.cumsum[-1]

    def __getitem__(self, idx):
        # figure out which array and local index
        file_idx = np.searchsorted(self.cumsum, idx, side='right')
        local_idx = idx if file_idx == 0 else idx - self.cumsum[file_idx - 1]
        x = self.data[file_idx][local_idx]
        x = torch.from_numpy(x)
        if self.transform is not None:
            x = self.transform(x)
        return x


class CorruptedDataset(Dataset):
    def __init__(self, base_dataset, corruption_fn, tied_rng=True, base_seed: int = 0):
        """
        base_dataset   : any Dataset returning (img, label)
        corruption_fn  : fn(img, *, generator) -> img_corrupted
        base_seed      : optional global offset for all seeds
        """
        self.base = base_dataset
        self.corrupt = corruption_fn
        self.base_seed = base_seed
        self.tied_rng = tied_rng

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img = self.base[idx]
        # make a fresh generator, seed it with (base_seed + idx)
        if self.tied_rng:
            gen = torch.Generator()
            gen.manual_seed(self.base_seed + idx)
        else:
            gen = None
        # apply your corruption; it must accept a `generator` kwarg
        img_corrupted, latents = self.corrupt(img, return_latents=True, generator=gen)
        return img, img_corrupted, latents


# -----------------------------------------------------------------------------
# Unified dataset registry
# -----------------------------------------------------------------------------
# New API: get_dataset(name, data_root, seed=42) -> (dataset, D, nc)
#
# Each dataset downloads or generates into `data_root` on first call and
# returns the cached copy on subsequent calls. Synthetic datasets cache
# under `data_root/<name>/seed_<seed>/` so multiple seeds can coexist.
# Image/QSO datasets cache under `data_root/<name>/`.


class SyntheticDataset(Dataset):
    """In-memory dataset wrapping a numpy array of shape (N, ...)."""

    def __init__(self, array):
        self.array = torch.from_numpy(np.asarray(array)).float()

    def __len__(self):
        return len(self.array)

    def __getitem__(self, idx):
        return self.array[idx]


def _fetch_mnist(data_root, seed=None):
    return datasets.MNIST(
        root=str(Path(data_root) / "mnist"),
        train=True,
        download=True,
        transform=mnist_transforms,
    )


def _fetch_cifar10(data_root, seed=None):
    return datasets.CIFAR10(
        root=str(Path(data_root) / "cifar10"),
        train=True,
        download=True,
        transform=cifar10_transforms,
    )


def _fetch_celebA(data_root, seed=None):
    # CelebA images are too large / licensed to auto-download. Place the
    # `img_align_celeba/` folder under `{data_root}/celebA/`.
    folder = Path(data_root) / "celebA" / "img_align_celeba"
    if not folder.exists():
        raise FileNotFoundError(
            f"CelebA images not found at {folder}. Download img_align_celeba.zip "
            "from https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html and extract "
            f"it so images live in {folder}."
        )
    return ImageOnlyFolder(str(folder), transform=celebA_transforms)


def _fetch_two_moons(data_root, seed=42, n_samples=10000, noise=0.1):
    cache = Path(data_root) / "two_moons" / f"seed_{seed}" / f"n_{n_samples}.npy"
    if not cache.exists():
        from sklearn.datasets import make_moons
        cache.parent.mkdir(parents=True, exist_ok=True)
        X, _ = make_moons(n_samples=n_samples, noise=noise, random_state=seed)
        X = (4.0 * (X - 0.5)).astype(np.float32)  # preserve original scaling
        np.save(cache, X)
    return SyntheticDataset(np.load(cache))


def _fetch_checkerboard(data_root, seed=42, n_samples=10000):
    cache = Path(data_root) / "checkerboard" / f"seed_{seed}" / f"n_{n_samples}.npy"
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        gen = torch.Generator().manual_seed(seed)
        x1 = torch.rand(n_samples, generator=gen) * 4 - 2
        x2_ = torch.rand(n_samples, generator=gen) - torch.randint(2, (n_samples,), generator=gen) * 2
        x2 = x2_ + (torch.floor(x1) % 2)
        X = (torch.cat([x1[:, None], x2[:, None]], dim=1) * 2).numpy().astype(np.float32)
        np.save(cache, X)
    return SyntheticDataset(np.load(cache))


class QSODataset(Dataset):
    """QSO spectra dataset.

    Items are flux arrays of shape (length,). The shared log-wavelength
    grid is exposed as the `loglam` attribute, since all spectra are
    clipped to the same LOGLAM_MIN/MAX window in `qso_download.py`.
    """

    def __init__(self, flux, loglam):
        self.flux = np.asarray(flux, dtype=np.float32)
        self.loglam = np.asarray(loglam, dtype=np.float32)

    def __len__(self):
        return len(self.flux)

    def __getitem__(self, idx):
        return torch.from_numpy(self.flux[idx])


def _fetch_qso(data_root, seed=None, max_spectra=1000, z_range=(2.75, 3.25)):
    from qso_download import download_qso
    path = download_qso(
        data_root, z_min=z_range[0], z_max=z_range[1], max_spectra=max_spectra
    )
    arr = np.load(path)  # (N, 3, 2999): channels = (loglam, flux, ivar)
    return QSODataset(flux=arr[:, 1, :], loglam=arr[0, 0, :])


DATASETS = {
    "mnist":        {"fetch": _fetch_mnist,        "D": 32,   "nc": 1},
    "cifar10":      {"fetch": _fetch_cifar10,      "D": 32,   "nc": 3},
    "celebA":       {"fetch": _fetch_celebA,       "D": 64,   "nc": 3},
    "two_moons":    {"fetch": _fetch_two_moons,    "D": 2,    "nc": 1},
    "checkerboard": {"fetch": _fetch_checkerboard, "D": 2,    "nc": 1},
    "qso":          {"fetch": _fetch_qso,          "D": 2999, "nc": 1},
}


def get_dataset(name, data_root, seed=42, **fetch_kwargs):
    """Load (or download/generate) a dataset by name.

    Args:
        name: one of DATASETS keys.
        data_root: root directory for dataset caches.
        seed: RNG seed for synthetic datasets; ignored for others.
        **fetch_kwargs: forwarded to the dataset's fetch function
            (e.g. `max_spectra`, `z_range` for qso).

    Returns:
        (dataset, D, nc): dataset object plus spatial dim and channel count.
    """
    if name not in DATASETS:
        raise KeyError(f"Unknown dataset: {name!r}. Available: {sorted(DATASETS)}")
    entry = DATASETS[name]
    dataset = entry["fetch"](data_root, seed=seed, **fetch_kwargs)
    return dataset, entry["D"], entry["nc"]
