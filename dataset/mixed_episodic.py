"""
Mixed episodic dataset: alternates ISIC and DermNet episodes.

Strategy
--------
Each __getitem__ call flips a (weighted) coin to decide whether this episode
is sampled from ISIC or DermNet classes. The two datasets are NEVER mixed
within a single episode — each episode uses classes from exactly one source.

Splits
------
  Train: 4 ISIC train classes + 15 DermNet train classes (mixed episodes)
  Val:   2 ISIC val classes   + 4 DermNet val classes   (mixed episodes)
  Test:  ISIC only (VASC, SCC) — unchanged

DermNet val classes are NEVER seen during training, so they provide a
genuine out-of-domain generalization signal alongside the ISIC val classes.
This gives a much less noisy checkpoint selection signal than ISIC-only val
(which has only 1 possible 2-way task: BKL vs DF).

Usage
-----
    from dataset.mixed_episodic import setup_mixed_dataloaders
    train_dl, val_dl = setup_mixed_dataloaders(config)
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from dataset.isic import (
    ISICEpisodicDataset,
    collate_episodic,
)
from dataset.dermnet import DermNetEpisodicDataset


class MixedEpisodicDataset(Dataset):
    """
    Wraps an ISIC dataset and a DermNet dataset (same split).
    Each episode is drawn from one or the other (never mixed).

    Args:
        isic_ds:        ISICEpisodicDataset
        dermnet_ds:     DermNetEpisodicDataset
        dermnet_ratio:  probability of sampling a DermNet episode (default 0.5)
        n_episodes:     total episodes per epoch
    """

    def __init__(
        self,
        isic_ds: ISICEpisodicDataset,
        dermnet_ds: DermNetEpisodicDataset,
        dermnet_ratio: float = 0.5,
        n_episodes: int = 10_000,
    ):
        self.isic_ds       = isic_ds
        self.dermnet_ds    = dermnet_ds
        self.dermnet_ratio = dermnet_ratio
        self.n_episodes    = n_episodes

        # Both datasets must produce embeddings of the same dimension
        assert isic_ds.embed_dim == dermnet_ds.embed_dim, (
            f"Embedding dim mismatch: ISIC={isic_ds.embed_dim} vs "
            f"DermNet={dermnet_ds.embed_dim}"
        )
        # Both must use the same n_ways for batching compatibility
        assert isic_ds.n_ways == dermnet_ds.n_ways, (
            f"n_ways mismatch: ISIC={isic_ds.n_ways} vs "
            f"DermNet={dermnet_ds.n_ways}"
        )
        self.embed_dim = isic_ds.embed_dim
        self.n_ways    = isic_ds.n_ways

        print(f"\nMixedEpisodicDataset: {n_episodes} episodes/epoch, "
              f"dermnet_ratio={dermnet_ratio:.0%}")
        print(f"  ISIC classes:    {len(isic_ds.split_class_names)}")
        print(f"  DermNet classes:  {len(dermnet_ds.class_names)}")

    def __len__(self):
        return self.n_episodes

    def __getitem__(self, idx: int):
        # Decide which dataset to sample from
        rng = np.random.default_rng()  # fresh RNG each call (training is stochastic)
        use_dermnet = rng.random() < self.dermnet_ratio

        if use_dermnet:
            dermnet_idx = idx % len(self.dermnet_ds)
            return self.dermnet_ds[dermnet_idx]
        else:
            isic_idx = idx % len(self.isic_ds)
            return self.isic_ds[isic_idx]


# ---------------------------------------------------------------------------
# Convenience: build mixed dataloaders
# ---------------------------------------------------------------------------

def setup_mixed_dataloaders(config):
    """
    Build train (mixed ISIC + DermNet) and val (mixed ISIC + DermNet) DataLoaders.

    Train uses ISIC train classes (4) + DermNet train classes (15).
    Val uses ISIC val classes (2) + DermNet val classes (4, held-out).

    Additional config fields (beyond those used by setup_isic_dataloaders):
        config.dermnet_data_root    default "data/dermnet"
        config.dermnet_ratio        probability of DermNet episode (default 0.5)
    """
    data_root       = getattr(config, "data_root",         "data/isic2019")
    dermnet_root    = getattr(config, "dermnet_data_root",  "data/dermnet")
    encoder         = getattr(config, "encoder_type",       "clip")
    n_ways          = getattr(config, "n_ways",             2)
    min_shots       = getattr(config, "min_num_context",    0)
    max_shots       = getattr(config, "max_num_context",    10)
    num_query       = getattr(config, "num_targets",        20)
    n_ep_train      = getattr(config, "n_episodes_train",   10_000)
    n_ep_val        = getattr(config, "n_episodes_val",     1_000)
    batch_size      = getattr(config, "batch_size",         32)
    use_knowledge   = getattr(config, "use_knowledge",      True)
    seed            = getattr(config, "seed",               42)
    num_workers     = getattr(config, "num_workers",        4)
    dermnet_ratio   = getattr(config, "dermnet_ratio",      0.5)

    # ── Training: ISIC train (4 classes) + DermNet train (15 classes) ─────
    isic_train_ds = ISICEpisodicDataset(
        split="train", data_root=data_root, encoder=encoder,
        n_ways=n_ways, min_shots=min_shots, max_shots=max_shots,
        num_query=num_query, n_episodes=n_ep_train,
        use_knowledge=use_knowledge, seed=None,
    )
    dermnet_train_ds = DermNetEpisodicDataset(
        split="train", data_root=dermnet_root, encoder=encoder,
        n_ways=n_ways, min_shots=min_shots, max_shots=max_shots,
        num_query=num_query, n_episodes=n_ep_train,
        use_knowledge=use_knowledge, seed=None,
    )
    mixed_train_ds = MixedEpisodicDataset(
        isic_ds=isic_train_ds,
        dermnet_ds=dermnet_train_ds,
        dermnet_ratio=dermnet_ratio,
        n_episodes=n_ep_train,
    )

    # ── Val: ISIC val (2 classes) + DermNet val (4 held-out classes) ──────
    isic_val_ds = ISICEpisodicDataset(
        split="val", data_root=data_root, encoder=encoder,
        n_ways=n_ways, min_shots=min_shots, max_shots=max_shots,
        num_query=num_query, n_episodes=n_ep_val,
        use_knowledge=use_knowledge, seed=seed,
    )
    dermnet_val_ds = DermNetEpisodicDataset(
        split="val", data_root=dermnet_root, encoder=encoder,
        n_ways=n_ways, min_shots=min_shots, max_shots=max_shots,
        num_query=num_query, n_episodes=n_ep_val,
        use_knowledge=use_knowledge, seed=seed + 1,
    )
    mixed_val_ds = MixedEpisodicDataset(
        isic_ds=isic_val_ds,
        dermnet_ds=dermnet_val_ds,
        dermnet_ratio=dermnet_ratio,
        n_episodes=n_ep_val,
    )

    # ── DataLoaders ───────────────────────────────────────────────────────
    train_dl = DataLoader(
        mixed_train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
        collate_fn=collate_episodic,
    )
    val_dl = DataLoader(
        mixed_val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        collate_fn=collate_episodic,
    )

    return train_dl, val_dl



























