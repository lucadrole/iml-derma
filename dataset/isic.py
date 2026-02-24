"""
ISIC 2019 episodic dataset for N-way, k-shot image classification with INPs.

Two-step workflow
-----------------
Step 1 – Pre-compute and cache CLIP/BiomedCLIP embeddings (run once):

    python dataset/isic.py \\
        --data-root data/isic2019 \\
        --encoder clip            # or biomedclip
        --biomedclip-dir /path/to/biomedclip   # only needed for biomedclip

    Produces:
        data/isic2019/embeddings_clip.pt       (or embeddings_biomedclip.pt)

Step 2 – Use ISICEpisodicDataset in your training script:

    from dataset.isic import ISICEpisodicDataset, collate_episodic, setup_isic_dataloaders
    train_dl, val_dl = setup_isic_dataloaders(config)

Data layout (Kaggle download)
------------------------------
    data/isic2019/
    ├── ISIC_2019_Training_GroundTruth.csv
    ├── ISIC_2019_Training_Metadata.csv
    └── ISIC_2019_Training_Input/
        └── ISIC_2019_Training_Input/      <- images are one level deeper
            ├── ISIC_0000000.jpg
            └── ...

Class split (design doc §3, 4/2/2 over 8 labelled classes)
------------------------------------------------------------
    Meta-train : MEL, NV, BCC, AK          (4 classes)
    Meta-val   : BKL, DF                   (2 classes)
    Meta-test  : VASC, SCC                 (2 classes)
    UNK exists in CSV but has zero training images -> always skipped.

Knowledge (Setup C)
-------------------
    One natural-language description per class, loaded from
    data/isic2019/descriptions.json.
    Knowledge masking is handled by ClassificationTrainer, not here.
"""

import os
import json
import argparse

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# ---------------------------------------------------------------------------
# Class metadata
# ---------------------------------------------------------------------------

# Full class name -> CSV column code
CLASS_CODES = {
    "Melanoma":                "MEL",
    "Melanocytic Nevus":       "NV",
    "Basal Cell Carcinoma":    "BCC",
    "Actinic Keratosis":       "AK",
    "Benign Keratosis":        "BKL",
    "Dermatofibroma":          "DF",
    "Vascular Lesion":         "VASC",
    "Squamous Cell Carcinoma": "SCC",
}

# Canonical order -- position = global class index stored in embedding cache
ALL_CLASSES = list(CLASS_CODES.keys())   # 8 entries

# CSV column order for the 8 labelled classes (UNK handled separately)
GT_COLUMNS = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]

# MODALITY = 1
# # Meta-learning split assignment
# if MODALITY == 0:
#     SPLIT_CLASSES = {
#         "train": ["Melanoma", "Melanocytic Nevus", "Basal Cell Carcinoma", "Actinic Keratosis"],
#         "val":   ["Benign Keratosis", "Dermatofibroma"],
#         "test":  ["Vascular Lesion", "Squamous Cell Carcinoma"],
#     }
# elif MODALITY == 1:
#      SPLIT_CLASSES = {
#         "train": ["Melanoma", "Melanocytic Nevus", "Basal Cell Carcinoma", "Actinic Keratosis"],
#         "val":   ["Vascular Lesion", "Squamous Cell Carcinoma"],
#         "test":  ["Benign Keratosis", "Dermatofibroma"],
#     }
     
# SPLIT_CLASSES_ORIGINAL = {
#     "train": ["Melanoma", "Melanocytic Nevus", "Basal Cell Carcinoma", "Actinic Keratosis"],
#     "val":   ["Benign Keratosis", "Dermatofibroma"],
#     "test":  ["Vascular Lesion", "Squamous Cell Carcinoma"],
#     "test":  ["Benign Keratosis", "Dermatofibroma"],
# }
SPLIT_CLASSES = {
    "train": ["Melanoma", "Melanocytic Nevus", "Basal Cell Carcinoma", "Actinic Keratosis"],
    "val":   ["Benign Keratosis", "Dermatofibroma"],
    "test":  ["Vascular Lesion", "Squamous Cell Carcinoma"],
}

# Kaggle download nests images one extra level deep
_IMG_SUBDIR = os.path.join("ISIC_2019_Training_Input", "ISIC_2019_Training_Input")


def _img_dir(data_root: str) -> str:
    return os.path.join(data_root, _IMG_SUBDIR)


# ---------------------------------------------------------------------------
# Step 1 -- Embedding pre-computation
# ---------------------------------------------------------------------------

def build_embedding_cache(
    data_root: str,
    encoder: str = "clip",
    biomedclip_dir: str = None,
    batch_size: int = 128,
) -> None:
    """
    Encode all ISIC 2019 training images with a frozen vision encoder and
    save the result as a .pt cache file.

    Cache schema
    ------------
    {
        "embeddings": Tensor[N, embed_dim],   float32, L2-normalised
        "image_ids":  list[str],              e.g. "ISIC_0000000"
        "labels":     Tensor[N],              int64, index into ALL_CLASSES
        "encoder":    str,
        "embed_dim":  int,
    }

    UNK images are skipped (UNK column == 1.0 in the CSV).
    """
    import pandas as pd

    # ---- Ground truth ----
    gt_path = os.path.join(data_root, "ISIC_2019_Training_GroundTruth.csv")
    if not os.path.exists(gt_path):
        raise FileNotFoundError(
            f"Not found: {gt_path}\n"
            "Expected: data/isic2019/ISIC_2019_Training_GroundTruth.csv"
        )
    gt = pd.read_csv(gt_path)

    code_to_global_idx = {
        code: ALL_CLASSES.index(name)
        for name, code in CLASS_CODES.items()
    }

    image_ids, labels = [], []
    skipped_unk = 0

    for _, row in gt.iterrows():
        # Skip UNK rows
        if "UNK" in gt.columns and float(row.get("UNK", 0.0)) == 1.0:
            skipped_unk += 1
            continue
        # Values are floats (1.0 / 0.0) -- argmax over the 8 labelled columns
        vals = row[GT_COLUMNS].astype(float).values
        code = GT_COLUMNS[int(vals.argmax())]
        labels.append(code_to_global_idx[code])
        image_ids.append(str(row["image"]))

    print(f"Images to encode : {len(image_ids)}")
    print(f"UNK rows skipped : {skipped_unk}")
    labels_tensor = torch.tensor(labels, dtype=torch.long)

    # ---- Load encoder ----
    model, preprocess = _load_vision_encoder(encoder, biomedclip_dir)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Encoding with {encoder!r} on {device}")

    # ---- Encode in batches ----
    img_dir = _img_dir(data_root)
    all_embeds = []
    n = len(image_ids)

    for start in range(0, n, batch_size):
        batch_ids = image_ids[start: start + batch_size]
        imgs = []
        for img_id in batch_ids:
            path = os.path.join(img_dir, f"{img_id}.jpg")
            imgs.append(preprocess(Image.open(path).convert("RGB")))

        imgs_t = torch.stack(imgs).to(device)
        with torch.no_grad():
            embeds = _encode_images(model, imgs_t)   # [B, d]
        all_embeds.append(embeds.cpu())

        done = start + len(batch_ids)
        if done % 2000 < batch_size or done == n:
            print(f"  {done}/{n}")

    embeddings = torch.cat(all_embeds, dim=0)   # [N, d]
    embed_dim  = int(embeddings.shape[1])

    # ---- Save ----
    out_path = os.path.join(data_root, f"embeddings_{encoder}.pt")
    torch.save({
        "embeddings": embeddings,
        "image_ids":  image_ids,
        "labels":     labels_tensor,
        "encoder":    encoder,
        "embed_dim":  embed_dim,
    }, out_path)

    print(f"\nSaved -> {out_path}")
    print(f"  Shape    : {list(embeddings.shape)}")
    print(f"  embed_dim: {embed_dim}")
    for local_idx, name in enumerate(ALL_CLASSES):
        count = int((labels_tensor == local_idx).sum())
        print(f"  {CLASS_CODES[name]:4s}  {name:30s}  {count:5d} images")


def _load_vision_encoder(encoder: str, biomedclip_dir: str = None):
    """Return (model, preprocess_fn) for the requested encoder."""
    try:
        import open_clip
    except ImportError:
        raise ImportError("Run: pip install open-clip-torch")

    if encoder == "clip":
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai"
        )
        return model, preprocess

    elif encoder == "biomedclip":
        if biomedclip_dir is None:
            raise ValueError("--biomedclip-dir is required for biomedclip encoder")

        from open_clip.factory import HF_HUB_PREFIX, _MODEL_CONFIGS

        with open(os.path.join(biomedclip_dir, "open_clip_config.json")) as f:
            clip_cfg = json.load(f)

        model_name = "biomedclip_local"
        if model_name not in _MODEL_CONFIGS:
            _MODEL_CONFIGS[model_name] = clip_cfg["model_cfg"]

        preprocess_cfg = clip_cfg["preprocess_cfg"]
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name=model_name,
            pretrained=os.path.join(biomedclip_dir, "open_clip_pytorch_model.bin"),
            **{f"image_{k}": v for k, v in preprocess_cfg.items()},
        )
        return model, preprocess

    else:
        raise ValueError(f"Unknown encoder {encoder!r}. Choose 'clip' or 'biomedclip'.")


def _encode_images(model, images: torch.Tensor) -> torch.Tensor:
    """Return L2-normalised image embeddings [B, d]."""
    embeds = model.encode_image(images)
    embeds = embeds / embeds.norm(dim=-1, keepdim=True)
    return embeds.float()


# ---------------------------------------------------------------------------
# Step 2 -- Episodic dataset
# ---------------------------------------------------------------------------

class ISICEpisodicDataset(Dataset):
    """
    N-way, k-shot episodic dataset over cached CLIP/BiomedCLIP embeddings.

    Each __getitem__ samples one episode:
        x_context : Tensor[k*N, d]           context image embeddings
        y_context : Tensor[k*N]              episode-local labels 0..N-1
        x_query   : Tensor[num_query*N, d]   query image embeddings
        y_query   : Tensor[num_query*N]      episode-local labels 0..N-1
        knowledge : list[str] of length N    class descriptions, or None
        task_id   : int

    Args:
        split         : "train" | "val" | "test"
        data_root     : folder with embeddings_{encoder}.pt and descriptions.json
        encoder       : "clip" or "biomedclip"
        n_ways        : classes per episode (default: all available in split)
        min_shots     : min context shots per class (default 0)
        max_shots     : max context shots per class (default 10, per paper)
        num_query     : query images per class per episode (default 20, per paper)
        n_episodes    : virtual dataset length
        use_knowledge : if True load descriptions.json; else knowledge=None
        seed          : fixed seed for reproducible val/test episodes (None=random)
    """

    def __init__(
        self,
        split: str = "train",
        data_root: str = "data/isic2019",
        encoder: str = "clip",
        n_ways: int = None,
        min_shots: int = 0,
        max_shots: int = 10,
        num_query: int = 20,
        n_episodes: int = 10_000,
        use_knowledge: bool = True,
        seed: int = None,
    ):
        assert split in ("train", "val", "test"), f"Bad split: {split!r}"

        self.split         = split
        self.min_shots     = min_shots
        self.max_shots     = max_shots
        self.num_query     = num_query
        self.n_episodes    = n_episodes
        self.use_knowledge = use_knowledge
        self.seed          = seed

        # ---- Load embedding cache ----
        cache_path = os.path.join(data_root, f"embeddings_{encoder}.pt")
        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                f"Cache not found: {cache_path}\n"
                f"Run first:  python dataset/isic.py "
                f"--data-root {data_root} --encoder {encoder}"
            )
        cache = torch.load(cache_path, map_location="cpu")
        all_embeddings = cache["embeddings"]   # [N_total, d]
        all_labels     = cache["labels"]       # [N_total]
        self.embed_dim  = int(cache["embed_dim"])

        # ---- Per-class embedding pools ----
        self.split_class_names = SPLIT_CLASSES[split]
        self.class_embeddings  = {}   # local_idx -> Tensor[n_imgs, d]

        for local_idx, name in enumerate(self.split_class_names):
            global_idx = ALL_CLASSES.index(name)
            mask = (all_labels == global_idx)
            self.class_embeddings[local_idx] = all_embeddings[mask]

        # ---- n_ways ----
        n_avail = len(self.split_class_names)
        if n_ways is None:
            self.n_ways = n_avail
        else:
            assert n_ways <= n_avail, (
                f"n_ways={n_ways} > available classes ({n_avail}) "
                f"for split '{split}'"
            )
            self.n_ways = n_ways

        # ---- Knowledge descriptions ----
        self.descriptions = None
        if use_knowledge:
            desc_path = os.path.join(data_root, "descriptions.json")
            if not os.path.exists(desc_path):
                raise FileNotFoundError(
                    f"Descriptions not found: {desc_path}\n"
                    "Generate them first (see dataset/generate_descriptions.py)."
                )
            with open(desc_path) as f:
                desc_dict = json.load(f)

            missing = [n for n in self.split_class_names if n not in desc_dict]
            if missing:
                raise ValueError(f"Missing descriptions for: {missing}")

            self.descriptions = {
                local_idx: desc_dict[name]
                for local_idx, name in enumerate(self.split_class_names)
            }

        # ---- Summary ----
        print(f"ISICEpisodicDataset [{split}]  n_ways={self.n_ways}  "
              f"shots=[{min_shots},{max_shots}]  query/class={num_query}  "
              f"episodes={n_episodes}")
        for li, name in enumerate(self.split_class_names):
            n = len(self.class_embeddings[li])
            print(f"  [{li}] {CLASS_CODES[name]:4s}  {name:30s}  {n} images")

    def __len__(self):
        return self.n_episodes

    def __getitem__(self, idx: int):
        # Per-episode RNG: fixed for val/test, random for train
        rng = np.random.default_rng(
            self.seed * 100_000 + idx if self.seed is not None else None
        )

        # Sample which N classes appear in this episode
        chosen_local = sorted(
            rng.choice(
                len(self.split_class_names), self.n_ways, replace=False
            ).tolist()
        )
        # Map chosen class local-idx -> episode label 0..N-1
        ep_label = {cls: ep for ep, cls in enumerate(chosen_local)}

        # Sample k shots uniformly
        k = int(rng.integers(self.min_shots, self.max_shots + 1))

        x_ctx_parts, y_ctx_parts = [], []
        x_qry_parts, y_qry_parts = [], []

        for cls in chosen_local:
            ep_lbl = ep_label[cls]
            pool   = self.class_embeddings[cls]   # [n_img, d]
            n_img  = len(pool)
            need   = k + self.num_query

            if need > n_img:
                indices = rng.choice(n_img, need, replace=True)
            else:
                indices = rng.choice(n_img, need, replace=False)

            if k > 0:
                x_ctx_parts.append(pool[indices[:k]])
                y_ctx_parts.append(
                    torch.full((k,), ep_lbl, dtype=torch.long)
                )

            x_qry_parts.append(pool[indices[k: k + self.num_query]])
            y_qry_parts.append(
                torch.full((self.num_query,), ep_lbl, dtype=torch.long)
            )

        if k > 0:
            x_context = torch.cat(x_ctx_parts, dim=0)   # [k*N, d]
            y_context = torch.cat(y_ctx_parts, dim=0)   # [k*N]
        else:
            x_context = torch.zeros(0, self.embed_dim)
            y_context = torch.zeros(0, dtype=torch.long)

        x_query = torch.cat(x_qry_parts, dim=0)   # [num_query*N, d]
        y_query = torch.cat(y_qry_parts, dim=0)   # [num_query*N]

        if self.use_knowledge and self.descriptions is not None:
            knowledge = [self.descriptions[cls] for cls in chosen_local]
        else:
            knowledge = None

        return x_context, y_context, x_query, y_query, knowledge, idx


# ---------------------------------------------------------------------------
# Collate function
# ---------------------------------------------------------------------------

def collate_episodic(batch):
    """
    Collate a list of episodes into batched tensors.

    Variable-length context (k varies per episode) is zero-padded to the
    maximum k*N in the batch. encode_context_per_class() in the model divides
    by the true per-class count so padded zeros don't bias the representation.

    Returns
    -------
    x_context : Tensor[bs, max_ctx, d]
    y_context : Tensor[bs, max_ctx]         zero-padded
    x_query   : Tensor[bs, num_query*N, d]
    y_query   : Tensor[bs, num_query*N]
    knowledge : list[str] of length N  (shared across batch), or None
    task_ids  : Tensor[bs]
    """
    x_ctx_list, y_ctx_list = [], []
    x_qry_list, y_qry_list = [], []
    knowledges, task_ids   = [], []

    for x_ctx, y_ctx, x_qry, y_qry, knowledge, tid in batch:
        x_ctx_list.append(x_ctx)
        y_ctx_list.append(y_ctx)
        x_qry_list.append(x_qry)
        y_qry_list.append(y_qry)
        knowledges.append(knowledge)
        task_ids.append(tid)

    bs = len(batch)
    d  = x_ctx_list[0].shape[-1]

    max_ctx = max(x.shape[0] for x in x_ctx_list)
    pad_ctx = max(max_ctx, 1)   # avoid 0-dim tensors for zero-shot batches

    x_context_padded = torch.zeros(bs, pad_ctx, d)
    y_context_padded = torch.zeros(bs, pad_ctx, dtype=torch.long)

    for i, (xc, yc) in enumerate(zip(x_ctx_list, y_ctx_list)):
        if xc.shape[0] > 0:
            x_context_padded[i, :xc.shape[0]] = xc
            y_context_padded[i, :yc.shape[0]] = yc

    x_query = torch.stack(x_qry_list)   # [bs, num_query*N, d]
    y_query = torch.stack(y_qry_list)   # [bs, num_query*N]

    # All episodes in a batch share the same split class pool and descriptions
    if any(k is None for k in knowledges):
        knowledge_out = None
    else:
        knowledge_out = knowledges   # list[list[str]], shape [bs][N]


    return (
        x_context_padded,
        y_context_padded,
        x_query,
        y_query,
        knowledge_out,
        torch.tensor(task_ids),
    )


# ---------------------------------------------------------------------------
# Convenience: build dataloaders for ClassificationTrainer
# ---------------------------------------------------------------------------

def setup_isic_dataloaders(config):
    """
    Build train and val DataLoaders ready for ClassificationTrainer.

    Reads from config (all have sensible defaults):
        config.data_root          default "data/isic2019"
        config.encoder_type       "clip" or "biomedclip"  (default "clip")
        config.n_ways             default None (max per split)
        config.min_num_context    default 0
        config.max_num_context    default 10
        config.num_targets        query images per class  (default 20)
        config.n_episodes_train   default 10_000
        config.n_episodes_val     default 1_000
        config.batch_size         default 32
        config.use_knowledge      default True
        config.seed               for reproducible val episodes
        config.num_workers        default 4

    Returns: (train_dl, val_dl)
    """
    data_root     = getattr(config, "data_root",        "data/isic2019")
    encoder       = getattr(config, "encoder_type",     "clip")
    n_ways        = getattr(config, "n_ways",           None)
    min_shots     = getattr(config, "min_num_context",  0)
    max_shots     = getattr(config, "max_num_context",  10)
    num_query     = getattr(config, "num_targets",      20)
    n_ep_train    = getattr(config, "n_episodes_train", 10_000)
    n_ep_val      = getattr(config, "n_episodes_val",   1_000)
    batch_size    = getattr(config, "batch_size",       32)
    use_knowledge = getattr(config, "use_knowledge",    True)
    seed          = getattr(config, "seed",             42)
    num_workers   = getattr(config, "num_workers",      4)

    train_ds = ISICEpisodicDataset(
        split="train", data_root=data_root, encoder=encoder,
        n_ways=n_ways, min_shots=min_shots, max_shots=max_shots,
        num_query=num_query, n_episodes=n_ep_train,
        use_knowledge=use_knowledge, seed=None,
    )
    val_ds = ISICEpisodicDataset(
        split="val", data_root=data_root, encoder=encoder,
        n_ways=n_ways, min_shots=min_shots, max_shots=max_shots,
        num_query=num_query, n_episodes=n_ep_val,
        use_knowledge=use_knowledge, seed=seed,
    )

    train_dl = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
        collate_fn=collate_episodic,
    )
    val_dl = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        collate_fn=collate_episodic,
    )
    return train_dl, val_dl


# ---------------------------------------------------------------------------
# CLI -- Step 1
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pre-compute CLIP/BiomedCLIP embeddings for ISIC 2019."
    )
    parser.add_argument(
        "--data-root",      type=str, default="data/isic2019",
        help="Folder with ISIC_2019_Training_Input/ and GroundTruth.csv",
    )
    parser.add_argument(
        "--encoder",        type=str, default="clip",
        choices=["clip", "biomedclip"],
    )
    parser.add_argument(
        "--biomedclip-dir", type=str, default=None,
        help="Local BiomedCLIP checkpoint dir (required for biomedclip)",
    )
    parser.add_argument(
        "--batch-size",     type=int, default=128,
        help="Image batch size (increase if VRAM allows)",
    )
    args = parser.parse_args()

    build_embedding_cache(
        data_root=args.data_root,
        encoder=args.encoder,
        biomedclip_dir=args.biomedclip_dir,
        batch_size=args.batch_size,
    )
