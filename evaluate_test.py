"""
Final test-set evaluation for INP Medical Image Classification.

Replicates the paper's Table 1 protocol (Section 5.2.2):
  - Shot settings: k = 0, 1, 3, 5, 10
  - 500 test tasks with bootstrap standard errors
  - Compares two SEPARATE models:
      NP  = stage-1 checkpoint (trained without knowledge)
      INP = stage-2 checkpoint (fine-tuned with knowledge)

Usage
-----
# INP only:
python evaluate_test.py \
    --inp-checkpoint saves/isic/inp_0/model_best.pt \
    --config config_isic.toml

# NP vs INP comparison (paper Table 1):
python evaluate_test.py \
    --inp-checkpoint saves/isic/inp_0/model_best.pt \
    --np-checkpoint  saves/isic/np_0/model_best.pt \
    --config config_isic.toml
    
    # NP vs INP comparison (paper Table 1):
python evaluate_test.py \
    --inp-checkpoint /home/ldrole/my_space/work/cam_phd/informed-meta-learning/saves/INPs_isic_dermnet/inp_mixed_10/model_best.pt \
    --np-checkpoint  /home/ldrole/my_space/work/cam_phd/informed-meta-learning/saves/INPs_isic_dermnet/inp_mixed_6/model_best.pt \
    --config config_isic.toml
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.inp import INP_MedClassification
from dataset.isic import ISICEpisodicDataset, collate_episodic, SPLIT_CLASSES


# ── Helpers ────────────────────────────────────────────────────────────────────

def bootstrap_stderr(values, n_bootstrap=1000, seed=0):
    rng = np.random.default_rng(seed)
    arr = np.array(values, dtype=float)
    means = [arr[rng.integers(0, len(arr), len(arr))].mean() for _ in range(n_bootstrap)]
    return float(np.std(means))


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def load_model(config, clip_model, tokenizer, checkpoint_path, device):
    model = INP_MedClassification(config, clip_model, tokenizer)
    state_dict = torch.load(checkpoint_path, map_location=device)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  Missing keys: {sorted(missing)}")
    if unexpected:
        print(f"  Unexpected keys: {sorted(unexpected)}")
    model.to(device)
    model.eval()
    return model


# ── Build context for a specific shot count ────────────────────────────────────

def build_context_for_k(x_context, y_context, k_shot, n_ways, device):
    """
    Sub-sample exactly k_shot per class from the provided context.
    Detects padding by checking for all-zero embedding rows.

    Returns (x_ctx_eval, y_ctx_eval) on `device`.
    """
    bs = x_context.shape[0]
    d = x_context.shape[-1]

    if k_shot == 0:
        x_ctx_eval = torch.zeros(bs, 1, d, device=device)
        y_ctx_eval = torch.full((bs, 1), -1, dtype=torch.long, device=device)
        return x_ctx_eval, y_ctx_eval

    x_ctx_list, y_ctx_list = [], []
    for b in range(bs):
        parts_x, parts_y = [], []
        for c in range(n_ways):
            cls_mask = (y_context[b] == c).nonzero(as_tuple=True)[0]
            # Filter out zero-padded positions
            valid = cls_mask[x_context[b, cls_mask].abs().sum(-1) > 0]
            n_avail = len(valid)
            k = min(k_shot, n_avail)
            if k > 0:
                perm = valid[torch.randperm(n_avail, device=device)[:k]]
                parts_x.append(x_context[b, perm])
                parts_y.append(y_context[b, perm])
        if parts_x:
            x_ctx_list.append(torch.cat(parts_x))
            y_ctx_list.append(torch.cat(parts_y))
        else:
            x_ctx_list.append(torch.zeros(1, d, device=device))
            y_ctx_list.append(torch.full((1,), -1, dtype=torch.long, device=device))

    max_len = max(xc.shape[0] for xc in x_ctx_list)
    x_ctx_eval = torch.zeros(bs, max_len, d, device=device)
    y_ctx_eval = torch.full((bs, max_len), -1, dtype=torch.long, device=device)
    for b in range(bs):
        n = x_ctx_list[b].shape[0]
        x_ctx_eval[b, :n] = x_ctx_list[b]
        y_ctx_eval[b, :n] = y_ctx_list[b]

    return x_ctx_eval, y_ctx_eval


# ── Core evaluation ────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_checkpoint(model, dataloader, shot_settings, n_ways,
                        use_knowledge, device, n_bootstrap=1000):
    model.eval()
    per_shot_accs = defaultdict(list)

    for batch in dataloader:
        # Compatible with original 6-tuple collate_episodic
        x_context, y_context, x_query, y_query, knowledge, _ids = batch

        x_context = x_context.to(device)
        y_context = y_context.to(device)
        x_query = x_query.to(device)
        y_query = y_query.to(device)

        if isinstance(knowledge, torch.Tensor):
            knowledge = knowledge.to(device)

        k_in = knowledge if use_knowledge else None

        for k_shot in shot_settings:
            x_ctx_eval, y_ctx_eval = build_context_for_k(
                x_context, y_context, k_shot, n_ways, device,
            )

            logits, z_samples, q_zCc, _ = model(
                x_ctx_eval, y_ctx_eval, x_query,
                y_query=None, knowledge=k_in,
            )

            mean_logits = logits.mean(dim=0)
            preds = mean_logits.argmax(dim=-1)
            acc_per_ep = (preds == y_query).float().mean(dim=-1)
            per_shot_accs[k_shot].extend(acc_per_ep.cpu().tolist())

    results = {}
    for k_shot in shot_settings:
        accs = per_shot_accs[k_shot]
        mean = float(np.mean(accs)) * 100
        stderr = bootstrap_stderr(accs, n_bootstrap=n_bootstrap) * 100
        results[k_shot] = {"mean": mean, "stderr": stderr, "n_tasks": len(accs)}

    return results


# ── Pretty-print ───────────────────────────────────────────────────────────────

def print_table(shot_settings, inp_results, np_results=None, n_tasks=500, n_ways=2):
    print("\n" + "=" * 60)
    print(f"  Final Test Results  ({n_tasks} tasks, {n_ways}-way)")
    print("=" * 60)

    if np_results is not None:
        header = f"{'k':>4}  {'NP (%)':>14}  {'INP (%)':>14}  {'Δ':>8}"
    else:
        header = f"{'k':>4}  {'INP (%)':>14}"
    print(header)
    print("-" * len(header))

    for k in shot_settings:
        inp_m = inp_results[k]["mean"]
        inp_se = inp_results[k]["stderr"]
        if np_results is not None:
            np_m = np_results[k]["mean"]
            np_se = np_results[k]["stderr"]
            delta = inp_m - np_m
            print(f"{k:>4}  {np_m:>8.1f} ({np_se:.1f})  {inp_m:>8.1f} ({inp_se:.1f})  {delta:>+7.1f}")
        else:
            print(f"{k:>4}  {inp_m:>8.1f} ({inp_se:.1f})")

    print("=" * 60)
    print("Format: mean (bootstrap stderr), both in %")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Final test-set evaluation matching paper Table 1 protocol."
    )
    parser.add_argument("--inp-checkpoint", type=str, required=True,
                        help="Path to INP (stage-2) model_best.pt")
    parser.add_argument("--np-checkpoint", type=str, default=None,
                        help="Path to NP (stage-1) model_best.pt")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--n-tasks", type=int, default=500)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-json", type=str, default="test_results.json")
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    from config import Config
    config = Config.from_toml(args.config)
    config.device = device

    import open_clip
    clip_model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    clip_model = clip_model.to(device)
    tokenizer = open_clip.get_tokenizer("ViT-B-32")

    # ── Load INP model (stage 2) ──────────────────────────────────────────────
    print(f"\nLoading INP checkpoint: {args.inp_checkpoint}")
    inp_model = load_model(config, clip_model, tokenizer, args.inp_checkpoint, device)

    # ── Load NP model (stage 1) — separate instance ──────────────────────────
    np_model = None
    if args.np_checkpoint is not None:
        print(f"Loading NP checkpoint:  {args.np_checkpoint}")
        np_model = load_model(config, clip_model, tokenizer, args.np_checkpoint, device)

    # ── Test DataLoader (original 6-tuple collate) ────────────────────────────
    test_ds = ISICEpisodicDataset(
        split="test",
        data_root=getattr(config, "data_root", "data/isic2019"),
        encoder=getattr(config, "encoder_type", "clip"),
        n_ways=getattr(config, "n_ways", None),
        min_shots=10,
        max_shots=10,
        num_query=getattr(config, "num_targets", 20),
        n_episodes=args.n_tasks,
        use_knowledge=True,
        seed=args.seed,
    )
    test_dl = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        collate_fn=collate_episodic,
    )

    print(f"\nTest classes: {SPLIT_CLASSES['test']}")
    print(f"N-ways: {config.n_ways}  |  Test tasks: {args.n_tasks}")

    shot_settings = [0, 1, 3, 5, 10]

    # ── Evaluate INP ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Evaluating INP (stage-2, with knowledge)...")
    print("=" * 60)
    inp_results = evaluate_checkpoint(
        inp_model, test_dl, shot_settings,
        n_ways=config.n_ways, use_knowledge=True,
        device=device, n_bootstrap=args.n_bootstrap,
    )

    # ── Evaluate NP ───────────────────────────────────────────────────────────
    np_results = None
    if np_model is not None:
        print("\n" + "=" * 60)
        print("Evaluating NP (stage-1, no knowledge)...")
        print("=" * 60)
        np_results = evaluate_checkpoint(
            np_model, test_dl, shot_settings,
            n_ways=config.n_ways, use_knowledge=False,
            device=device, n_bootstrap=args.n_bootstrap,
        )

    # ── Results ───────────────────────────────────────────────────────────────
    print_table(shot_settings, inp_results, np_results,
                n_tasks=args.n_tasks, n_ways=config.n_ways)

    output = {
        "inp_checkpoint": args.inp_checkpoint,
        "np_checkpoint": args.np_checkpoint,
        "n_tasks": args.n_tasks, "n_ways": config.n_ways,
        "test_classes": SPLIT_CLASSES["test"],
        "shot_settings": shot_settings,
        "INP": {str(k): v for k, v in inp_results.items()},
    }
    if np_results is not None:
        output["NP"] = {str(k): v for k, v in np_results.items()}

    with open(args.output_json, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {args.output_json}")


if __name__ == "__main__":
    main()





























