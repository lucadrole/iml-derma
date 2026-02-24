"""
Final test-set evaluation for INP Medical Image Classification.

Replicates the paper's Table 1 protocol (Section 5.2.2):
  - Shot settings: k = 0, 1, 3, 5, 10
  - 500 test tasks with bootstrap standard errors
  - Reports accuracy (%) for each shot setting
  - Compares INP (with knowledge) vs NP (knowledge masked) in a single run

Usage
-----
python evaluate_test.py \\
    --checkpoint saves/isic/run_0/model_best.pt \\
    --config     config_isic.toml \\
    --n-tasks    500 \\
    --n-bootstrap 1000

Outputs a table like Table 1 from the paper, plus saves results to a JSON file.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from collections import defaultdict

# Allow running from project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.inp import INP_MedClassification
from models.loss import ELBOLoss
from dataset.isic import ISICEpisodicDataset, collate_episodic, SPLIT_CLASSES


# ── Helpers ────────────────────────────────────────────────────────────────────

def bootstrap_stderr(values: list, n_bootstrap: int = 1000, seed: int = 0) -> float:
    """Bootstrap standard error of the mean."""
    rng = np.random.default_rng(seed)
    arr = np.array(values, dtype=float)
    means = [arr[rng.integers(0, len(arr), len(arr))].mean() for _ in range(n_bootstrap)]
    return float(np.std(means))


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


# ── Core evaluation ────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_checkpoint(
    model: INP_MedClassification,
    dataloader: DataLoader,
    shot_settings: list,
    n_ways: int,
    use_knowledge: bool,
    device: torch.device,
    n_bootstrap: int = 1000,
) -> dict:
    """
    Run evaluation at every shot setting over all batches in `dataloader`.

    Returns
    -------
    dict mapping shot -> {"mean": float, "stderr": float}
    """
    model.eval()

    # Accumulate per-episode accuracies for each shot setting
    per_shot_accs = defaultdict(list)

    loss_func = ELBOLoss(beta=1.0, categorical=True)
    loss_func.eval()

    for batch in dataloader:
        x_context, y_context, x_query, y_query, knowledge, _ids = batch

        x_context = x_context.to(device)
        y_context = y_context.to(device)
        x_query   = x_query.to(device)
        y_query   = y_query.to(device)

        if isinstance(knowledge, torch.Tensor):
            knowledge = knowledge.to(device)

        bs = x_context.shape[0]

        for k_shot in shot_settings:
            # Build the exact context for this shot count
            if k_shot == 0:
                x_ctx_eval = torch.zeros(bs, 1, x_context.shape[-1], device=device)
                y_ctx_eval = torch.full(
                    (bs, 1), n_ways, dtype=torch.long, device=device
                )
            else:
                x_ctx_list, y_ctx_list = [], []
                for b in range(bs):
                    parts_x, parts_y = [], []
                    for c in range(n_ways):
                        cls_mask = (y_context[b] == c).nonzero(as_tuple=True)[0]
                        take = cls_mask[: min(k_shot, len(cls_mask))]
                        if len(take) > 0:
                            parts_x.append(x_context[b][take])
                            parts_y.append(y_context[b][take])
                    if parts_x:
                        x_ctx_list.append(torch.cat(parts_x, dim=0))
                        y_ctx_list.append(torch.cat(parts_y, dim=0))
                    else:
                        x_ctx_list.append(torch.zeros(1, x_context.shape[-1], device=device))
                        y_ctx_list.append(torch.full((1,), n_ways, dtype=torch.long, device=device))

                # Pad to same length
                max_len = max(x.shape[0] for x in x_ctx_list)
                d = x_context.shape[-1]
                x_ctx_eval = torch.zeros(bs, max_len, d, device=device)
                y_ctx_eval = torch.zeros(bs, max_len, dtype=torch.long, device=device)
                for b, (xc, yc) in enumerate(zip(x_ctx_list, y_ctx_list)):
                    x_ctx_eval[b, :xc.shape[0]] = xc
                    y_ctx_eval[b, :yc.shape[0]] = yc

            # Optionally mask knowledge (for NP baseline comparison)
            k_in = knowledge if use_knowledge else None

            logits, z_samples, q_zCc, _ = model(
                x_ctx_eval, y_ctx_eval, x_query,
                y_query=None, knowledge=k_in,
            )
            # print("logits shape:", logits.shape)        # expect [n_z, bs, Q*N, 2]
            # print("logits sample:", logits[0, 0, :5])   # are all values the same?
            # print("knowledge:", knowledge[:2] if knowledge else None)  # are descriptions loading?
            # exit(0)

            # Accuracy: mean over z-samples
            mean_logits = logits.mean(dim=0)           # [bs, Q*N, N]
            preds       = mean_logits.argmax(dim=-1)   # [bs, Q*N]
            acc_per_ep  = (preds == y_query).float().mean(dim=-1)  # [bs]

            per_shot_accs[k_shot].extend(acc_per_ep.cpu().tolist())

    # Aggregate
    results = {}
    for k_shot in shot_settings:
        accs = per_shot_accs[k_shot]
        mean = float(np.mean(accs)) * 100
        stderr = bootstrap_stderr(accs, n_bootstrap=n_bootstrap) * 100
        results[k_shot] = {"mean": mean, "stderr": stderr, "n_tasks": len(accs)}

    return results


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Final test-set evaluation matching paper Table 1 protocol."
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model_best.pt (or any .pt checkpoint)")
    parser.add_argument("--config",     type=str, required=True,
                        help="Path to config TOML used during training")
    parser.add_argument("--n-tasks",    type=int, default=500,
                        help="Number of test episodes (paper uses 500)")
    parser.add_argument("--n-bootstrap", type=int, default=1000,
                        help="Bootstrap resamples for std error")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size for test DataLoader")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed",       type=int, default=42,
                        help="Seed for reproducible test episodes")
    parser.add_argument("--output-json", type=str, default="test_results.json",
                        help="Where to save results JSON")
    parser.add_argument("--also-eval-np", action="store_true",
                        help="Also evaluate with knowledge masked (NP baseline)")
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    # ── Load config ────────────────────────────────────────────────────────────
    from config import Config
    config = Config.from_toml(args.config)
    config.device = device

    # ── Load CLIP ──────────────────────────────────────────────────────────────
    import open_clip
    clip_model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai"
    )
    clip_model = clip_model.to(device)
    tokenizer  = open_clip.get_tokenizer("ViT-B-32")

    # ── Load model ─────────────────────────────────────────────────────────────
    model = INP_MedClassification(config, clip_model, tokenizer)
    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    print(f"Loaded checkpoint: {args.checkpoint}")

    # ── Build test DataLoader ──────────────────────────────────────────────────
    test_ds = ISICEpisodicDataset(
        split="test",
        data_root=getattr(config, "data_root", "data/isic2019"),
        encoder=getattr(config, "encoder_type", "clip"),
        n_ways=getattr(config, "n_ways", None),
        min_shots=10,                  # provide enough context to subsample from
        max_shots=10,
        num_query=getattr(config, "num_targets", 20),
        n_episodes=args.n_tasks,
        use_knowledge=True,            # always load descriptions; we mask in eval
        seed=args.seed,
    )
    test_dl = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_episodic,
    )

    print(f"\nTest classes: {SPLIT_CLASSES['test']}")
    print(f"N-ways: {config.n_ways}  |  Test tasks: {args.n_tasks}\n")

    shot_settings = [0, 1, 3, 5, 10]


    # ── Evaluate INP (with knowledge) ─────────────────────────────────────────
    print("=" * 60)
    print("Evaluating INP (with knowledge)...")
    print("=" * 60)
    inp_results = evaluate_checkpoint(
        model, test_dl, shot_settings,
        n_ways=config.n_ways,
        use_knowledge=True,
        device=device,
        n_bootstrap=args.n_bootstrap,
    )

    # ── Optionally evaluate NP baseline (knowledge masked) ────────────────────
    np_results = None
    if args.also_eval_np:
        print("\nEvaluating NP baseline (knowledge masked)...")
        np_results = evaluate_checkpoint(
            model, test_dl, shot_settings,
            n_ways=config.n_ways,
            use_knowledge=False,
            device=device,
            n_bootstrap=args.n_bootstrap,
        )

    # ── Print table ───────────────────────────────────────────────────────────
    print("\n")
    print("=" * 60)
    print(f"  Final Test Results  ({args.n_tasks} tasks, {config.n_ways}-way)")
    print("=" * 60)

    if np_results is not None:
        header = f"{'k':>4}  {'NP (%)':>14}  {'INP (%)':>14}"
    else:
        header = f"{'k':>4}  {'INP (%)':>14}"
    print(header)
    print("-" * len(header))

    for k in shot_settings:
        inp_m  = inp_results[k]["mean"]
        inp_se = inp_results[k]["stderr"]
        if np_results is not None:
            np_m  = np_results[k]["mean"]
            np_se = np_results[k]["stderr"]
            print(f"{k:>4}  {np_m:>8.1f} ({np_se:.1f})  {inp_m:>8.1f} ({inp_se:.1f})")
        else:
            print(f"{k:>4}  {inp_m:>8.1f} ({inp_se:.1f})")

    print("=" * 60)
    print("Format: mean (bootstrap stderr), both in %")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    output = {
        "checkpoint": args.checkpoint,
        "n_tasks":    args.n_tasks,
        "n_ways":     config.n_ways,
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









