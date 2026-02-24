"""
Trainer for INP image classification (N-way, k-shot episodic training).

Supports knowledge setups:
  B – pre-computed per-class averaged CLIP text embeddings  (N × 512 tensor)
  C – raw text descriptions per class → CLIP text encoder   (list of N strings)

Paper reference: Section 5.2.2 & Appendix A.6.3
  • lr = 1e-4, batch_size = 32, knowledge mask rate = 0.5
  • k (shots per class) ~ Uniform(0, 10), target = 20 images per class
  • For C: two-stage training (NP first, then fine-tune with knowledge + CLIP text encoder)
    -- set `load_path` to a trained NP checkpoint for stage 2
  • For B: end-to-end training
"""

import torch
import wandb
import numpy as np
import os
import sys
import toml
import optuna
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir))))
from models.inp import INP_MedClassification
from models.loss import ELBOLoss

EVAL_ITER = 500
SAVE_ITER = 500
MAX_EVAL_IT = 50


class ClassificationTrainer:
    """
    Episodic trainer for N-way, k-shot image classification with INPs.

    Expected batch format from the dataloader:
        (x_context, y_context, x_query, y_query, knowledge, task_ids)

    Where:
        x_context : [bs, k*N, d]       pre-computed CLIP image embeddings (context)
        y_context : [bs, k*N]           class labels 0..N-1
        x_query   : [bs, num_query, d]  CLIP image embeddings (target/query)
        y_query   : [bs, num_query]     class labels 0..N-1
        knowledge : depends on setup:
            C – list of N strings (class descriptions), same for all items in batch
            B – [bs, N, knowledge_dim] tensor of pre-computed CLIP text embeddings
            None – no knowledge (plain NP)
        task_ids  : any metadata (ignored by trainer)
    """

    def __init__(self, config, clip_model, tokenizer, save_dir,
                 load_path=None, last_save_it=0):
        self.config = config
        self.last_save_it = last_save_it
        self.device = config.device
        self.save_dir = save_dir

        # ------------------------------------------------------------------
        # Model
        # ------------------------------------------------------------------
        self.model = INP_MedClassification(config, clip_model, tokenizer)
        self.model.to(self.device)

        # ------------------------------------------------------------------
        # Load pretrained weights (stage-2 for setup C, or resume)
        # ------------------------------------------------------------------
        if load_path is not None:
            print(f"Loading model from state dict: {load_path}")
            state_dict = torch.load(load_path, map_location=self.device)
            missing, unexpected = self.model.load_state_dict(
                state_dict, strict=False
            )
            if missing:
                print("  Missing keys (will be randomly initialised):")
                for k in sorted(missing):
                    print(f"    {k}")
            if unexpected:
                print("  Unexpected keys (ignored):")
                for k in sorted(unexpected):
                    print(f"    {k}")

        # Optimiser
        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=config.lr,
        )

        # Loss – categorical ELBO
        self.loss_func = ELBOLoss(
            beta=getattr(config, "beta", 1.0),
            categorical=True,
        )

        # Dataloaders tTODO – user must supply via setup_dataloaders or similar
        self.train_dataloader = None
        self.val_dataloader = None

        # Training hyper-params
        self.num_epochs = config.num_epochs
        self.knowledge_mask_rate = getattr(config, "knowledge_mask_rate", 0.5)
        self.knowledge_setup = getattr(config, "knowledge_setup", "C")
        self.use_knowledge = getattr(config, "use_knowledge", True)

        # Print trainable parameters
        n_trainable = 0
        print("\nTrainable parameters:")
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                print(f"  {name}  {tuple(param.shape)}")
                n_trainable += param.numel()
        print(f"Total trainable: {n_trainable:,}\n")

    # ------------------------------------------------------------------
    # Attach dataloaders after construction (flexible for different setups)
    # ------------------------------------------------------------------
    def set_dataloaders(self, train_dl, val_dl):
        self.train_dataloader = train_dl
        self.val_dataloader = val_dl

    # ------------------------------------------------------------------
    # Knowledge masking
    # ------------------------------------------------------------------
    def _maybe_mask_knowledge(self, knowledge):
        """
        With probability `knowledge_mask_rate`, replace knowledge with None
        (equivalent to setting k = 0 in the paper).
        """
        if not self.use_knowledge:
            return None
        if knowledge is None:
            return None
        if self.training_phase and np.random.rand() < self.knowledge_mask_rate:
            return None
        
        return knowledge

    # ------------------------------------------------------------------
    # Single-batch forward + loss (training)
    # ------------------------------------------------------------------
    def run_batch_train(self, batch):
        """
        Run one training batch.

        batch: (x_context, y_context, x_query, y_query, knowledge, task_ids)
        """
        x_context, y_context, x_query, y_query, knowledge, _ids = batch

        x_context = x_context.to(self.device)
        y_context = y_context.to(self.device)
        x_query = x_query.to(self.device)
        y_query = y_query.to(self.device)

        # Move knowledge to device if it's a tensor (setup B)
        if isinstance(knowledge, torch.Tensor):
            knowledge = knowledge.to(self.device)

        knowledge = self._maybe_mask_knowledge(knowledge)

        # Forward
        logits, z_samples, q_zCc, q_zCct = self.model(
            x_context, y_context, x_query,
            y_query=y_query, knowledge=knowledge,
        )

        # Loss
        self.loss_func.train()
        loss, kl, neg_ll = self.loss_func(
            (logits, z_samples, q_zCc, q_zCct), y_query
        )

        # Accuracy (use mean logits across z samples)
        mean_logits = logits.mean(dim=0)  # [bs, num_query, N]
        preds = mean_logits.argmax(dim=-1)  # [bs, num_query]
        acc = (preds == y_query).float().mean().item()

        return {
            "loss": loss,
            "kl": kl,
            "negative_ll": neg_ll,
            "accuracy": acc,
        }

    # ------------------------------------------------------------------
    # Single-batch forward + loss (evaluation with variable num_context)
    # ------------------------------------------------------------------
    def run_batch_eval(self, batch, num_shots_per_class=5):
        """
        Evaluate with a specific number of shots per class.

        Sub-sample context from the CONTEXT portion of the batch,
        not from query. Query images are always kept intact (num_query * N)
        so accuracy is comparable across different k settings.
        """
        x_context, y_context, x_query, y_query, knowledge, _ids = batch

        x_context = x_context.to(self.device)
        y_context = y_context.to(self.device)
        x_query   = x_query.to(self.device)
        y_query   = y_query.to(self.device)

        if isinstance(knowledge, torch.Tensor):
            knowledge = knowledge.to(self.device)

        bs = x_context.shape[0]
        n_ways = self.config.n_ways

        if num_shots_per_class == 0:
            # Zero-shot: empty context (padded to length 1 for tensor shape)
            x_ctx_eval = torch.zeros(bs, 1, x_context.shape[-1], device=self.device)
            y_ctx_eval = torch.full((bs, 1), self.config.n_ways, dtype=torch.long, device=self.device)
        else:
            # Sub-sample from the context set: take up to num_shots_per_class
            # per class from the already-provided context embeddings.
            x_ctx_list, y_ctx_list = [], []
            for b in range(bs):
                parts_x, parts_y = [], []
                for c in range(n_ways):
                    cls_mask = (y_context[b] == c).nonzero(as_tuple=True)[0]
                    # Filter out zero-padding (where x is all zeros)
                    valid = cls_mask[x_context[b, cls_mask].abs().sum(-1) > 0]
                    n_avail = len(valid)
                    k = min(num_shots_per_class, n_avail)
                    if k > 0:
                        perm = valid[torch.randperm(n_avail, device=self.device)[:k]]
                        parts_x.append(x_context[b, perm])
                        parts_y.append(y_context[b, perm])
                if parts_x:
                    x_ctx_list.append(torch.cat(parts_x))
                    y_ctx_list.append(torch.cat(parts_y))
                else:
                    x_ctx_list.append(torch.zeros(1, x_context.shape[-1], device=self.device))
                    y_ctx_list.append(torch.zeros(1, dtype=torch.long, device=self.device))

            # Pad to uniform length across batch
            max_ctx = max(xc.shape[0] for xc in x_ctx_list)
            x_ctx_eval = torch.zeros(bs, max_ctx, x_context.shape[-1], device=self.device)
            y_ctx_eval = torch.zeros(bs, max_ctx, dtype=torch.long, device=self.device)
            for b in range(bs):
                n = x_ctx_list[b].shape[0]
                x_ctx_eval[b, :n] = x_ctx_list[b]
                y_ctx_eval[b, :n] = y_ctx_list[b]

        # Don't mask knowledge at eval time
        if not self.use_knowledge:
            knowledge = None

        # Forward (model in eval mode → q_zCct = None)
        logits, z_samples, q_zCc, q_zCct = self.model(
            x_ctx_eval, y_ctx_eval, x_query,
            y_query=None, knowledge=knowledge,
        )

        # Loss (eval path in ELBOLoss → importance-weighted NLL)
        self.loss_func.eval()
        loss, kl, neg_ll = self.loss_func(
            (logits, z_samples, q_zCc, q_zCct), y_query
        )

        # Accuracy
        mean_logits = logits.mean(dim=0)  # [bs, num_query, N]
        preds = mean_logits.argmax(dim=-1)
        acc = (preds == y_query).float().mean().item()

        return {
            "loss": loss.item(),
            "negative_ll": neg_ll.item(),
            "accuracy": acc,
        }

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    def train(self):
        assert self.train_dataloader is not None, "Call set_dataloaders() first"
        assert self.val_dataloader is not None, "Call set_dataloaders() first"

        it = 0
        best_eval_acc = 0.0
        min_eval_loss = np.inf

        for epoch in range(self.num_epochs + 1):
            for batch in self.train_dataloader:
                self.model.train()
                self.training_phase = True
                self.optimizer.zero_grad()

                results = self.run_batch_train(batch)
                loss = results["loss"]
                loss.backward()
                self.optimizer.step()

                wandb.log({
                    "train/loss": results["loss"].item(),
                    "train/kl": results["kl"].item(),
                    "train/neg_ll": results["negative_ll"].item(),
                    "train/accuracy": results["accuracy"],
                    "iteration": self.last_save_it + it,
                })

                # ---- Evaluation ----
                if it % EVAL_ITER == 0 and it > 0:
                    eval_results = self.evaluate()
                    wandb.log(eval_results)

                    eval_loss = eval_results.get("eval/loss", np.inf)
                    eval_acc = eval_results.get("eval/accuracy_5shot", 0.0)

                    if eval_loss < min_eval_loss and it > 1500:
                        min_eval_loss = eval_loss
                        best_eval_acc = eval_acc
                        torch.save(
                            self.model.state_dict(),
                            f"{self.save_dir}/model_best.pt",
                        )
                        torch.save(
                            self.optimizer.state_dict(),
                            f"{self.save_dir}/optim_best.pt",
                        )
                        print(
                            f"Best model saved at it={self.last_save_it + it}  "
                            f"loss={eval_loss:.4f}  acc@5={eval_acc:.2%}"
                        )

                it += 1

        return min_eval_loss

    # ------------------------------------------------------------------
    # Full evaluation
    # ------------------------------------------------------------------
    def evaluate(self):
        """
        Evaluate across multiple shot settings (0, 1, 3, 5, 10).
        Returns a flat dict suitable for wandb.log().
        """
        print("Evaluating...")
        self.model.eval()
        self.training_phase = False

        shot_settings = [0, 1, 3, 5, 10]
        metrics = defaultdict(list)  # key -> list of values across batches

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_dataloader):
                for k_shot in shot_settings:
                    res = self.run_batch_eval(batch, num_shots_per_class=k_shot)
                    metrics[f"acc_{k_shot}shot"].append(res["accuracy"])
                    metrics[f"nll_{k_shot}shot"].append(res["negative_ll"])
                    metrics[f"loss_{k_shot}shot"].append(res["loss"])

                if batch_idx >= MAX_EVAL_IT:
                    break

        # Aggregate
        results = {}
        total_loss = []
        for key, vals in metrics.items():
            mean_val = np.mean(vals)
            results[f"eval/{key}"] = mean_val
            if key.startswith("loss_"):
                total_loss.append(mean_val)

        results["eval/loss"] = np.mean(total_loss) if total_loss else 0.0

        # Convenience aliases
        for k_shot in shot_settings:
            acc_key = f"acc_{k_shot}shot"
            if acc_key in metrics:
                results[f"eval/accuracy_{k_shot}shot"] = np.mean(
                    metrics[acc_key]
                )

        print(
            "  " + "  ".join(
                f"{k}shot: {results.get(f'eval/accuracy_{k}shot', 0):.2%}"
                for k in shot_settings
            )
        )

        return results


# ======================================================================
# Entry-point helper
# ======================================================================

def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    return device


def meta_train(trial, config, clip_model, tokenizer,
               setup_dataloaders_fn, run_name_prefix="run"):
    """
    Top-level training function (compatible with Optuna).

    Args:
        trial:                  Optuna trial (unused if n_trials=1)
        config:                 Config namespace
        clip_model:             Pre-loaded CLIP model
        tokenizer:              CLIP tokenizer
        setup_dataloaders_fn:   Callable(config) -> (train_dl, val_dl)
        run_name_prefix:        Prefix for save dirs / wandb
    """
    device = get_device()
    config.device = device

    # Save directory
    save_dir = f"./saves/{config.project_name}"
    os.makedirs(save_dir, exist_ok=True)

    existing = [
        int(x.split("_")[-1])
        for x in os.listdir(save_dir)
        if x.startswith(run_name_prefix)
    ]
    save_no = max(existing) + 1 if existing else 0
    save_dir = f"{save_dir}/{run_name_prefix}_{save_no}"
    os.makedirs(save_dir, exist_ok=True)

    # Build trainer
    load_path = getattr(config, "load_path", None) or None
    trainer = ClassificationTrainer(
        config, clip_model, tokenizer, save_dir, load_path=load_path
    )

    # Dataloaders
    train_dl, val_dl = setup_dataloaders_fn(config)
    trainer.set_dataloaders(train_dl, val_dl)

    # Save config
    if hasattr(config, "write_config"):
        config.write_config(f"{save_dir}/config.toml")

    # Train
    wandb.init(
        project=config.project_name,
        name=f"{run_name_prefix}_{save_no}",
        config=vars(config) if hasattr(config, "__dict__") else config,
    )
    best_eval_loss = trainer.train()
    wandb.finish()

    return best_eval_loss

if __name__ == "__main__":
    import argparse
    from config import Config
    from dataset.isic import setup_isic_dataloaders
    import open_clip

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True,
                        help="Path to config toml file (e.g. config_isic.toml)")
    args = parser.parse_args()

    config = Config.from_toml(args.config)

    # Load CLIP model
    device = get_device()
    clip_model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai"
    )
    clip_model = clip_model.to(device)
    tokenizer = open_clip.get_tokenizer("ViT-B-32")

    meta_train(
        trial=None,
        config=config,
        clip_model=clip_model,
        tokenizer=tokenizer,
        setup_dataloaders_fn=setup_isic_dataloaders,
        run_name_prefix=getattr(config, "run_name_prefix", "run"),
    )
