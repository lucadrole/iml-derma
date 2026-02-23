"""
Smoke test for the ISIC 2019 INP classification pipeline.

Tests the full stack in order:
  1. ISICEpisodicDataset + collate_episodic
  2. INP_MedClassification forward pass
  3. ELBOLoss (categorical) forward pass
  4. Backward pass (gradients flow)
  5. ClassificationTrainer.run_batch_train
  6. ClassificationTrainer.run_batch_eval

Run from repo root:
    python smoke_test_isic.py

Expected output: all checks print OK and no exceptions are raised.
Takes ~30 seconds on CPU, ~10 seconds on GPU.
"""

import sys
import os
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Config ────────────────────────────────────────────────────────────────────

DATA_ROOT    = "data/isic2019"
ENCODER_TYPE = "clip"
N_WAYS       = 2
BATCH_SIZE   = 4
HIDDEN_DIM   = 512
CLIP_DIM     = 512

# ── Helpers ───────────────────────────────────────────────────────────────────

def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def ok(msg):
    print(f"  ✓  {msg}")

def check_shape(tensor, expected, name):
    assert tensor.shape == torch.Size(expected), \
        f"{name}: expected {expected}, got {list(tensor.shape)}"
    ok(f"{name}: {list(tensor.shape)}")

# ── 1. Dataset & DataLoader ───────────────────────────────────────────────────

section("1. ISICEpisodicDataset + collate_episodic")

from dataset.isic import ISICEpisodicDataset, collate_episodic
from torch.utils.data import DataLoader

train_ds = ISICEpisodicDataset(
    split="train",
    data_root=DATA_ROOT,
    encoder=ENCODER_TYPE,
    n_ways=N_WAYS,
    min_shots=0,
    max_shots=10,
    num_query=20,
    n_episodes=50,
    use_knowledge=True,
    seed=None,
)
val_ds = ISICEpisodicDataset(
    split="val",
    data_root=DATA_ROOT,
    encoder=ENCODER_TYPE,
    n_ways=N_WAYS,
    min_shots=0,
    max_shots=10,
    num_query=20,
    n_episodes=20,
    use_knowledge=True,
    seed=42,
)

train_dl = DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=0, collate_fn=collate_episodic,
)
val_dl = DataLoader(
    val_ds, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=0, collate_fn=collate_episodic,
)

# Grab one batch
batch = next(iter(train_dl))
x_ctx, y_ctx, x_qry, y_qry, knowledge, task_ids = batch

ok(f"x_context : {list(x_ctx.shape)}  (bs, max_k*N, d)")
ok(f"y_context : {list(y_ctx.shape)}")
ok(f"x_query   : {list(x_qry.shape)}  (bs, 20*N, d)")
ok(f"y_query   : {list(y_qry.shape)}")
ok(f"knowledge : {knowledge}")
ok(f"task_ids  : {task_ids.tolist()}")

assert x_ctx.shape[0] == BATCH_SIZE,      "batch size mismatch"
assert x_qry.shape == (BATCH_SIZE, 20 * N_WAYS, CLIP_DIM), \
    f"query shape wrong: {x_qry.shape}"
assert y_qry.shape == (BATCH_SIZE, 20 * N_WAYS), \
    f"y_query shape wrong: {y_qry.shape}"
assert isinstance(knowledge, list) and len(knowledge) == N_WAYS, \
    f"knowledge should be list of {N_WAYS} strings, got {knowledge}"
assert y_ctx.max().item() <= N_WAYS - 1,  "label out of range in context"
assert y_qry.max().item() <= N_WAYS - 1,  "label out of range in query"

ok("Dataset checks passed")

# ── 2. Load CLIP model ────────────────────────────────────────────────────────

section("2. Load CLIP model")

try:
    import open_clip
except ImportError:
    raise ImportError("Run: pip install open-clip-torch")

clip_model, _, _ = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="openai"
)
tokenizer = open_clip.get_tokenizer("ViT-B-32")
clip_model.eval()
ok(f"CLIP ViT-B/32 loaded")

# ── 3. Model forward pass ─────────────────────────────────────────────────────

section("3. INP_MedClassification forward pass")

from argparse import Namespace
from models.inp import INP_MedClassification

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ok(f"Device: {device}")

config = Namespace(
    n_ways=N_WAYS,
    hidden_dim=HIDDEN_DIM,
    clip_dim=CLIP_DIM,
    train_num_z_samples=1,
    test_num_z_samples=4,
)

model = INP_MedClassification(config, clip_model, tokenizer)
model.to(device)

x_ctx_d  = x_ctx.to(device)
y_ctx_d  = y_ctx.to(device)
x_qry_d  = x_qry.to(device)
y_qry_d  = y_qry.to(device)

# Training forward
model.train()
logits, z_samples, q_zCc, q_zCct = model(
    x_ctx_d, y_ctx_d, x_qry_d, y_query=y_qry_d, knowledge=knowledge
)

n_z = config.train_num_z_samples
Q   = 20 * N_WAYS

check_shape(logits,   [n_z, BATCH_SIZE, Q, N_WAYS], "logits (train)")
check_shape(z_samples,[n_z, BATCH_SIZE, N_WAYS, HIDDEN_DIM], "z_samples")
ok(f"q_zCc  mean shape : {list(q_zCc.mean.shape)}")
ok(f"q_zCct mean shape : {list(q_zCct.mean.shape)}")

# Eval forward (no y_query -> q_zCct should be None)
model.eval()
with torch.no_grad():
    logits_eval, _, q_zCc_eval, q_zCct_eval = model(
        x_ctx_d, y_ctx_d, x_qry_d, y_query=None, knowledge=knowledge
    )
assert q_zCct_eval is None, "q_zCct should be None at eval time"
check_shape(logits_eval, [config.test_num_z_samples, BATCH_SIZE, Q, N_WAYS],
            "logits (eval)")

ok("Forward pass checks passed")

# ── 4. Loss ───────────────────────────────────────────────────────────────────

section("4. ELBOLoss (categorical)")

from models.loss import ELBOLoss

loss_fn = ELBOLoss(beta=1.0, categorical=True)

# Training loss
model.train()
loss_fn.train()
logits, z_samples, q_zCc, q_zCct = model(
    x_ctx_d, y_ctx_d, x_qry_d, y_query=y_qry_d, knowledge=knowledge
)
loss, kl, neg_ll = loss_fn((logits, z_samples, q_zCc, q_zCct), y_qry_d)

ok(f"loss   : {loss.item():.4f}")
ok(f"kl     : {kl.item():.4f}")
ok(f"neg_ll : {neg_ll.item():.4f}")

assert torch.isfinite(loss),   "loss is not finite"
assert torch.isfinite(kl),     "kl is not finite"
assert torch.isfinite(neg_ll), "neg_ll is not finite"
ok("Loss values are finite")

# ── 5. Backward pass ─────────────────────────────────────────────────────────

section("5. Backward pass")

loss.backward()

# Check that trainable parameters received gradients
n_with_grad = 0
n_trainable = 0
for name, param in model.named_parameters():
    if param.requires_grad:
        n_trainable += 1
        if param.grad is not None:
            n_with_grad += 1
        else:
            print(f"  ⚠  No gradient: {name}")

ok(f"Gradients received: {n_with_grad}/{n_trainable} trainable parameters")
assert n_with_grad == n_trainable, \
    f"Some trainable parameters have no gradient ({n_with_grad}/{n_trainable})"

# ── 6. Zero-shot forward (k=0) ────────────────────────────────────────────────

section("6. Zero-shot forward (k=0)")

model.train()
x_ctx_zero = torch.zeros(BATCH_SIZE, 1, CLIP_DIM, device=device)
y_ctx_zero = torch.zeros(BATCH_SIZE, 1, dtype=torch.long, device=device)

logits_zero, z_zero, q_zCc_zero, q_zCct_zero = model(
    x_ctx_zero, y_ctx_zero, x_qry_d, y_query=y_qry_d, knowledge=knowledge
)
check_shape(logits_zero, [n_z, BATCH_SIZE, Q, N_WAYS], "logits (zero-shot)")
ok("Zero-shot forward passed")

# ── 7. ClassificationTrainer ──────────────────────────────────────────────────

section("7. ClassificationTrainer.run_batch_train + run_batch_eval")

from models.train_classification import ClassificationTrainer

trainer_config = Namespace(
    n_ways=N_WAYS,
    hidden_dim=HIDDEN_DIM,
    clip_dim=CLIP_DIM,
    train_num_z_samples=1,
    test_num_z_samples=4,
    lr=1e-4,
    beta=1.0,
    num_epochs=1,
    knowledge_mask_rate=0.5,
    knowledge_setup="C",
    use_knowledge=True,
    project_name="smoke_test",
    load_path=None,
    device=device,
)

os.makedirs("saves/smoke_test/run_0", exist_ok=True)
trainer = ClassificationTrainer(
    trainer_config, clip_model, tokenizer,
    save_dir="saves/smoke_test/run_0",
)
trainer.set_dataloaders(train_dl, val_dl)

# run_batch_train
trainer.model.train()
trainer.training_phase = True
train_batch = next(iter(train_dl))
train_results = trainer.run_batch_train(train_batch)

ok(f"train loss     : {train_results['loss'].item():.4f}")
ok(f"train accuracy : {train_results['accuracy']:.2%}")
assert torch.isfinite(train_results["loss"]), "train loss not finite"

# run_batch_eval
trainer.model.eval()
trainer.training_phase = False
val_batch = next(iter(val_dl))
for k_shot in [0, 1, 5]:
    eval_results = trainer.run_batch_eval(val_batch, num_shots_per_class=k_shot)
    ok(f"eval {k_shot}-shot  loss={eval_results['loss']:.4f}  "
       f"acc={eval_results['accuracy']:.2%}")
    assert np.isfinite(eval_results["loss"]), f"eval loss not finite at {k_shot}-shot"

# ── 8. Knowledge masking ──────────────────────────────────────────────────────

section("8. Knowledge masking")

trainer.training_phase = True
trainer.knowledge_mask_rate = 1.0   # force always mask
masked = trainer._maybe_mask_knowledge(knowledge)
assert masked is None, "Knowledge should be None when mask_rate=1.0"
ok("Knowledge masked correctly at rate=1.0")

trainer.knowledge_mask_rate = 0.0   # force never mask
unmasked = trainer._maybe_mask_knowledge(knowledge)
assert unmasked is not None, "Knowledge should not be None when mask_rate=0.0"
ok("Knowledge preserved correctly at rate=0.0")

# ── Done ──────────────────────────────────────────────────────────────────────

section("ALL CHECKS PASSED")
print()
print("  Pipeline is ready for training.")
print("  Next steps:")
print("    Stage 1:  python models/train_classification.py --config config_isic.toml")
print("    Stage 2:  set use_knowledge=true and load_path in config_isic.toml, re-run")
print()
