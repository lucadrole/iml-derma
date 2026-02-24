#!/bin/sh
#SBATCH --job-name=inp-medclass
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/usr/bmicnas03/data-biwi-01/ldrole_student/data/work/cam_phd/informed-meta-learning/out/%j.out
#SBATCH --error=/usr/bmicnas03/data-biwi-01/ldrole_student/data/work/cam_phd/informed-meta-learning/out/%j.err
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0-12:00:00

# Exit on error
set -o errexit

# Allocator hygiene
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,garbage_collection_threshold:0.8,expandable_segments:True
export OMP_NUM_THREADS=4

# Log info
echo "Script name:	train_inp.sh"
echo "Running on node:	$(hostname -f)"
echo "In directory:	$(pwd)"
echo "Starting on:	$(date)"
echo "SLURM_JOB_ID:	${SLURM_JOB_ID:-N/A}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"

# Activate environment
eval "$(conda shell.bash hook)"
conda activate /usr/bmicnas03/data-biwi-01/ldrole_student/data/envs/senatraenv

echo "After conda activate:"
echo "  which python: $(which python || echo 'none')"
echo "  python --version: $(python --version 2>&1 || echo 'none')"

# WandB
export WANDB_API_KEY="21eec93d4b125e6132dbf2900d9ba82240aef3aa"
export WANDB_ENTITY=lucadrole-eth-zurich
export WANDB_PROJECT=INPs_isic_dermnet
export WANDB_MODE=online

# Set up paths
REPO=/usr/bmicnas03/data-biwi-01/ldrole_student/data/work/cam_phd/informed-meta-learning
export PYTHONPATH=${REPO}:${PYTHONPATH}
cd ${REPO}

# Create output dir if needed
mkdir -p out

# ── Choose what to run ────────────────────────────────────────────────────
# Uncomment ONE of the blocks below.

# --- Option A: Build BiomedCLIP embeddings (run once) ---
# python dataset/isic.py --data-root data/isic2019 --encoder biomedclip --biomedclip-dir data/biomedclip
# python dataset/dermnet.py --data-root data/dermnet --encoder biomedclip --biomedclip-dir data/biomedclip
conda activate inps
# --- Option B: Train with mixed ISIC + DermNet (BiomedCLIP) ---
python models/train_classification.py --config /home/ldrole/my_space/work/cam_phd/informed-meta-learning/config_isic_biomedclip.toml

# --- Option C: Train ISIC-only baseline (for comparison) ---
# python train_classification.py --config config_isic.toml

# Log end
echo "Finished at:	$(date)"
exit 0





























