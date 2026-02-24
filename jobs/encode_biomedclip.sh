#!/bin/sh
#SBATCH --job-name=biomedclip-embed
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/usr/bmicnas03/data-biwi-01/ldrole_student/data/work/cam_phd/informed-meta-learning/out/%j.out
#SBATCH --error=/usr/bmicnas03/data-biwi-01/ldrole_student/data/work/cam_phd/informed-meta-learning/out/%j.err
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0-02:00:00


set -o errexit

export OMP_NUM_THREADS=4

echo "Script name:	encode_biomedclip.sh"
echo "Running on node:	$(hostname -f)"
echo "Starting on:	$(date)"
echo "SLURM_JOB_ID:	${SLURM_JOB_ID:-N/A}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"

eval "$(conda shell.bash hook)"
conda activate inps

echo "python: $(which python) — $(python --version 2>&1)"

REPO=/usr/bmicnas03/data-biwi-01/ldrole_student/data/work/cam_phd/informed-meta-learning
export PYTHONPATH=${REPO}:${PYTHONPATH}
cd ${REPO}

# ── Step 1: Download BiomedCLIP weights (skip if already done) ────────────
if [ ! -f data/biomedclip/open_clip_pytorch_model.bin ]; then
    echo "Downloading BiomedCLIP weights..."
    python download_biomedclip.py
else
    echo "BiomedCLIP weights already present, skipping download."
fi

# ── Step 2: Encode ISIC 2019 ─────────────────────────────────────────────
echo ""
echo "=== Encoding ISIC 2019 ==="
python dataset/isic.py \
    --data-root data/isic2019 \
    --encoder biomedclip \
    --biomedclip-dir data/biomedclip \
    --batch-size 128

# ── Step 3: Encode DermNet ────────────────────────────────────────────────
echo ""
echo "=== Encoding DermNet ==="
python dataset/dermnet.py \
    --data-root data/dermnet \
    --encoder biomedclip \
    --biomedclip-dir data/biomedclip \
    --batch-size 128

echo ""
echo "=== Done ==="
echo "Produced:"
ls -lh data/isic2019/embeddings_biomedclip.pt
ls -lh data/dermnet/embeddings_biomedclip.pt
echo "Finished at:	$(date)"
exit 0





























