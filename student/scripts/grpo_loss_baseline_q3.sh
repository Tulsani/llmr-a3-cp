#!/bin/bash
#SBATCH --job-name=grpo-q3
#SBATCH --output=./grpo_baseline_logs_q3/%j_%x_%a.out
#SBATCH --error=./grpo_baseline_logs_q3/%j_%x_%a.err
#SBATCH --mail-type=END
#SBATCH --mail-user=at6646@nyu.edu
#SBATCH --partition=a100_dev
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=4:00:00
#SBATCH --requeue
#SBATCH --array=0-1

set -e
mkdir -p grpo_baseline_logs

echo "############### Run Log: $(date +%Y-%m-%d_%H:%M:%S) ###############"
echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
nvidia-smi

# --- Baseline sweep ---
LOSS_CONFIGS=(no_baseline reinforce_with_baseline)
LOSS_TYPE=${LOSS_CONFIGS[$SLURM_ARRAY_TASK_ID]}

# --- Best LR from sweep ---
LEARNING_RATE=3e-5

# --- Paths ---
MODEL="/gpfs/scratch/an4462/at6646/llmr-a3/models/models/Qwen2.5-Math-1.5B-Instruct"
TRAIN_PATH="/gpfs/scratch/an4462/at6646/llmr-a3/data/data-distrib/countdown/train_10k.parquet"
VAL_PATH="/gpfs/scratch/an4462/at6646/llmr-a3/data/data-distrib/countdown/dev.parquet"
TEST_PATH="/gpfs/scratch/an4462/at6646/llmr-a3/data/data-distrib/countdown/test.parquet"
OUTPUT_DIR="/gpfs/scratch/an4462/at6646/llmr-a3/grpo_baseline_q3_lr3e-5/grpo_model_${LOSS_TYPE}"

# --- Cache dirs ---
export HF_HOME=/gpfs/scratch/an4462/at6646/hf_cache
export TRANSFORMERS_CACHE=/gpfs/scratch/an4462/at6646/hf_cache
export HF_DATASETS_CACHE=/gpfs/scratch/an4462/at6646/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Starting GRPO baseline comparison..."
echo "  loss_type:     $LOSS_TYPE"
echo "  learning_rate: $LEARNING_RATE"
echo "  output_dir:    $OUTPUT_DIR"

uv run python /gpfs/scratch/an4462/at6646/llmr-a3-cp/student/grpo_experiments.py \
    --model "$MODEL" \
    --train-path "$TRAIN_PATH" \
    --val-path "$VAL_PATH" \
    --test-path "$TEST_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --policy-device cuda:1 \
    --vllm-device cuda:0 \
    --n-grpo-steps 200 \
    --rollout-batch-size 32 \
    --group-size 4 \
    --gradient-accumulation-steps 8 \
    --epochs-per-rollout-batch 1 \
    --learning-rate $LEARNING_RATE \
    --sampling-temperature 0.7 \
    --sampling-min-tokens 4 \
    --sampling-max-tokens 1024 \
    --loss-type $LOSS_TYPE \
    --use-std-normalization \
    --eval-interval 10 \
    --max-val-examples 200 \
    --gpu-memory-utilization 0.45 \
    --wandb-project "llm-reasoners-grpo" \
    --wandb-name "baseline_${LOSS_TYPE}"

echo "Done: $(date +%Y-%m-%d_%H:%M:%S)"