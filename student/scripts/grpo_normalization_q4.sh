#!/bin/bash
#SBATCH --job-name=grpo_q4
#SBATCH --output=./grpo_norm_logs_v3_q4/%j_%x.out
#SBATCH --error=./grpo_norm_logs_v3_q4/%j_%x.err
#SBATCH --mail-type=END
#SBATCH --mail-user=at6646@nyu.edu
#SBATCH --partition=a100_dev
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=4:00:00
#SBATCH --requeue

set -e
mkdir -p grpo_norm_logs_v3_q4

echo "############### Run Log: $(date +%Y-%m-%d_%H:%M:%S) ###############"
nvidia-smi

MODEL="/gpfs/scratch/an4462/at6646/llmr-a3/models/models/Qwen2.5-Math-1.5B-Instruct"
TRAIN_PATH="/gpfs/scratch/an4462/at6646/llmr-a3/data/data-distrib/countdown/train_10k.parquet"
VAL_PATH="/gpfs/scratch/an4462/at6646/llmr-a3/data/data-distrib/countdown/dev.parquet"
TEST_PATH="/gpfs/scratch/an4462/at6646/llmr-a3/data/data-distrib/countdown/test.parquet"
OUTPUT_DIR="/gpfs/scratch/an4462/at6646/llmr-a3/grpo_norm_v3_q4_lr3e-5/grpo_model_masked_normalize"

export HF_HOME=/gpfs/scratch/an4462/at6646/hf_cache
export TRANSFORMERS_CACHE=/gpfs/scratch/an4462/at6646/hf_cache
export HF_DATASETS_CACHE=/gpfs/scratch/an4462/at6646/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Starting GRPO masked_normalize run..."

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
    --learning-rate 3e-5 \
    --sampling-temperature 0.7 \
    --sampling-min-tokens 4 \
    --sampling-max-tokens 1024 \
    --loss-type reinforce_with_baseline \
    --norm-type masked_normalize \
    --use-std-normalization \
    --eval-interval 10 \
    --max-val-examples 200 \
    --gpu-memory-utilization 0.45 \
    --wandb-project "llm-reasoners-grpo" \
    --wandb-name "norm_masked_normalize"

echo "Done: $(date +%Y-%m-%d_%H:%M:%S)"