#!/bin/bash
#SBATCH --job-name=grpo_q1
#SBATCH --output=./grpo_logs_v3_q1/%j_%x.out
#SBATCH --error=./grpo_logs_v3_q1/%j_%x.err
#SBATCH --mail-type=END
#SBATCH --mail-user=at6646@nyu.edu
#SBATCH --partition=a100_dev
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=4:00:00
#SBATCH --requeue

set -e
mkdir -p grpo_logs

echo "############### Run Log: $(date +%Y-%m-%d_%H:%M:%S) ###############"
nvidia-smi

# --- Paths ---
MODEL="/gpfs/scratch/an4462/at6646/llmr-a3/models/models/Qwen2.5-Math-1.5B-Instruct"
TRAIN_PATH="/gpfs/scratch/an4462/at6646/llmr-a3/data/data-distrib/countdown/train_10k.parquet"
VAL_PATH="/gpfs/scratch/an4462/at6646/llmr-a3/data/data-distrib/countdown/dev.parquet"
TEST_PATH="/gpfs/scratch/an4462/at6646/llmr-a3/data/data-distrib/countdown/test.parquet"
OUTPUT_DIR="/gpfs/scratch/an4462/at6646/llmr-a3/grpo_model_v3_q1/"

# --- Cache dirs ---
export HF_HOME=/gpfs/scratch/an4462/at6646/hf_cache
export TRANSFORMERS_CACHE=/gpfs/scratch/an4462/at6646/hf_cache
export HF_DATASETS_CACHE=/gpfs/scratch/an4462/at6646/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --- Hyperparams ---
ROLLOUT_BATCH_SIZE=16
GROUP_SIZE=8
GRAD_ACCUM_STEPS=8
EPOCHS_PER_ROLLOUT=1      # on-policy

LEARNING_RATE=1e-5        # best LR from sweep
N_GRPO_STEPS=200
SAMPLING_TEMP=0.7
EVAL_EVERY=10
N_EVAL_EXAMPLES=200
GPU_MEM_UTIL=0.45

echo "Starting GRPO training..."

uv run python /gpfs/scratch/an4462/at6646/llmr-a3-cp/student/grpo_experiments.py \
    --model "$MODEL" \
    --train-path "$TRAIN_PATH" \
    --val-path "$VAL_PATH" \
    --test-path "$TEST_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --policy-device cuda:1 \
    --vllm-device cuda:0 \
    --n-grpo-steps $N_GRPO_STEPS \
    --rollout-batch-size $ROLLOUT_BATCH_SIZE \
    --group-size $GROUP_SIZE \
    --gradient-accumulation-steps $GRAD_ACCUM_STEPS \
    --epochs-per-rollout-batch $EPOCHS_PER_ROLLOUT \
    --learning-rate $LEARNING_RATE \
    --sampling-temperature $SAMPLING_TEMP \
    --sampling-min-tokens 4 \
    --sampling-max-tokens 1024 \
    --loss-type reinforce_with_baseline \
    --use-std-normalization \
    --eval-interval $EVAL_EVERY \
    --max-val-examples $N_EVAL_EXAMPLES \
    --gpu-memory-utilization $GPU_MEM_UTIL \
    --wandb-project "llm-reasoners-grpo" \
    --wandb-name "grpo_main_run"

echo "Done: $(date +%Y-%m-%d_%H:%M:%S)"