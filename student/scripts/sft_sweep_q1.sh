#!/bin/bash
#SBATCH --job-name=q1-sft
#SBATCH --output=./logs_sft_q1/%j_%x.out
#SBATCH --error=./logs_sft_q1/%j_%x.err
#SBATCH --mail-type=END
#SBATCH --mail-user=at6646@nyu.edu
#SBATCH --partition=a100_long
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=6:00:00
#SBATCH --requeue
#SBATCH --array=0-4%2

set -e

echo "############### Run Log: $(date +%Y-%m-%d_%H:%M:%S) ###############"
echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
nvidia-smi

# Dataset sizes
MAX_EXAMPLES_CONFIGS=(128 256 512 1024 "None")
MAX_EXAMPLES=${MAX_EXAMPLES_CONFIGS[$SLURM_ARRAY_TASK_ID]}

# Hyperparameters
LEARNING_RATE=2e-5
BATCH_SIZE=32          # effective batch size
MICRO_BATCH_SIZE=2     # actual DataLoader batch size
WARMUP_RATIO=0.1
LOG_INTERVAL=10
SEED=42
GPU_MEMORY_UTILIZATION=0.3  # vLLM on cuda:0, keep low

# Epochs per dataset size
EPOCHS_CONFIGS=(50 30 20 10 5)
NUM_EPOCHS=${EPOCHS_CONFIGS[$SLURM_ARRAY_TASK_ID]}

EVAL_POINTS=10

# Paths
MODEL="/gpfs/scratch/an4462/at6646/llmr-a3/models/Qwen2.5-Math-1.5B"
TRAIN_PATH="/gpfs/scratch/an4462/at6646/llmr-a3/data/data-distrib/intellect_math/train"
VAL_PATH="/gpfs/scratch/an4462/at6646/llmr-a3/data/data-distrib/intellect_math/dev"
TEST_PATH="/gpfs/scratch/an4462/at6646/llmr-a3/data/data-distrib/intellect_math/test"

# Cache dirs
export HF_HOME=/gpfs/scratch/an4462/at6646/hf_cache
export TRANSFORMERS_CACHE=/gpfs/scratch/an4462/at6646/hf_cache
export HF_DATASETS_CACHE=/gpfs/scratch/an4462/at6646/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Resolve run config
if [ "$MAX_EXAMPLES" = "None" ]; then
    RUN_NAME="sft-full"
    OUTPUT_DIR="/gpfs/scratch/an4462/at6646/llmr-a3/sft_sweep_checkpoints_v2_q1/full_ep${NUM_EPOCHS}"
    MAX_EXAMPLES_ARG=""
    N_EXAMPLES=$(uv run python -c \
        "from datasets import load_from_disk; print(len(load_from_disk('$TRAIN_PATH')))")
else
    RUN_NAME="sft-${MAX_EXAMPLES}"
    OUTPUT_DIR="/gpfs/scratch/an4462/at6646/llmr-a3/sft_sweep_checkpoints_v2_q1/${MAX_EXAMPLES}_ep${NUM_EPOCHS}"
    MAX_EXAMPLES_ARG="--max-examples $MAX_EXAMPLES"
    N_EXAMPLES=$MAX_EXAMPLES
fi

# Compute total optimizer steps and eval interval
EFFECTIVE_BATCH_SIZE=$BATCH_SIZE  # = 32
TOTAL_STEPS=$(( (N_EXAMPLES * NUM_EPOCHS) / EFFECTIVE_BATCH_SIZE ))
EVAL_INTERVAL=$(( TOTAL_STEPS / EVAL_POINTS ))
if [ "$EVAL_INTERVAL" -eq 0 ]; then EVAL_INTERVAL=1; fi

echo "=== SFT CONFIG ==="
echo "Dataset size:       $MAX_EXAMPLES (actual: $N_EXAMPLES)"
echo "Epochs:             $NUM_EPOCHS"
echo "Effective batch:    $EFFECTIVE_BATCH_SIZE"
echo "Total steps:        $TOTAL_STEPS"
echo "Eval every:         $EVAL_INTERVAL steps"
echo "Output:             $OUTPUT_DIR"

mkdir -p logs_sft
mkdir -p "$OUTPUT_DIR"

echo "Starting SFT training..."
uv run python /gpfs/scratch/an4462/at6646/llmr-a3-cp/student/sft_experiments.py \
    --model "$MODEL" \
    --data-path "$TRAIN_PATH" \
    --val-path "$VAL_PATH" \
    --test-path "$TEST_PATH" \
    --output-dir "$OUTPUT_DIR" \
    $MAX_EXAMPLES_ARG \
    --epochs $NUM_EPOCHS \
    --batch-size $BATCH_SIZE \
    --micro-batch-size $MICRO_BATCH_SIZE \
    --learning-rate $LEARNING_RATE \
    --warmup-ratio $WARMUP_RATIO \
    --eval-every $EVAL_INTERVAL \
    --log-interval $LOG_INTERVAL \
    --gpu-memory-utilization $GPU_MEMORY_UTILIZATION

echo "############### End: $(date +%Y-%m-%d_%H:%M:%S) ###############"