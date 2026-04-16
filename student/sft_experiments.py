import argparse
import os
from pathlib import Path
from unittest.mock import patch
import time
import json

import torch
import wandb
from datasets import load_from_disk, load_dataset
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from vllm import LLM, SamplingParams

from student.sft_helpers import (
    get_response_log_probs,
    sft_microbatch_train_step,
    tokenize_prompt_and_output,
)
from student.math_baseline_script import evaluate, load_prompt


class InstructDataset(Dataset):
    def __init__(self, examples, max_examples=None):
        if max_examples is not None:
            examples = examples.select(range(min(max_examples, len(examples))))
        self.examples = [examples[i] for i in range(len(examples))]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        msgs = ex.get("messages", [])
        sys_msg  = next((m["content"] for m in msgs if m["role"] == "system"), "")
        user_msg = next((m["content"] for m in msgs if m["role"] == "user"),   "")
        prompt   = (sys_msg + "\n\n" + user_msg).strip() if sys_msg else user_msg
        asst_msg = next((m["content"] for m in msgs if m["role"] == "assistant"), "")
        return {"prompt": prompt, "response": asst_msg}


def collate_fn(batch, tokenizer):
    prompts   = [b["prompt"]   for b in batch]
    responses = [b["response"] for b in batch]
    return tokenize_prompt_and_output(prompts, responses, tokenizer)


def init_vllm(model_id, device, seed, gpu_memory_utilization=0.85):
    from vllm.model_executor import set_random_seed as vllm_set_random_seed
    vllm_set_random_seed(seed)
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch  = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None,
    )
    with world_size_patch, profiling_patch:
        return LLM(
            model=model_id,
            device=device,
            dtype=torch.bfloat16,
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
        )


def load_policy_into_vllm_instance(policy, llm):
    state_dict = policy.state_dict()
    llm_model  = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())


def run_math_eval(llm, policy, prompt_template, math_ds, max_eval=500):
    """Evaluate on MATH dataset."""
    load_policy_into_vllm_instance(policy, llm)
    prompts = [prompt_template + "\n\n" + ex["problem"] for ex in math_ds]
    gts     = [ex["answer"] for ex in math_ds]
    acc, _, _, _ = evaluate(llm, prompts[:max_eval], gts[:max_eval])
    return acc


def run_intellect_eval(llm, prompts, gts):
    """Evaluate on Prime Intellect dataset (already loaded into vLLM)."""
    acc, _, _, _ = evaluate(llm, prompts, gts)
    return acc


def train(args):
    wandb.init(project="llm-reasoners-sft", config=vars(args))
    wandb.define_metric("train_step")
    wandb.define_metric("eval_step")
    wandb.define_metric("train/*", step_metric="train_step")
    wandb.define_metric("eval/*",  step_metric="eval_step")

    device = "cuda:1"
    vllm_device = "cuda:0"

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    # Load model
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    policy = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device)

    # Init vLLM
    print("Initializing vLLM for evaluation...")
    llm = init_vllm(args.model, vllm_device, seed=42,
                    gpu_memory_utilization=args.gpu_memory_utilization)

    # MATH eval dataset
    math_ds = load_dataset("hiyouga/math12k", split="test")
    prompt_template = load_prompt("intellect")

    # Prime Intellect val dataset for periodic eval
    print("Loading Prime Intellect val set...")
    val_raw = load_from_disk(args.val_path)
    val_prompts, val_gts = [], []
    for ex in val_raw:
        msgs = ex.get("messages", [])
        sys_msg  = next((m["content"] for m in msgs if m["role"] == "system"), "")
        user_msg = next((m["content"] for m in msgs if m["role"] == "user"), "")
        prompt   = (sys_msg + "\n\n" + user_msg).strip() if sys_msg else user_msg
        # ground truth is the assistant response for intellect eval
        asst_msg = next((m["content"] for m in msgs if m["role"] == "assistant"), "")
        val_prompts.append(prompt)
        val_gts.append(asst_msg)

    # SFT training dataset
    print(f"Loading Prime Intellect train data from {args.data_path}...")
    raw = load_from_disk(args.data_path)
    dataset = InstructDataset(raw, max_examples=args.max_examples)
    print(f"  Using {len(dataset)} examples")

    loader = DataLoader(
        dataset,
        batch_size=args.micro_batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer),
    )

    gradient_accumulation_steps = max(1, args.batch_size // args.micro_batch_size)

    # Compute total optimizer steps for scheduler
    steps_per_epoch = len(dataset) // args.batch_size
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(args.warmup_ratio * total_steps)
    print(f"  Total optimizer steps: {total_steps}, warmup: {warmup_steps}")

    # Optimizer + scheduler
    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=args.learning_rate,
        weight_decay=0.0,
        betas=(0.9, 0.95),
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    train_step = 0
    eval_step  = 0
    global_start = time.time()

    # Accumulators for averaged logging
    running_loss     = 0.0
    running_grad_norm = 0.0
    running_count    = 0

    # Initial eval
    print("Running initial evaluation...")
    policy.eval()
    math_acc = run_math_eval(llm, policy, prompt_template, math_ds)
    load_policy_into_vllm_instance(policy, llm)
    intellect_acc = run_intellect_eval(llm, val_prompts, val_gts)
    wandb.log({
        "eval/math_accuracy": math_acc,
        "eval/intellect_accuracy": intellect_acc,
        "eval_step": eval_step,
    })
    print(json.dumps({
        "eval_step": eval_step,
        "math_accuracy": round(math_acc, 4),
        "intellect_accuracy": round(intellect_acc, 4),
    }))
    eval_step += 1

    print("Starting SFT training...")
    for epoch in range(args.epochs):
        policy.train()
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(tqdm(loader, desc=f"Epoch {epoch}")):
            input_ids     = batch["input_ids"].to(device)
            labels        = batch["labels"].to(device)
            response_mask = batch["response_mask"].to(device)

            log_probs_out = get_response_log_probs(
                model=policy,
                input_ids=input_ids,
                labels=labels,
                return_token_entropy=False,
            )
            policy_log_probs = log_probs_out["log_probs"]

            loss, metadata = sft_microbatch_train_step(
                policy_log_probs=policy_log_probs,
                response_mask=response_mask,
                gradient_accumulation_steps=gradient_accumulation_steps,
            )

            running_loss  += loss.item() * gradient_accumulation_steps  # unscale
            running_count += 1

            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                running_grad_norm += grad_norm.item()
                train_step += 1

                if train_step % args.log_interval == 0:
                    avg_loss      = running_loss / running_count
                    avg_grad_norm = running_grad_norm / max(1, args.log_interval)
                    elapsed       = time.time() - global_start
                    current_lr    = scheduler.get_last_lr()[0]

                    wandb.log({
                        "train/loss": avg_loss,
                        "train/grad_norm": avg_grad_norm,
                        "train/learning_rate": current_lr,
                        "train/elapsed": elapsed,
                        "train_step": train_step,
                    })

                    print(json.dumps({
                        "train_step": train_step,
                        "loss": round(avg_loss, 4),
                        "grad_norm": round(avg_grad_norm, 4),
                        "lr": f"{current_lr:.2e}",
                        "elapsed": round(elapsed, 1),
                    }), flush=True)

                    running_loss = 0.0
                    running_grad_norm = 0.0
                    running_count = 0

                if train_step > 0 and train_step % args.eval_every == 0:
                    policy.eval()
                    math_acc = run_math_eval(llm, policy, prompt_template, math_ds)
                    load_policy_into_vllm_instance(policy, llm)
                    intellect_acc = run_intellect_eval(llm, val_prompts, val_gts)
                    wandb.log({
                        "eval/math_accuracy": math_acc,
                        "eval/intellect_accuracy": intellect_acc,
                        "eval_step": eval_step,
                    })
                    print(json.dumps({
                        "eval_step": eval_step,
                        "train_step": train_step,
                        "math_accuracy": round(math_acc, 4),
                        "intellect_accuracy": round(intellect_acc, 4),
                    }), flush=True)
                    eval_step += 1
                    policy.train()

    # Save final model
    print(f"Saving model to {args.output_dir}...")
    os.makedirs(args.output_dir, exist_ok=True)
    policy.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Final test evaluation on both datasets
    print("Running final test evaluation...")
    policy.eval()

    test_raw = load_from_disk(args.test_path)
    test_prompts, test_gts = [], []
    for ex in test_raw:
        msgs = ex.get("messages", [])
        sys_msg  = next((m["content"] for m in msgs if m["role"] == "system"), "")
        user_msg = next((m["content"] for m in msgs if m["role"] == "user"), "")
        prompt   = (sys_msg + "\n\n" + user_msg).strip() if sys_msg else user_msg
        asst_msg = next((m["content"] for m in msgs if m["role"] == "assistant"), "")
        test_prompts.append(prompt)
        test_gts.append(asst_msg)

    load_policy_into_vllm_instance(policy, llm)
    intellect_test_acc = run_intellect_eval(llm, test_prompts, test_gts)
    math_test_acc = run_math_eval(llm, policy, prompt_template, math_ds)

    # Save test results
    results = {
        "intellect_test_accuracy": round(intellect_test_acc, 4),
        "math_test_accuracy":      round(math_test_acc, 4),
    }
    with open(os.path.join(args.output_dir, "test_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results), flush=True)
    wandb.log({
        "test/intellect_accuracy": intellect_test_acc,
        "test/math_accuracy":      math_test_acc,
    })

    print("Done!")
    wandb.finish()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",        default="Qwen/Qwen2.5-Math-1.5B")
    parser.add_argument("--data-path",    default="data/intellect_math_train_dev_test/train")
    parser.add_argument("--val-path",     default="data/intellect_math_train_dev_test/dev")
    parser.add_argument("--test-path",    default="data/intellect_math_train_dev_test/test")
    parser.add_argument("--output-dir",   default="/scratch/at6646/sft_model_sweep")
    parser.add_argument("--max-examples",     type=int,   default=None)
    parser.add_argument("--epochs",           type=int,   default=3)
    parser.add_argument("--batch-size",       type=int,   default=32)
    parser.add_argument("--micro-batch-size", type=int,   default=2)
    parser.add_argument("--learning-rate",    type=float, default=2e-5)
    parser.add_argument("--warmup-ratio",     type=float, default=0.1)
    parser.add_argument("--eval-every",       type=int,   default=50)
    parser.add_argument("--log-interval",     type=int,   default=10)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.7)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()