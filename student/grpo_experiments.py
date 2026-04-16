import argparse
import json
import os
import random
import re
import time
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import torch
import wandb
from datasets import load_dataset, load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)
from vllm import LLM, SamplingParams
from vllm.model_executor import set_random_seed as vllm_set_random_seed

from student.grpo_helpers import (
    compute_group_normalized_rewards,
    grpo_microbatch_train_step,
)
from student.sft_helpers import get_response_log_probs, tokenize_prompt_and_output

def load_countdown_data(path: str) -> list[dict]:
    p = Path(path)
    if p.is_dir():
        parquet_files = list(p.glob("*.parquet"))
        if parquet_files:
            ds = load_dataset(
                "parquet",
                data_files=[str(f) for f in parquet_files],
                split="train",
            )
        else:
            ds = load_from_disk(path)
    else:
        ds = load_dataset("parquet", data_files=str(p), split="train")

    examples = []
    for ex in ds:
        numbers = list(ex.get("nums", ex.get("numbers", [])))
        target = int(ex.get("target", ex.get("answer", 0)))
        examples.append({"numbers": numbers, "target": target})
    return examples



def load_countdown_prompt() -> str:
    path = Path(__file__).parent / "prompts" / "countdown.prompt"
    return path.read_text().strip()


def make_prompt(prompt_template: str, ex: dict) -> str:
    numbers = ex["numbers"]
    target = ex["target"]
    problem = (
        f"Using the numbers in the list {numbers}, create an equation that equals {target}. "
        f"You may use the basic arithmetic operations (+, -, *, /), and each number can be used at most once."
    )

    if "{question}" in prompt_template:
        return prompt_template.replace("{question}", problem)
    return f"{prompt_template}\n\n{problem}"


def make_ground_truth(ex: dict) -> str:
    return json.dumps(
        {
            "target": int(ex["target"]),
            "numbers": sorted(int(x) for x in ex["numbers"]),
        }
    )


def _try_evaluate(expr: str):
    safe_chars = set("0123456789+-*/() .\t\n")
    if not expr or not all(c in safe_chars for c in expr):
        return None
    try:
        return float(eval(expr, {"__builtins__": {}}, {}))
    except Exception:
        return None


def _extract_answer_text(response: str) -> str | None:
    if "<answer>" not in response or "</answer>" not in response:
        return None
    return response.split("<answer>", 1)[-1].split("</answer>", 1)[0].strip()


def _extract_candidate_expressions(answer_text: str) -> list[str]:
    candidates = []

    for raw_line in answer_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        # Remove Step prefixes 
        line = re.sub(r"^\s*Step\s*\d+\s*[:.]\s*", "", line).strip()
        if not line:
            continue

        # If there is an "=" use the LHS first
        if "=" in line:
            lhs = line.rsplit("=", 1)[0].strip()
            if lhs:
                candidates.append(lhs)
        else:
            candidates.append(line)

    # Fallback
    flattened = " ".join(answer_text.split())
    if flattened:
        candidates.append(flattened)

    # Preserve order
    deduped = []
    seen = set()
    for c in candidates:
        if c not in seen:
            deduped.append(c)
            seen.add(c)
    return deduped


def countdown_reward_fn(response: str, ground_truth: str) -> dict[str, float]:
    gt = json.loads(ground_truth)
    target = int(gt["target"])
    allowed = sorted(int(x) for x in gt["numbers"])

    answer_text = _extract_answer_text(response)
    if answer_text is None:
        return {"format_reward": 0.0, "answer_reward": 0.0, "reward": 0.0}

    candidates = _extract_candidate_expressions(answer_text)

    for expr in candidates:
        result = _try_evaluate(expr)
        if result is None:
            continue
        if abs(result - target) >= 1e-6:
            continue

        nums_used = sorted(int(n) for n in re.findall(r"\b\d+\b", expr))
        if Counter(nums_used) == Counter(allowed):
            return {"format_reward": 1.0, "answer_reward": 1.0, "reward": 1.0}

    # Properly formatted but not correct
    return {"format_reward": 1.0, "answer_reward": 0.0, "reward": 0.0}



def init_vllm(
    model_id: str,
    device: str,
    seed: int,
    gpu_memory_utilization: float = 0.8,
) -> LLM:

    vllm_set_random_seed(seed)
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
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


def load_policy_into_vllm(policy, llm: LLM):
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())



def evaluate_countdown(
    llm: LLM,
    examples: list[dict],
    prompt_template: str,
    reward_fn,
    max_examples: int | None = None,
) -> dict[str, float]:
    if max_examples is not None:
        examples = examples[:max_examples]

    prompts = [make_prompt(prompt_template, ex) for ex in examples]
    ground_truths = [make_ground_truth(ex) for ex in examples]

    params = SamplingParams(
        temperature=0.0,
        max_tokens=1024,
        stop=["</answer>"],
    )
    outputs = llm.generate(prompts, params)

    # Debug: print first sample output to verify format
    print("DEBUG sample output:", repr(outputs[0].outputs[0].text[:400]))

    total = 0.0
    fmt = 0.0
    ans = 0.0

    for out, gt in zip(outputs, ground_truths):
        text = out.outputs[0].text + "</answer>"
        reward = reward_fn(text, gt)
        total += reward["reward"]
        fmt += reward["format_reward"]
        ans += reward["answer_reward"]

    n = max(len(examples), 1)
    return {
        "accuracy": total / n,
        "format_accuracy": fmt / n,
        "answer_accuracy": ans / n,
    }



def grpo_train_loop(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)

    device = torch.device(args.policy_device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.wandb_project is not None:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            entity=args.wandb_entity,
            config=vars(args),
        )
        wandb.define_metric("train_step")
        wandb.define_metric("eval_step")
        wandb.define_metric("train/*", step_metric="train_step")
        wandb.define_metric("eval/*", step_metric="eval_step")


    assert args.rollout_batch_size % args.group_size == 0, (
        "rollout_batch_size must be divisible by group_size"
    )

    n_prompts_per_rollout_batch = args.rollout_batch_size // args.group_size
    train_batch_size = args.rollout_batch_size * args.epochs_per_rollout_batch

    assert train_batch_size % args.gradient_accumulation_steps == 0, (
        "Derived train_batch_size must be divisible by gradient_accumulation_steps"
    )

    micro_train_batch_size = train_batch_size // args.gradient_accumulation_steps
    n_microbatches_per_rollout_batch = (
        args.rollout_batch_size // micro_train_batch_size
    )

    normalize_constant = None
    if args.norm_type == "masked_normalize":
        normalize_constant = (
            args.normalize_constant
            if args.normalize_constant is not None
            else float(args.sampling_max_tokens)
        )

    print(f"n_prompts_per_rollout_batch: {n_prompts_per_rollout_batch}")
    print(f"train_batch_size: {train_batch_size}")
    print(f"micro_train_batch_size: {micro_train_batch_size}")
    print(
        f"n_microbatches_per_rollout_batch (per epoch): "
        f"{n_microbatches_per_rollout_batch}"
    )


    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    policy = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="sdpa",
    ).to(device)
    policy.gradient_checkpointing_enable()

    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=args.learning_rate,
        weight_decay=0.0,
        betas=(0.9, 0.95),
    )

    warmup_steps = int(args.warmup_ratio * args.n_grpo_steps)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=args.n_grpo_steps,
    )


    print("Initializing vLLM...")
    llm = init_vllm(
        args.model,
        args.vllm_device,
        seed=args.seed,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )


    print("Loading Countdown datasets...")
    train_data = load_countdown_data(args.train_path)
    val_data = load_countdown_data(args.val_path)
    test_data = load_countdown_data(args.test_path)

    prompt_template = load_countdown_prompt()
    reward_fn = countdown_reward_fn

    val_examples = (
        val_data[: args.max_val_examples]
        if args.max_val_examples is not None
        else val_data
    )

    print(
        f"train: {len(train_data)} | val: {len(val_data)} | "
        f"eval_val: {len(val_examples)} | test: {len(test_data)}"
    )

    sampling_params = SamplingParams(
        temperature=args.sampling_temperature,
        min_tokens=args.sampling_min_tokens,
        max_tokens=args.sampling_max_tokens,
        stop=["</answer>"],
    )


    print("\nRunning initial evaluation...")
    eval_step = 0
    policy.eval()
    load_policy_into_vllm(policy, llm)

    eval_results = evaluate_countdown(
        llm=llm,
        examples=val_examples,
        prompt_template=prompt_template,
        reward_fn=reward_fn,
    )
    print(json.dumps({"step": 0, **eval_results}))

    if args.wandb_project is not None:
        wandb.log(
            {
                "eval/accuracy": eval_results["accuracy"],
                "eval/format_accuracy": eval_results["format_accuracy"],
                "eval/answer_accuracy": eval_results["answer_accuracy"],
                "eval_step": eval_step,
            }
        )


    print(f"\nStarting GRPO for {args.n_grpo_steps} steps...")
    global_start = time.time()

    running = {
        "loss": 0.0,
        "reward": 0.0,
        "format_reward": 0.0,
        "answer_reward": 0.0,
        "entropy": 0.0,
        "grad_norm": 0.0,
        "clip_frac": 0.0,
        "count": 0,
    }

    for grpo_step in range(args.n_grpo_steps):

        questions = random.sample(train_data, n_prompts_per_rollout_batch)

        repeated_prompts = [
            make_prompt(prompt_template, q)
            for q in questions
            for _ in range(args.group_size)
        ]
        repeated_gts = [
            make_ground_truth(q)
            for q in questions
            for _ in range(args.group_size)
        ]


        policy.eval()
        load_policy_into_vllm(policy, llm)
        outputs = llm.generate(repeated_prompts, sampling_params)
        rollout_responses = [out.outputs[0].text + "</answer>" for out in outputs]


        all_reward_dicts = [
            reward_fn(r, gt)
            for r, gt in zip(rollout_responses, repeated_gts)
        ]

        step_format_reward = sum(d["format_reward"] for d in all_reward_dicts) / max(
            len(all_reward_dicts), 1
        )
        step_answer_reward = sum(d["answer_reward"] for d in all_reward_dicts) / max(
            len(all_reward_dicts), 1
        )

        reward_iter = iter(all_reward_dicts)
        advantages, raw_rewards, reward_metadata = compute_group_normalized_rewards(
            reward_fn=lambda response, gt: next(reward_iter),
            rollout_responses=rollout_responses,
            repeated_ground_truths=repeated_gts,
            group_size=args.group_size,
            advantage_eps=args.advantage_eps,
            normalize_by_std=args.use_std_normalization,
        )

        tokenized = tokenize_prompt_and_output(
            repeated_prompts,
            rollout_responses,
            tokenizer,
        )
        input_ids = tokenized["input_ids"].to(device)
        labels = tokenized["labels"].to(device)
        response_mask = tokenized["response_mask"].float().to(device)

        adv_tensor = advantages.to(device).unsqueeze(1)
        raw_tensor = raw_rewards.to(device).unsqueeze(1)

        need_old_log_probs = (
            args.loss_type == "grpo_clip" or args.epochs_per_rollout_batch > 1
        )

        if need_old_log_probs:
            policy.eval()
            with torch.inference_mode():
                old_log_probs_all = get_response_log_probs(
                    model=policy,
                    input_ids=input_ids,
                    labels=labels,
                    return_token_entropy=False,
                )["log_probs"].detach()
        else:
            old_log_probs_all = None


        epoch_loss = 0.0
        epoch_entropy = 0.0
        epoch_clip_frac = 0.0
        epoch_grad_norm = 0.0
        opt_steps_this_rollout = 0

        for epoch in range(args.epochs_per_rollout_batch):
            optimizer.zero_grad()

            indices = list(range(args.rollout_batch_size))
            random.shuffle(indices)

            for mb_idx in range(n_microbatches_per_rollout_batch):
                mb_start = mb_idx * micro_train_batch_size
                mb_end = mb_start + micro_train_batch_size
                mb_ids = indices[mb_start:mb_end]

                mb_input_ids = input_ids[mb_ids]
                mb_labels = labels[mb_ids]
                mb_response_mask = response_mask[mb_ids]
                mb_adv = adv_tensor[mb_ids]
                mb_raw = raw_tensor[mb_ids]
                mb_old_lp = (
                    old_log_probs_all[mb_ids]
                    if old_log_probs_all is not None
                    else None
                )

                policy.train()
                log_probs_out = get_response_log_probs(
                    model=policy,
                    input_ids=mb_input_ids,
                    labels=mb_labels,
                    return_token_entropy=True,
                )
                mb_policy_log_probs = log_probs_out["log_probs"]
                token_entropy = log_probs_out["token_entropy"]

                loss, metadata = grpo_microbatch_train_step(
                    policy_log_probs=mb_policy_log_probs,
                    response_mask=mb_response_mask,
                    gradient_accumulation_steps=args.gradient_accumulation_steps,
                    loss_type=args.loss_type,
                    raw_rewards=mb_raw,
                    advantages=mb_adv,
                    old_log_probs=mb_old_lp,
                    cliprange=args.cliprange,
                    normalize_constant=normalize_constant,
                )

                with torch.no_grad():
                    denom = mb_response_mask.sum().clamp(min=1)
                    mean_entropy = (
                        (token_entropy * mb_response_mask).sum() / denom
                    ).item()
                    epoch_entropy += mean_entropy

                if "is_clipped" in metadata:
                    clip_frac = (
                        (metadata["is_clipped"].float() * mb_response_mask).sum()
                        / mb_response_mask.sum().clamp(min=1)
                    ).item()
                    epoch_clip_frac += clip_frac
                elif "clip_fraction" in metadata:
                    cf = metadata["clip_fraction"]
                    epoch_clip_frac += cf.item() if torch.is_tensor(cf) else float(cf)

                epoch_loss += loss.item()

            grad_norm = torch.nn.utils.clip_grad_norm_(
                policy.parameters(),
                args.grad_clip,
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            epoch_grad_norm += grad_norm.item()
            opt_steps_this_rollout += 1

        n_mb_total = (
            n_microbatches_per_rollout_batch * args.epochs_per_rollout_batch
        )


        running["loss"] += epoch_loss / max(n_mb_total, 1)
        running["reward"] += reward_metadata["mean_reward"]
        running["format_reward"] += step_format_reward
        running["answer_reward"] += step_answer_reward
        running["entropy"] += epoch_entropy / max(n_mb_total, 1)
        running["grad_norm"] += epoch_grad_norm / max(opt_steps_this_rollout, 1)
        running["clip_frac"] += epoch_clip_frac / max(n_mb_total, 1)
        running["count"] += 1

        step = grpo_step + 1


        if step % args.log_interval == 0:
            c = max(running["count"], 1)
            current_lr = scheduler.get_last_lr()[0]
            log_dict = {
                "train/loss": running["loss"] / c,
                "train/mean_reward": running["reward"] / c,
                "train/format_reward": running["format_reward"] / c,
                "train/answer_reward": running["answer_reward"] / c,
                "train/token_entropy": running["entropy"] / c,
                "train/grad_norm": running["grad_norm"] / c,
                "train/clip_frac": running["clip_frac"] / c,
                "train/learning_rate": current_lr,
                "train/elapsed": time.time() - global_start,
            }

            print(
                json.dumps(
                    {
                        "step": step,
                        "loss": round(log_dict["train/loss"], 4),
                        "reward": round(log_dict["train/mean_reward"], 4),
                        "grad_norm": round(log_dict["train/grad_norm"], 4),
                    }
                )
            )

            if args.wandb_project is not None:
                wandb.log({**log_dict, "train_step": step})

            running = {
                "loss": 0.0,
                "reward": 0.0,
                "format_reward": 0.0,
                "answer_reward": 0.0,
                "entropy": 0.0,
                "grad_norm": 0.0,
                "clip_frac": 0.0,
                "count": 0,
            }

        if step % args.eval_interval == 0:
            print(f"\nEvaluating at step {step}...")
            policy.eval()
            load_policy_into_vllm(policy, llm)

            eval_results = evaluate_countdown(
                llm=llm,
                examples=val_examples,
                prompt_template=prompt_template,
                reward_fn=reward_fn,
            )

            eval_step += 1
            print(json.dumps({"step": step, **eval_results}))

            if args.wandb_project is not None:
                wandb.log(
                    {
                        "eval/accuracy": eval_results["accuracy"],
                        "eval/format_accuracy": eval_results["format_accuracy"],
                        "eval/answer_accuracy": eval_results["answer_accuracy"],
                        "eval_step": eval_step,
                    }
                )

            ckpt_dir = output_dir / f"step_{step}"
            policy.save_pretrained(str(ckpt_dir))
            tokenizer.save_pretrained(str(ckpt_dir))
            print(f"Saved checkpoint to {ckpt_dir}")


    final_dir = output_dir / "final"
    policy.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"\nSaved final model to {final_dir}")


    print("\nRunning final test evaluation...")
    policy.eval()
    load_policy_into_vllm(policy, llm)

    test_results = evaluate_countdown(
        llm=llm,
        examples=test_data,
        prompt_template=prompt_template,
        reward_fn=reward_fn,
    )
    print(json.dumps({"final_test": test_results}))

    if args.wandb_project is not None:
        wandb.log(
            {
                "test/accuracy": test_results["accuracy"],
                "test/format_accuracy": test_results["format_accuracy"],
                "test/answer_accuracy": test_results["answer_accuracy"],
            }
        )
        wandb.finish()

    print("Done")



def main():
    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument("--train-path", default="data/countdown/train")
    parser.add_argument("--val-path", default="data/countdown/dev")
    parser.add_argument("--test-path", default="data/countdown/test")


    parser.add_argument("--model", default="Qwen/Qwen2.5-Math-1.5B-Instruct")
    parser.add_argument("--output-dir", default="/scratch/at6646/llmr-a3/grpo_model")

    parser.add_argument("--policy-device", default="cuda:1")
    parser.add_argument("--vllm-device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.45)

    parser.add_argument("--n-grpo-steps", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--advantage-eps", type=float, default=1e-6)
    parser.add_argument("--rollout-batch-size", type=int, default=16)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--epochs-per-rollout-batch", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)


    parser.add_argument("--sampling-temperature", type=float, default=0.7)
    parser.add_argument("--sampling-min-tokens", type=int, default=4)
    parser.add_argument("--sampling-max-tokens", type=int, default=1024)

    parser.add_argument(
        "--loss-type",
        default="reinforce_with_baseline",
        choices=["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    )
    parser.add_argument(
        "--norm-type",
        default="masked_mean",
        choices=["masked_mean", "masked_normalize"],
    )
    parser.add_argument("--use-std-normalization", action="store_true", default=True)
    parser.add_argument(
        "--no-std-normalization",
        dest="use_std_normalization",
        action="store_false",
    )
    parser.add_argument("--cliprange", type=float, default=0.2)
    parser.add_argument(
        "--normalize-constant",
        type=float,
        default=None,
        help="Used only when norm-type=masked_normalize. Defaults to sampling-max-tokens.",
    )

    # Optimization
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)

    # logging
    parser.add_argument("--eval-interval", type=int, default=10)
    parser.add_argument("--max-val-examples", type=int, default=200)
    parser.add_argument("--log-interval", type=int, default=5)

    # wandb
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--wandb-name", type=str, default=None)
    parser.add_argument("--wandb-entity", type=str, default=None)

    args = parser.parse_args()
    print("\n======== starting training ========")
    print(args)
    grpo_train_loop(args)


if __name__ == "__main__":
    main()