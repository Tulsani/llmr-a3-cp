from pathlib import Path

from datasets import load_dataset, load_from_disk
from tqdm import tqdm
from vllm import LLM, SamplingParams

from drgrpo_grader import question_only_reward_fn


def load_prompt(name: str = "intellect") -> str:
    path = Path(__file__).parent / "prompts" / f"{name}.prompt"
    return path.read_text()


def evaluate(llm, prompts, ground_truths, log_outputs=False):
    """Run evaluation and return accuracy."""
    params = SamplingParams(temperature=0.0, max_tokens=2048)
    outputs = llm.generate(prompts, params)

    correct = 0
    # Counters for the three categories
    cat1 = []  # format=1, answer=1
    cat2 = []  # format=1, answer=0
    cat3 = []  # format=0, answer=0

    for i, output in enumerate(tqdm(outputs, desc="Grading")):
        text = output.outputs[0].text
        reward = question_only_reward_fn(text, ground_truths[i])
        correct += reward["reward"]

        entry = {
            "prompt": prompts[i],
            "response": text,
            "ground_truth": ground_truths[i],
            "format_reward": reward["format_reward"],
            "answer_reward": reward["answer_reward"],
        }

        if reward["format_reward"] == 1 and reward["answer_reward"] == 1:
            cat1.append(entry)
        elif reward["format_reward"] == 1 and reward["answer_reward"] == 0:
            cat2.append(entry)
        else:
            cat3.append(entry)

    if log_outputs:
        print(f"\n{'='*60}")
        print(f"Category counts:")
        print(f"  (1) format=1, answer=1: {len(cat1)}")
        print(f"  (2) format=1, answer=0: {len(cat2)}")
        print(f"  (3) format=0, answer=0: {len(cat3)}")

        def print_examples(cat, label, n=10):
            print(f"\n--- {label} (showing {min(n, len(cat))}) ---")
            for ex in cat[:n]:
                print(f"  PROMPT (truncated): {ex['prompt'][:100]}")
                print(f"  RESPONSE (truncated): {ex['response'][:300]}")
                print(f"  GROUND TRUTH: {ex['ground_truth']}")
                print()

        print_examples(cat1, "Category 1: format=1, answer=1")
        print_examples(cat2, "Category 2: format=1, answer=0")
        print_examples(cat3, "Category 3: format=0, answer=0")

    return correct / len(outputs), cat1, cat2, cat3


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-Math-1.5B")
    parser.add_argument("--max-examples", type=int, default=500)
    parser.add_argument("--intellect-path", default="data/intellect_math_train_dev_test/test")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--log-outputs", action="store_true",
                        help="Print per-example outputs and category breakdowns")
    args = parser.parse_args()

    prompt_template = load_prompt("intellect")

    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    # Evaluate on MATH
    print("\n=== MATH Test ===")
    math_ds = load_dataset("hiyouga/math12k", split="test")
    if args.max_examples:
        math_ds = math_ds.select(range(min(args.max_examples, len(math_ds))))

    prompts = [prompt_template + "\n\n" + ex["problem"] for ex in math_ds]
    gts = [ex["answer"] for ex in math_ds]

    acc, cat1, cat2, cat3 = evaluate(llm, prompts, gts, log_outputs=args.log_outputs)
    print(f"\nMATH Accuracy: {acc:.4f}")
    print(f"Category (1) format=1, answer=1: {len(cat1)}")
    print(f"Category (2) format=1, answer=0: {len(cat2)}")
    print(f"Category (3) format=0, answer=0: {len(cat3)}")


if __name__ == "__main__":
    main()