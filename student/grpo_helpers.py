import torch


def compute_group_normalized_rewards(reward_fn,
                                     rollout_responses,
                                     repeated_ground_truths,
                                     group_size,
                                     advantage_eps,
                                     normalize_by_std):
    raw_rewards = []
    for response, ground_truth in zip(rollout_responses, repeated_ground_truths):
        reward_dict = reward_fn(response, ground_truth)
        raw_rewards.append(reward_dict["reward"])

    assert len(raw_rewards) % group_size == 0, "raw rewards not divisible by group size"

    rollout_batch_size = len(raw_rewards)
    n_groups = rollout_batch_size // group_size

    raw_rewards_tensor = torch.tensor(raw_rewards, dtype=torch.float32)
    grouped = raw_rewards_tensor.reshape(n_groups, group_size)

    group_mean = grouped.mean(dim=-1)

    if normalize_by_std:
        std_across_group = torch.std(grouped, dim=-1)
        advantage_across_group = (grouped - group_mean.unsqueeze(-1)) / (
            std_across_group.unsqueeze(-1) + advantage_eps
        )
    else:
        advantage_across_group = grouped - group_mean.unsqueeze(-1)

    advantages = advantage_across_group.reshape(rollout_batch_size)

    metadata = {
        "mean_reward": raw_rewards_tensor.mean().item(),
        "std_reward":  raw_rewards_tensor.std().item(),
        "max_reward":  raw_rewards_tensor.max().item(),
        "min_reward":  raw_rewards_tensor.min().item(),
    }

    return advantages, raw_rewards_tensor, metadata


def compute_naive_policy_gradient_loss(raw_rewards_or_advantages,
                                       policy_log_probs):
    return -raw_rewards_or_advantages * policy_log_probs


def compute_grpo_clip_loss(advantages,
                           policy_log_probs,
                           old_log_probs,
                           cliprange):
    log_ratio = policy_log_probs - old_log_probs
    ratio = torch.exp(log_ratio)

    unclipped = ratio * advantages
    clipped   = torch.clamp(ratio, 1 - cliprange, 1 + cliprange) * advantages
    loss      = -torch.min(unclipped, clipped)

    is_clipped    = (clipped < unclipped).float()
    clip_fraction = is_clipped.mean()

    metadata = {
        "clip_fraction":  clip_fraction,
        "mean_ratio":     ratio.mean(),
        "mean_log_ratio": log_ratio.mean(),
    }
    return loss, metadata


def compute_policy_gradient_loss(policy_log_probs,
                                 loss_type,
                                 raw_rewards=None,
                                 advanatages=None,
                                 old_log_probs=None,
                                 cliprange=None):
    metadata = {}
    if loss_type == "no_baseline":
        loss = compute_naive_policy_gradient_loss(
            raw_rewards_or_advantages=raw_rewards,
            policy_log_probs=policy_log_probs,
        )
    elif loss_type == "reinforce_with_baseline":
        loss = compute_naive_policy_gradient_loss(
            raw_rewards_or_advantages=advanatages,
            policy_log_probs=policy_log_probs,
        )
    elif loss_type == "grpo_clip":
        loss, metadata = compute_grpo_clip_loss(
            advantages=advanatages,
            policy_log_probs=policy_log_probs,
            old_log_probs=old_log_probs,
            cliprange=cliprange,
        )
    return loss, metadata


def mask_mean(tensor, mask, dim=None):
    masked_tensor = tensor * mask
    if dim is None:
        return masked_tensor.sum() / mask.sum()
    else:
        return masked_tensor.sum(dim=dim) / mask.sum(dim=dim)


def masked_normalize(tensor, mask, normalize_constant, dim=None):
    masked_tensor = tensor * mask
    if dim is None:
        return masked_tensor.sum() / normalize_constant
    else:
        return masked_tensor.sum(dim=dim) / normalize_constant


def grpo_microbatch_train_step(policy_log_probs,
                                response_mask,
                                gradient_accumulation_steps,
                                loss_type,
                                raw_rewards=None,
                                advantages=None,
                                old_log_probs=None,
                                cliprange=None,
                                normalize_constant=None):
    per_token_loss, metadata = compute_policy_gradient_loss(
        policy_log_probs=policy_log_probs,
        loss_type=loss_type,
        raw_rewards=raw_rewards,
        advanatages=advantages,
        old_log_probs=old_log_probs,
        cliprange=cliprange,
    )

    if normalize_constant is None:
        per_example_loss = mask_mean(per_token_loss, response_mask, dim=1)
    else:
        per_example_loss = masked_normalize(per_token_loss, response_mask,
                                            normalize_constant=normalize_constant,
                                            dim=1)

    loss        = per_example_loss.mean()
    loss_scaled = loss / gradient_accumulation_steps
    loss_scaled.backward()

    metadata["loss"]               = loss.item()
    metadata["num_response_tokens"] = response_mask.sum().item()

    return loss_scaled, metadata