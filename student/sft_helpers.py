import torch


def tokenize_prompt_and_output(prompt_strs, output_strs, tokenizer) -> dict[str, torch.Tensor]:
    # tokenize WITHOUT adding special tokens automatically
    tokenized_prompt = tokenizer(prompt_strs, add_special_tokens=False)["input_ids"]
    tokenized_output = tokenizer(output_strs, add_special_tokens=False)["input_ids"]

    eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    input_ids_list = []
    response_mask_list = []

    for prompt_ids, output_ids in zip(tokenized_prompt, tokenized_output):
        # append EOS once at the very end
        combined = prompt_ids + output_ids + [eos_id]
        input_ids_list.append(combined)

        # response mask: 0 for prompt, 1 for output tokens + EOS
        mask = [0] * len(prompt_ids) + [1] * (len(output_ids) + 1)
        response_mask_list.append(mask)

    max_len = max(len(ids) for ids in input_ids_list)

    padded_input_ids = []
    padded_masks = []

    for ids, mask in zip(input_ids_list, response_mask_list):
        pad_len = max_len - len(ids)
        padded_input_ids.append(ids + [pad_id] * pad_len)
        padded_masks.append(mask + [0] * pad_len)

    input_ids_tensor = torch.tensor(padded_input_ids, dtype=torch.long)
    mask_tensor = torch.tensor(padded_masks, dtype=torch.long)

    return {
        "input_ids": input_ids_tensor[:, :-1],
        "labels": input_ids_tensor[:, 1:],
        "response_mask": mask_tensor[:, 1:],
    }

def compute_entropy(logits) -> torch.Tensor:
    # log_softmax
    log_probs = torch.nn.functional.log_softmax(logits,dim=-1) 
    # probs
    probs = torch.exp(log_probs)

    # entropy
    entropy = -(probs * log_probs).sum(dim=-1)

    return entropy

def get_response_log_probs(model,input_ids,labels,return_token_entropy) -> dict[str,torch.Tensor]:
    logits = model(input_ids).logits # (batch_size, seq_len, vocab_size)
    # log softmax
    log_probs_all = torch.nn.functional.log_softmax(logits,dim=-1)
    # selecting log probs of the true token label using gather
    log_probs = log_probs_all.gather(dim=-1, #along the vocab_size
                                     index = labels.unsqueeze(-1) # select the indexes using the labels => add a dimension on labels for projection
                                    ).squeeze(-1) # remove the extra element
    results = {"log_probs":log_probs}
    
    
    # entrop calc if check
    if return_token_entropy:
        results["token_entropy"] = compute_entropy(logits=logits)
    return results

def masked_normalize(tensor,mask,normalize_constant,dim=None) -> torch.Tensor:
    masked_tensor = tensor*mask
    
    if dim is None:
        results = masked_tensor.sum()
    else:
        results = masked_tensor.sum(dim=dim)
    
    return results/normalize_constant

def sft_microbatch_train_step(policy_log_probs,response_mask,gradient_accumulation_steps,normalize_constant=1.0):
    
    batch_size = policy_log_probs.shape[0]

    # get the mask log prob 
    masked_logprob_sum = masked_normalize(policy_log_probs,response_mask,normalize_constant=normalize_constant)

    #loss nll
    loss = -masked_logprob_sum / batch_size
    
    #gradient accumulation
    loss_scaled = loss / gradient_accumulation_steps
    
    loss_scaled.backward()

    # added metadata
    metadata = {
        "num_response_tokens": response_mask.sum().item(),
        "loss": loss.item()
    }

    #
    return loss_scaled, metadata
    