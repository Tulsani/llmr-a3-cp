import torch


def tokenize_prompt_and_output(prompt_strs, output_strs, tokenizer) -> dict[str, torch.Tensor]:
    # tokenize prompt and output separately without special tokens
    tokenized_prompt = [
        tokenizer.encode(prompt, add_special_tokens=False)
        for prompt in prompt_strs
    ]
    tokenized_output = [
        tokenizer.encode(output, add_special_tokens=False)
        for output in output_strs
    ]

    # concatenate prompt + output
    input_ids_list = []
    prompt_lengths = []
    output_lengths = []

    for prompt_ids, output_ids in zip(tokenized_prompt, tokenized_output):
        combined = prompt_ids + output_ids
        input_ids_list.append(combined)
        prompt_lengths.append(len(prompt_ids))
        output_lengths.append(len(output_ids))

    # pad full concatenated sequence first
    max_len = max(len(ids) for ids in input_ids_list)
    pad_id = tokenizer.eos_token_id
    if pad_id is None:
        pad_id = 0

    padded_input_ids = torch.full(
        (len(input_ids_list), max_len),
        pad_id,
        dtype=torch.long
    )

    for i, ids in enumerate(input_ids_list):
        padded_input_ids[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)

    # shift for causal LM
    input_ids = padded_input_ids[:, :-1]
    labels = padded_input_ids[:, 1:]

    # response mask should align with labels
    seq_len = max_len - 1
    response_mask = torch.zeros(len(input_ids_list), seq_len, dtype=torch.long)

    for i, (prompt_len, output_len) in enumerate(zip(prompt_lengths, output_lengths)):
        response_mask[i, prompt_len - 1 : prompt_len + output_len - 1] = 1

    return {
        "input_ids": input_ids,
        "labels": labels,
        "response_mask": response_mask,
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
    