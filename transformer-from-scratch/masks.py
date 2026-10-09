import torch

def padding_mask(tokens, pad_id=0):
    return (tokens != pad_id).unsqueeze(1).unsqueeze(1)

def causal_mask(length, device=None):
    return torch.ones(length, length, dtype=torch.bool, device=device).tril().unsqueeze(0).unsqueeze(0)

def decoder_mask(tokens, pad_id=0):
    return causal_mask(tokens.shape[1], tokens.device) & padding_mask(tokens, pad_id)
