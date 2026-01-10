import torch

def generate_causal_mask(sz):
    # Generates a square matrix of 'sz' where the upper triangle is -inf
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask