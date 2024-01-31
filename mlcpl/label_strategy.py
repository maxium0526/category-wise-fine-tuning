import torch

def unknown_to_unknown(y):
    return y

def unknown_to_negative(y):
    return torch.nan_to_num(y, 0)

def unknown_to_positive(y):
    return torch.nan_to_num(y, 1)

def uncertain_to_negative(y):
    return torch.where(y==-1, 0, y)

def uncertain_to_positive(y):
    return torch.where(y==-1, 1, y)

def uncertain_to_lsr(y, low=0.0, up=1.0):
    return torch.where(y==-1, torch.rand(y.shape, device=y.device) * (up-low) + low, y)

def uncertain_to_lsr_negative(y):
    return uncertain_to_lsr(y, low=0.0, up=0.3)

def uncertain_to_lsr_positive(y):
    return uncertain_to_lsr(y, low=0.55, up=0.85)

def uncertain_to_unknown(y):
    return torch.where(y==-1, torch.nan, y)
