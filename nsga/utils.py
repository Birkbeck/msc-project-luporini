import math
import torch


def flatten(model, device):
    """
     flatten each parameter into a 1D tensor and concatenate
    """
    device = next(model.parameters()).device
    return torch.cat([param.view(-1) for param in model.parameters()]).to(device)
    # return torch.cat([param.data.view(-1) for param in model.parameters()]) # NO!!!!
                            # BREAKS COMPUTATION GRAPH.. don't risk it
                            # might need autograd later

def embed(model, biggest, device):
    device = next(model.parameters()).device
    flat = flatten(model, device=device)
    mu = flat.mean().item()

    size = flat.numel()
    difference = biggest - size # 
    if difference % 2 == 0:
        lx_pad_size = difference // 2
        rx_pad_size = lx_pad_size
    else:
        lx_pad_size = difference // 2
        rx_pad_size = difference - lx_pad_size
    
    # ⛔️: every time embed() called, torch.random introduces randomness
    lx_padding = torch.full((lx_pad_size,), mu, dtype=flat.dtype, device=device)
    rx_padding = torch.full((rx_pad_size,), mu, dtype=flat.dtype, device=device)

    return (
        torch.cat([lx_padding, flat, rx_padding]), size, model
    ) # (flat, size, archi) (f, s, a)


def remodel(embedded, original_size, model, biggest):
    device = next(model.parameters()).device
    difference = biggest - original_size
    lx = difference // 2
    flat = embedded[lx:lx + original_size].to(device)

    index = 0
    for param in model.parameters():
        with torch.no_grad():
            n = param.numel()
            param.copy_(flat[index:index+n].view_as(param))
            index += n
    return model


# HELPER FUNCTIONS for evaluation convergence and spread⛔️
def convergence(p1, p2):
    """for a model: Euclidean distance from ideal s in nD"""
    return math.sqrt((p1 - 1)**2 + (p2 - 1)**2)
        

def euclidean(point1:tuple, point2:tuple)->float:
    """euclidean distance between points in 2D """
    x1, y1 = point1
    x2, y2 = point2
    return math.sqrt((x1-x2)**2 + (y1-y2)**2)