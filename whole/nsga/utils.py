import math
import torch


def flatten(model):
    """
     Flatten nn.Module into a 1D tensor.

    Parameters are concatenated in the order they are returned by model.parameters().

    Args:
        model [torch.nn.Module]
    Returns:
        torch.Tensor: 1D tensor, flattened moder
    """
    device = next(model.parameters()).device
    return torch.cat([param.view(-1) for param in model.parameters()]).to(device)
    # return torch.cat([param.data.view(-1) for param in model.parameters()]) # NO!!!!
                            # BREAKS COMPUTATION GRAPH.. don't risk it
                            # might need autograd later

def embed(model, biggest, device):
    """
    Create genomes for populations of heterogenenous complexity.

    Embed a flattened model within a standard-length 1D tensor, to anable crossover.

    The size of the standard-length tensor is given by biggest (the biggest size model in pop).

    In practice, the flattened model is symmetrically padded on each size so that
    the final 1D tensor has length equal to biggest.
    """
    device = next(model.parameters()).device
    flat = flatten(model)
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
    """
    Rebuild a PyTorch model from a flat genome configuration.

    Args:
        embedded [torch.Tensor]: padded model tensor
        original_size [int]: original parameter count, to know where to slice
        model [torch.nn.Module]: model template
        biggest [int]: dimensionality in biggest model in population (meh..could optimise..)
    
    Returns:
        torch.nn.Module: model!
    """
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
########################################################################
def convergence(p1, p2):
    """
    Given a point (p1, p2) in 2D space, compute Euclidean distance to (1, 1).
    """
    return math.sqrt((p1 - 1)**2 + (p2 - 1)**2)
        

def euclidean(point1:tuple, point2:tuple)->float:
    """
    Compute the Euclidean distance between points in 2D.

    Args:
        point1 [tuple]: (coord1, coord2)
        point2 [tuple]: (coord1, coord2)
        
        (coords are fitness values here!)
    """
    x1, y1 = point1
    x2, y2 = point2
    return math.sqrt((x1-x2)**2 + (y1-y2)**2)
########################################################################
