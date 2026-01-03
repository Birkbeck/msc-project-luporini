import torch


def flatten(model:torch.nn.Module):
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
    # return torch.cat([param.view(-1) for param in model.parameters()])
    # return torch.cat([param.data.view(-1) for param in model.parameters()]) # NO!!!!
                            # BREAKS COMPUTATION GRAPH.. don't risk it
                            # might need autograd later


def remodel(flat:torch.Tensor, model:torch.nn.Model):
    """
    Logically, remap a flat model into a proper PyTorch architecture.

    In practice, update a model template with the parameters from flat.
    The flat tensor is sliced and gradually reshaped to march each parameter in the model.

    Args:
        flat [torch.Tensor]: 1D tensor, a flat model
        model [torch.nn.Model]: torch model template

    Returns:
        nn.Module: a model with the same parameters of flat
    """
    index = 0
    for param in model.parameters():
        with torch.no_grad():
            n = param.numel()
            param.copy_(flat[index:index+n].view_as(param))
            index += n
    return model