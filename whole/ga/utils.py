import torch


def flatten(model):
    """
     flatten each parameter into a 1D tensor and concatenate
    """
    device = next(model.parameters()).device
    return torch.cat([param.view(-1) for param in model.parameters()]).to(device)
    # return torch.cat([param.view(-1) for param in model.parameters()])
    # return torch.cat([param.data.view(-1) for param in model.parameters()]) # NO!!!!
                            # BREAKS COMPUTATION GRAPH.. don't risk it
                            # might need autograd later


def remodel(flat, model):
    """
    remap a flat model into a non flat torch.architecture
    """
    index = 0
    for param in model.parameters():
        with torch.no_grad():
            n = param.numel()
            param.copy_(flat[index:index+n].view_as(param))
            index += n
    return model