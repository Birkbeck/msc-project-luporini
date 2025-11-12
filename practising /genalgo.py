import torch
from torch import nn

def flatten(model):
    """
     flatten each parameter into a 1D tensor and concatenate
    """
    return torch.cat([param.view(-1) for param in model.parameters()])
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

mymodel = nn.Linear(3, 2, bias=True) # model, not a tensor!!!

print(list(param.shape for param in mymodel.parameters()))
flat = flatten(mymodel)
print(flat)
print(flat.shape)

remodelled = remodel(flat, mymodel)
print(mymodel == remodelled)