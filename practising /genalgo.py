import torch
from torch import nn

def flatten(model): # flatten each parameter into a 1D tensor and concatenate
    return torch.cat([param.view(-1) for param in model.parameters()])
    # return torch.cat([param.data.view(-1) for param in model.parameters()]) # NO!!!!
                            # BREAKS COMPUTATION GRAPH.. don't risk it
                            # might need autograd later

# model = nn.Linear(3, 2, bias=True) # model, not a tensor!!!

model = nn.Sequential(
    nn.Linear(2, 3),
    nn.Linear(3, 2)
)

# for param in model.parameters():
#     print(param.shape)    

flat = flatten(model)

print(flat.shape)
print(flat)
