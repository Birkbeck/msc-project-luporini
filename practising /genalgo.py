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


def mutate(guy:torch.Tensor, m_chance=0.05, mode="small", m_rate=0.3) -> torch.Tensor:
    """
    randomly mutate parameters in a 1D tensor

    args:
        m_chance: chance of mutation in "small" mode - mutatate if rand below m_chance
        mode: either "50/50", where half of the genes mutate on avg. or else, where #mutation depends on m_chance
        m_rate: scaling factor for mutation strength
    """
    if mode == "50/50":
        mask = torch.randint_like(guy, 2) # 0-1s mask.. which params are mutated
        strength = torch.randn_like(guy) # raw mutation effect
        # mutation = mask * noise #?? effect too great on 1s, needs scaling
        mutation = m_rate * mask * strength
        #⛔️half of the genes mutated on avg!!! AGGRESSIVE?
    else:
        mask = torch.rand_like(guy) < m_chance
        strength = torch.randn_like(guy)
        mutation = m_rate * mask * strength
    return guy + mutation




mymodel = nn.Linear(3, 2, bias=True) # model, not a tensor!!!
