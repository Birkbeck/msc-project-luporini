from copy import deepcopy
import torch


def crossover(parent1: tuple, parent2: tuple)-> tuple[torch.Tensor, torch.Tensor]:
    """
    ALWAYS PERFORMED BY DESIGN
    masked crossover between two flat models:
    Args:
       parent1: flat model
       parent2: flat model
    """
    flat1, s, a = parent1
    flat2, _, _ = parent2
    
    device = flat1.device 
    mask = torch.randint(0, 2, flat1.shape, dtype=torch.bool, device=device) # mask with zeroes and ones
    child1 = (torch.where(mask, flat1, flat2), s, deepcopy(a))
    child2 = (torch.where(mask, flat2, flat1), s, deepcopy(a))

    return child1, child2


def mutate(guy:torch.Tensor, mode, m_rate=0.1, m_strength=0.3) -> torch.Tensor:
    """
    randomly mutate parameters in a 1D tensor

    args:
        guy: flat tensor
        mode: either "50/50", where half of the genes mutate on avg. or else, where #mutation depends on m_chance
        m_rate: chance of mutation in "small" mode - mutatate if rand below m_chance
        m_strength: scaling factor for mutation strength
    """
    device = guy.device
    if mode == "50/50":
        mask = torch.randint_like(guy, 2, device=device) # 0-1s mask.. which params are mutated
        strength = torch.randn_like(guy, device=device) # raw mutation effect
        # mutation = mask * noise #?? effect too great on 1s, needs scaling
        mutation = m_rate * mask * strength
        #⛔️half of the genes mutated on avg!!! AGGRESSIVE?
    else:
        mask = torch.rand_like(guy, device=device) < m_rate
        strength = torch.randn_like(guy, device=device)
        mutation = m_rate * mask * strength
    
    return guy + mutation
