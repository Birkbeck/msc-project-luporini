import torch


def mutate(guy:torch.Tensor, m_rate=0.2, m_strength=0.3, mode="small") -> torch.Tensor:
    """
    randomly mutate parameters in a 1D tensor

    args:
        guy: flat tensor
        m_chance: chance of mutation in "small" mode - mutatate if rand below m_chance
        mode: either "50/50", where half of the genes mutate on avg. or else, where #mutation depends on m_chance
        m_rate: scaling factor for mutation strength
    """
    device = guy.device
    if mode == "50/50":
        mask = torch.randint_like(guy, 2, device=device) # 0-1s mask.. which params are mutated
        # mutation = mask * noise #?? effect too great on 1s, needs scaling
        #⛔️half of the genes mutated on avg!!! AGGRESSIVE?
    else:
        mask = torch.rand_like(guy, device=device) < m_rate
    
    noise = torch.randn_like(guy, device=device)
    mutation = m_strength * mask * noise
    
    return guy + mutation


def crossover(parent1, parent2, mode="uniform"):
    """
    uniform crossover between two flat 1D tensors by default. If type != "uniform", then one-point.
    """
    device = parent1.device
    if mode == "uniform":
        mask = torch.randint_like(parent1, 2, device=device).bool()
        child1 = torch.where(mask, parent1, parent2)
        child2 = torch.where(mask, parent2, parent1)
    else:
        rip = torch.randint(0, parent1.numel()-1, size=(1,)).item() #for fun..
        child1 = torch.cat([parent1[:rip], parent2[rip:]])
        child2 = torch.cat([parent2[:rip], parent1[rip:]])
    return child1, child2