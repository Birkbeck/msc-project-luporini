import torch


def mutate(guy:torch.Tensor, m_rate=0.2, m_strength=0.3, mode="small") -> torch.Tensor:
    """
    Randomly mutate a 1D tensor by adding Gaussian noise to a subset of parameters.

    Modes:
    - "50/50": each parameters has 50% chance of mutation on avg. (aggressive)
    - "small": default, each parameter has m_rate chance of mutation on avg. (recommended)

    args:
        guy [torch.Tensor: flat tensor
        m_rate [float]: chance of mutation in "small" mode - mutatate if rand below m_chance
        m_strength [float]: scaling factor for Gaussian noise magnitude
        mode [str]: either "50/50", where half of the genes mutate on avg. or else, where #mutation depends on m_chance
    """
    device = guy.device
    if mode == "50/50":
        # 0-1 mask with 50% chance (aggressive mutation)
        mask = torch.randint_like(guy, 2, device=device)
    else:
        # 0-1 mask where 1 only where p < m_rate
        mask = torch.rand_like(guy, device=device) < m_rate
    
    noise = torch.randn_like(guy, device=device)
    mutation = m_strength * mask * noise
    
    return guy + mutation


def crossover(parent1: torch.Tensor, parent2: torch.Tensor, mode="uniform"):
    """
    Recombine two 1D tensors (crossover between two flat genomes).
    
    Modes:
    - "uniform": unform crossover (implemented)
    - "anything": one-point crossover 

    Args:
        parent1 [torch.Tensor]: first parent, a 1D tensor
        parent2 [torch.Tensor]: sencond parent, a 1D tensor
        mode [str]: crossover strategy
    
    Returns:
        tuple [torch.Tensor, torch.Tensor]: two offspring
    """
    device = parent1.device
    if mode == "uniform":
        # random mask with 50% change of recombination per gene
        mask = torch.randint_like(parent1, 2, device=device).bool()
        child1 = torch.where(mask, parent1, parent2)
        child2 = torch.where(mask, parent2, parent1)
    else:
        # split and recombine genomes at random index
        rip = torch.randint(0, parent1.numel()-1, size=(1,)).item() #for fun..
        child1 = torch.cat([parent1[:rip], parent2[rip:]])
        child2 = torch.cat([parent2[:rip], parent1[rip:]])
    return child1, child2