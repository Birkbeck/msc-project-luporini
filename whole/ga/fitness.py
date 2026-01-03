import torch
from torch import nn
from torch.utils.data import DataLoader


def model_fitness(data: DataLoader, problem="AE"):
    """
    High-order function that returns a task-specific fitness function as fitness(model).

    Given a PyTorch DataLoader and problem type, this function returns a callable 'fitness(model)'.
    The callable evaluates the model passed to it on the data of the DataLoader, 
    and it returns a scalar average fitness score across batches.

    Fitness definition by problem: 
    
    - Regression / autoencoder (AE) problems: fitness = -avg_loss

    - Classification: fitness = correct / total_images

    Args:
        data: image DataLoader with (X, y) batches
        problem: either "regression", "classification" or default ("AE").
    
    Returns:
        callable(nn.Module): function computes fitness for given model
    """
    if problem == "regression":
        loss_fn = nn.MSELoss()
        out = lambda X, y: y
    elif problem == "classification":
        loss_fn = nn.CrossEntropyLoss() # expect logits, y.shape((batch,))
        out = lambda X, y: y      # if working with encodings, will break!
    elif problem == "AE":
        loss_fn = nn.MSELoss()
        out = lambda X, y: X
    
    def fitness(model):
        device = next(model.parameters()).device
        model.eval()
        with torch.no_grad():
            tot_loss = 0
            if problem == "classification":
                correct = 0
                tot = 0
            for X, y in data:
                X = X.to(device, non_blocking=True)
                y = out(X, y).to(device, non_blocking=True)
                
                pred = model(X)
                loss = loss_fn(pred, y)
                tot_loss += loss.item()   # ⛔️SPIKING FITNESS if avg_loss very small
                
                if problem == "classification":
                    y_pred_class = torch.argmax(pred, dim=1)
                    correct += (y_pred_class == y).sum().item()
                    tot += y.size(0)
            
            if problem == "classification":
                return correct / tot
        
            avg_loss = tot_loss / len(data) # enumerate starts from 0
            # avg_fitness = 1 / (avg_loss + 1e-8) # if avg_loss–>0, avg_fit–>inf!!
            avg_fitness = -avg_loss
        
        return avg_fitness
    
    return fitness


def group_fitness(pop:list, fn):
    """
    Get fitness for population of models.

    Args:
        pop: list of models
        fn: function that computes model fitness given a model
    
    Returns:
        list: population fitness values
    """
    return [fn(i) for i in pop]


def normalise_fitness(fitnesses: list, bound: tuple):
    """
    Normalise fitnesses within the range [0, 1].

    Formula: normalised_f = (f - min_f)/(max_f - min_f)

    Args:
        fitnesses: list of fitness values
        bound: tuple (min_f, max_f)
    
    Returns:
        list: Normalised fitness values
    """
    mino, maxo = bound[0], bound[1]
    deno = (maxo - mino + 1e-8)     # small constant to avoid zero division!
    normalised_fitnesses = [
        (f - mino) / deno
        for f in fitnesses
    ]
    return normalised_fitnesses