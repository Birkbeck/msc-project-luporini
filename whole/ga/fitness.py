import torch
from torch import nn
from torch.utils.data import DataLoader


def model_fitness(data: DataLoader, problem="AE"):
    """
    a high-order function that returns a function, which computes average fitness value
    according to the chosen task when a model is given.
    
    In regression and AE problems, the returned function computes avg_fitness = -avg_loss
    where avg_loss is the loss across batches

    In classification, the returned function computes correct / total_images

    Args:
        data: image dataloader
        problem: either "regression", "classification" or default (AE).
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
    given a model pop and a fitness function, return list of fitnesses for each model
    """
    return [fn(i) for i in pop]


def normalise_fitness(fitnesses: list, bound: tuple):
    """
    given a fitness bound shaped (max_f, min_f)
    normalise fitnesses between 0 and 1
    normalised_f = (f - min)/(max - min) 
    """
    mino, maxo = bound[0], bound[1]
    deno = (maxo - mino + 1e-8)     # small constant to avoid zero division!
    normalised_fitnesses = [
        (f - mino) / deno
        for f in fitnesses
    ]
    return normalised_fitnesses