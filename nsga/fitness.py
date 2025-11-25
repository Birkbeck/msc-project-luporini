import time
import torch
from torch import nn
from torch.utils.data import DataLoader


# def model_fitness(data: DataLoader, problem="AE"):
#     """
#     returns a fitness function that computes 1/avg_loss = avg_fitness
#     across batches given a model

#     ⛔️ using non_blocking + pin_memory (DataLoader at the start of evolve):
#     https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html

#     Args:
#         problem: either "regression", "classification" or default (AE).
#     """
#     if problem == "regression":
#         loss_fn = nn.MSELoss()
#         out = lambda X, y: y
#     elif problem == "classification":
#         loss_fn = nn.CrossEntropyLoss() # expect logits, y.shape((batch,))
#         out = lambda X, y: y        # if working with encodings, will break!
#     else:
#         loss_fn = nn.MSELoss()
#         out = lambda X, y: X
    
#     def fitness(model):
#         device = next(model.parameters()).device
#         model.eval()
#         with torch.no_grad():
#             tot_loss = 0
#             for X, y in data:
#                 X = X.to(device, non_blocking=True)
#                 y = out(X, y).to(device, non_blocking=True)
#                 pred = model(X)
#                 loss = loss_fn(pred, y)
#                 tot_loss += loss.item()   # ⛔️SPIKING FITNESS if avg_loss very small
#             avg_loss = tot_loss / len(data) # enumerate starts from 0
#             # avg_fitness = 1 / (avg_loss + 1e-8) # if avg_loss–>0, avg_fit–>inf!!
#             avg_fitness = -avg_loss
#         return avg_fitness
    
#     return fitness

def model_fitness(data: DataLoader, problem="AE"):
    """
    returns model accuracy if problem == "classification" or
    a fitness function that computes 1/avg_loss = avg_fitness 
    across batches given a model if "AE or regression"

    ⛔️ using non_blocking + pin_memory (DataLoader at the start of evolve):
    https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html

    Args:
        problem: either "regression", "classification" or default (AE).
    """
    if problem == "regression":
        loss_fn = nn.MSELoss()
        out = lambda X, y: y
    elif problem == "classification":
        loss_fn = nn.CrossEntropyLoss() # expect logits, y.shape((batch,))
        out = lambda X, y: y        # if working with encodings, will break!
    else:
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

def model_runtime(data: DataLoader):
    """
    careful if using on GPU.. operations are asynchronous
    - torch.cuda.synchronise() ⁉️
    - synch for every time.time() ⁉️
    (https://discuss.pytorch.org/t/bizzare-extra-time-consumption-in-pytorch-gpu-1-1-0-1-2-0/87843)

    ALSO

    - just evaluate on representative batch ⁉️
    - return len(data)/interval ⁉️
    """
    def speed(model):
        device = next(model.parameters()).device
        model.eval()
        with torch.no_grad():
            ##########################
            # wait for all GPU kernels to be done !!!
            if device.type == "cuda":
                torch.cuda.synchronize() 
            ##########################

            start = time.time()
            for X, _ in data:
                X = X.to(device)
                _ = model(X)

            ##########################
            # wait for all GPU kernels to be done !!!
            if device.type == "cuda":
                torch.cuda.synchronize() 
            ##########################
            
            finish = time.time()
            # interval = (finish - start) / len(data)
            interval = finish - start # just raw time, not avg. (would be too quick?)
        return 1/interval
    
    return speed


def normalise_objective(fitnesses: list, bound):
    """
    normalises fitnesses between 0 and 1
    normalised_f = (f - min)/(max - min) 
    """
    mino, maxo = bound
    normalised_fitnesses = [
        (f - mino) / (maxo - mino + 1e-8)
        for f in fitnesses
    ]
    return normalised_fitnesses


def group_fitness(pop:list, fn, bound:tuple)->list:
    """
    given a model pop and a fitness function, return fitness for each model
    - fitnesses are clamped between the given bound (empirical bounds!!!)
    """
    if bound is None:
        return [fn(i) for i in pop] 
    
    mino, maxo = bound
    return [  # min(fit, maxo) + max(fit, mino) # use generator inside!!
        max(min(fit, maxo), mino) for fit in (fn(i) for i in pop)
    ]