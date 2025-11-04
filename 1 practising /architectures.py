import torch
from torch import nn

class TiniestAutoEncoder(nn.Module):


    def __init__(self, input, latent, nonlinearity=nn.ReLU):
        super().__init__()
        self._input = input
        self._latent = latent
        self._nonl = nonlinearity

        self.encoder = nn.Sequential(
            nn.Linear(self._input, self._latent),
            self._nonl()
        )

        self.decoder = nn.Sequential(
            nn.Linear(self._latent, self._input),
            nn.Sigmoid()
        )

    def forward(self, data):
        output = self.encoder(data)
        output = self.decoder(output)
        return output
    