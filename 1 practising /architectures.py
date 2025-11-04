import torch
from torch import nn

class TinyAutoEncoder(nn.Module):
    """
    Super simple auto encoder architecture for flat in/out configurations.
    
    Args: self-explanatory✨
    """
    def __init__(self, input, latent, nonlinearity=nn.ReLU):
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


class TinyConvAutoEncoder(nn.Module):
    """
    Super simple convolutional auto encoder for image input/output.
    """
    def __init__(self, channels=1, nonlinearity=nn.ReLU):
        super().__init__()
        self._channels = channels
        self._nonl = nonlinearity

        self.encoder = nn.Sequential(
            nn.Conv2d(self._channels, 10, kernel_size=3, stride=2, padding=1),
            self._nonl(),
            nn.Conv2d(10, 20, kernel_size=3, stride=2, padding=1),
            self._nonl()
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(20, 10, kernel_size=3, stride=2, padding=1, output_padding=1),
            self._nonl(),
            nn.ConvTranspose2d(10, self._channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, data):
        output = self.encoder(data)
        output = self.decoder(output)
        return output