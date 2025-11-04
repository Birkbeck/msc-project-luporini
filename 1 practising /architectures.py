import torch
from torch import nn

class TinyAE(nn.Module):
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


class TinyConvAE(nn.Module):
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
            self._nonl() #using nn.ReLU might push values out of 0,1 interval??
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
    

class FlexyConvAE(nn.Module):
    """
    Based on TinyConvAE but adding stride arg

    input: image width/height
    channels: input channels (e.g., greyscale == 1)
    stride: stride value.. this is the same for each convolutional layer
    padding: padding pixels
    kernel: filter size
    nonlinearity: activation function after convLayers
    """
    def conv_output_size(input, pad, kernel, stride):
        return (input + 2*pad - kernel)//stride + 1
    
    def deconv_output_padding(input, output, pad, kernel,  stride):
        return output - (( - 1)*stride - 2*pad + kernel)

    def __init__(self, input=28, channels=1, stride=2, padding=1, kernel=3, nonlinearity=nn.ReLU):
        super().__init__()
        self._input = input
        self._channels = channels
        self._stride = stride
        self._pad = padding
        self._kernel = kernel
        self._nonl = nonlinearity

        self.encoder = nn.Sequential(
            nn.Conv2d(self._channels, 10, kernel_size=self._kernel, stride=self._stride, padding=self._pad),
            self._nonl(),
            nn.Conv2d(10, 20, kernel_size=self._kernel, stride=self._stride, padding=self._pad),
            self._nonl() #using nn.ReLU might push values out of 0,1 interval??
        )

        # # need to compute out_padding for deconvolution????
        # but PyTorch constrains out_pad < stride ⛔️
        # compute it using convolution and transposeConv formulas
        C1 = (self._input + 2*self._pad - self._kernel)//self._stride + 1
        C2 = (C1 + 2*self._pad - self._kernel)//self._stride + 1
        out_p1 = C1 - ((C2 - 1)*self._stride - 2*self._pad + self._kernel)
        out_p2 = self._input - ((C1 - 1)*self._stride - 2*self._pad + self._kernel)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(20, 10, kernel_size=self._kernel, stride=self._stride, padding=self._pad, output_padding=out_p1),
            self._nonl(),
            nn.ConvTranspose2d(10, self._channels, kernel_size=self._kernel, stride=self._stride, padding=self._pad, output_padding=out_p2),
            nn.Sigmoid()
        )

    def forward(self, data):
        output = self.encoder(data)
        output = self.decoder(output)
        return output