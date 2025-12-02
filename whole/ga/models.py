import random
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from matplotlib import pyplot as plt
    

class TinyFlexyConvAE(nn.Module):
    """
    Auto encode based on TinyConvAE but adding stride arg

    Args:
        input_shape: e.g., MNIST == (1, 28, 28); CIFAR10 == (3, 32, 32)
        stride: stride value.. the same for each convolutional layer
        padding: padding pixels
        kernel: filter size
        nonlinearity: activation function after convLayers
    """
    def __init__(
            self,
            input_shape=(1, 28, 28),
            stride=2,
            padding=1,
            kernel=3,
            nonlinearity=nn.ReLU
    ):
        super().__init__()
        self._input = input_shape[1]
        self._channels = input_shape[0]
        self._stride = stride
        self._pad = padding
        self._kernel = kernel
        self._nonl = nonlinearity

        self.encoder = nn.Sequential(
            nn.Conv2d(
                self._channels,
                20,
                kernel_size=self._kernel,
                stride=self._stride,
                padding=self._pad
            ),
            self._nonl()
        )

        # # need to compute out_padding for deconvolution????
        # but PyTorch constrains out_pad < stride ⛔️
        # compute it using convolution and transposeConv formulas
        C1 = (self._input + 2*self._pad - self._kernel)//self._stride + 1
        out_p1 = self._input - ((C1 - 1)*self._stride - 2*self._pad + self._kernel)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                20,
                self._channels,
                kernel_size=self._kernel,
                stride=self._stride,
                padding=self._pad,
                output_padding=out_p1
            ),
            nn.Sigmoid()
        )

    def get_stride(self):
        return self._stride
    
    def forward(self, data):
        output = self.encoder(data)
        output = self.decoder(output)
        return output

    

class TinyConvClassifier(nn.Module):
    def __init__(
            self,
            input_shape=(1, 28, 28),
            stride=2,
            padding=1,
            kernel=3,
            nonlinearity=nn.ReLU,
            classes=10
    ):
        super().__init__()
        self._channels = input_shape[0]
        self._input = input_shape[1]
        self._stride = stride
        self._pad = padding
        self._kernel = kernel
        self._nonl = nonlinearity
        self._classes = classes

        self.encoder = nn.Sequential(
            nn.Conv2d(
                self._channels,
                20,
                kernel_size=self._kernel,
                stride=self._stride,
                padding=self._pad
            ),
            self._nonl()
        )

        fake_data = torch.zeros(1, self._channels, self._input, self._input)
        fake_output = self.encoder(fake_data)
        fake_shape = fake_output.shape
        flat = fake_shape[1]*fake_shape[2]*fake_shape[3]
        self.classifier_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat, self._classes)
        )

    def get_stride(self):
        return self._stride
    
    def forward(self, data):
        output = self.encoder(data)
        output = self.classifier_head(output)
        return output
    

def create_AE_pop(
        model,
        size,
        shape,
        epochs,
        interval,
        data,
        noise=0.4,
        device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
):
    """
    ⛔️ could train on subset?! less compute?!
    """
    loss_fn = nn.MSELoss()

    pop = []
    for m in range(size):
        print(f"* {m} model")
        auto = model(
            input_shape=shape,
            stride=random.randint(interval[0], interval[1])
        ).to(device)
        
        optimiser = torch.optim.Adam(auto.parameters(), lr=0.01)
        auto.train()
        for e in range(epochs):
            print(f"  - {e} epoch")
            for X, _ in data:
                X = X.to(device)
                optimiser.zero_grad()
                X_noisy = torch.clamp(X + noise * torch.randn_like(X), 0, 1) # avoid going out 0, 1???
                
                pred = auto(X_noisy)
                loss = loss_fn(pred, X)
                loss.backward()
                optimiser.step()

        pop.append(auto)
    
    return pop
