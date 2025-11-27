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
        device=torch.device("cuda") if torch.cuda.is_available() else "cpu"
):
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
                optimiser.zero_grad()
                X_noisy = X + noise * torch.randn_like(X)
                
                pred = auto(X_noisy)
                loss = loss_fn(pred, X)
                loss.backward()
                optimiser.step()

        pop.append(auto)
    
    return pop


# def plot_performance(pop, data, images:int, noise):
#     """
#     plotting a grid where the top row is original images,
#     the second row is noisy inputs, and following rows are AE reconstructions
#     """
#     subset = Subset(data, indices=range(images))
#     loader = DataLoader(subset, batch_size=images)
    
#     X, _ = next(iter(loader))
#     X = X.to(next(pop[0].parameters()).device)
#     noisy = X + noise * torch.randn_like(X)

#     predictions = []
#     for model in pop:
#         model.eval()
#         with torch.no_grad():
#             pred = model(noisy)
#         predictions.append(pred.cpu())

#     size = len(pop)
#     tot_rows = size + 2                                 # w x h
#     _, axes = plt.subplots(nrows=tot_rows, ncols=images, figsize=(images * 3, tot_rows * 3))

    # filling the top row
    for i in range(images):
        axes[0, i].imshow(X[i].cpu().squeeze(), cmap="gray")
        # axes[0, i].imshow(X[i].cpu().permute(1, 2, 0), cmap="gray")
    
    # filling the second row
    for i in range(images):
        axes[1, i].imshow(noisy[i].cpu().squeeze(), cmap="gray")
    
    # filling the following rows
    for i in range(images):
        for j in range(len(predictions)):
            axes[j + 2, i].imshow(predictions[j][i].cpu().squeeze(), cmap="gray")
    
    # better visualisation?
    for row in axes:
        for ax in row:
            ax.axis("off")
    
    plt.show()


# from torchvision.transforms import ToTensor
# from torchvision.datasets import MNIST

# train= MNIST("./datasets", download=True, train=True, transform=ToTensor())
# test = MNIST("./datasets", download=True, train=False, transform=ToTensor())

# train_loader = DataLoader(train, batch_size=30, shuffle=True)

# AEs = create_AE_pop(2, (1, 28, 28), 5, (1, 4), train_loader)
# plot_performance(AEs, test, 4, 0.4)



