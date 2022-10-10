import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import spectral_norm
from torchsummary import summary
import matplotlib.pyplot as plt

if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
dev = torch.device(dev)

class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=512, kernel_size=4, stride=1, padding=0, bias = False),
            nn.BatchNorm2d(num_features=512,momentum=0.1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1, bias = False),
            nn.BatchNorm2d(num_features=256,momentum=0.1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1, bias = False),
            nn.BatchNorm2d(num_features=128,momentum=0.1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1, bias = False),
            nn.BatchNorm2d(num_features=64,momentum=0.1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1, bias = False),
            nn.Tanh()
        )

    def forward(self, input):
        output = self.main(input)
        return output

class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()
        self.power_iters = 1
        self.main = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias = False),n_power_iterations=self.power_iters),
            nn.LeakyReLU(negative_slope= 0.1, inplace = True),

            spectral_norm(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1, bias = False),n_power_iterations=self.power_iters),
            nn.LeakyReLU(negative_slope= 0.1, inplace = True),

            spectral_norm(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias = False),n_power_iterations=self.power_iters),
            nn.LeakyReLU(negative_slope= 0.1, inplace = True),

            spectral_norm(nn.Conv2d(in_channels=128, out_channels=128, kernel_size=4, stride=2, padding=1, bias = False),n_power_iterations=self.power_iters),
            nn.LeakyReLU(negative_slope= 0.1, inplace = True),

            # need to calculate the number of neurons in this layer to connect each of their outputs to the next layer
            spectral_norm(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, bias = False),n_power_iterations=self.power_iters),
            nn.LeakyReLU(negative_slope= 0.1, inplace = True),
            nn.Flatten(), #flatten the output
            spectral_norm(nn.Linear(in_features =4096,out_features =1, bias = False))
            # nn.Sigmoid()
        )

    def neuron_calculator(in_channels,padding,kernel_size,stride,out_channels):
        return (in_channels+2*padding-kernel_size)**2 * out_channels
    def forward(self, input):
        output = self.main(input)
        return output.view(-1)


def initialize_weights(model):
    # Initializes weights
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, mean= 0.0, std=0.02)

if __name__ == "__main__":
    netG = Generator()
    netD = Discriminator()
    initialize_weights(netG)
    summary(netG,(128,1,1))
    summary(netD,(3,32,32))

    input_d = torch.randn((2000,3,32,32)).to(dev)
    input_g = torch.randn((2000,128,1,1)).to(dev)

    output_disc = netD(input_d)
    output_disc = output_disc.cpu().detach().numpy()

    plt.figure('discriminator')
    plt.hist(output_disc.ravel(), bins=50, density=False)
    plt.xlabel("output values")
    plt.ylabel("frequency")


    output_gen = netG(input_g)
    output_gen = output_gen.cpu().detach().numpy()
    plt.figure('generator')
    bin_size = 100
    plt.hist(output_gen[2].ravel(), bins=bin_size,color='b', density=False)
    plt.hist(output_gen[0].ravel(), bins=bin_size,fc='r', density=False)
    plt.hist(output_gen[1].ravel(), bins=bin_size,color='g', density=False)

    plt.xlabel("output values")
    plt.ylabel("frequency")

    plt.show()