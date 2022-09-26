import torch
from gan_model import Discriminator, Generator
if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
dev = torch.device(dev)

class Server:
    def __init__(self,id,lr):
        self.id = id
        self.generator = Generator().to(dev)
        self.g_optimizer = torch.optim.Adam(self.generator.parameters(),lr=lr)
    def f2u_discriminator(self):
        pass
    

class Worker:
    def __init__(self,id,lr):
        self.id = id
        self.x_data = []
        self.y_data = []
        self.discriminator = Discriminator().to(dev)
        self.d_optimizer = torch.optim.Adam(self.discriminator.parameters(),lr=lr)
        self.loss_disc_real = None
        self.loss_disc_fake = None
    def load_worker_data(self,x,y):
        self.x_data = x
        self.y_data = y
    def train(self):
        pass
    

if __name__ == "__main__":
    server = Server(0)
    clinet = Worker(0)
    print(server.id)