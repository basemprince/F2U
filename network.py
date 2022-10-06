import torch
from gan_model import Discriminator, Generator
import copy
if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
dev = torch.device(dev)

class Server:
    def __init__(self,id,lr,b1,b2):
        self.id = id
        self.global_disc = Discriminator().to(dev)
        self.generator = Generator().to(dev)
        self.g_optimizer = torch.optim.Adam(self.generator.parameters(),lr=lr,betas=(b1, b2))
        self.loss_gen = None

    def f2u_discriminator(self):
        pass
    def f2a_discriminator(self):
        pass
    def fed_average(self,w):
        w_avg = copy.deepcopy(w[0])
        for k in w_avg.keys():
            tmp = torch.zeros_like(w[0][k], dtype = torch.float32).to(dev)
            for i in range(len(w)):
                tmp += w[i][k]
            tmp = torch.true_divide(tmp, len(w))
            w_avg[k].copy_(tmp)
        return w_avg  

class Worker:
    def __init__(self,id,lr,b1,b2):
        self.id = id
        self.x_data = []
        self.y_data = []
        self.discriminator = Discriminator().to(dev)
        self.d_optimizer = torch.optim.Adam(self.discriminator.parameters(),lr=lr,betas=(b1, b2))
        self.loss_disc_real = None
        self.loss_disc_fake = None
        self.loss_disc = None
    def load_worker_data(self,x,y):
        self.x_data = x
        self.y_data = y
    

if __name__ == "__main__":
    server = Server(0)
    clinet = Worker(0)
    print(server.id)