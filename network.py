import torch
from gan_model import Discriminator, Generator
import copy


class Server:
    def __init__(self,id,lr=0.002,b1=0.9,b2=0.999,dev='cuda:0',load_saved=True):
        self.id = id
        self.dev = dev
        self.global_disc = Discriminator().to(self.dev)
        self.generator = Generator().to(self.dev)
        if load_saved:
            self.generator.load_state_dict(torch.load('init_models/G_init'))
        self.g_optimizer = torch.optim.Adam(self.generator.parameters(),lr=lr,betas=(b1, b2))
        self.loss_gen = None

    def f2u_discriminator(self):
        pass
    
    def weighted_fed_average(self,w,contribution):
        w_avg = copy.deepcopy(w[0])
        for k in w_avg.keys():
            tmp = torch.zeros_like(w[0][k],dtype= torch.float32).to(self.dev)
            for i in range(len(w)):
                tmp+=w[i][k]*contribution[i]
            w_avg[k].copy_(tmp)
        return w_avg

    def fed_average(self,w):
        w_avg = copy.deepcopy(w[0])
        for k in w_avg.keys():
            tmp = torch.zeros_like(w[0][k], dtype = torch.float32).to(self.dev)
            for i in range(len(w)):
                tmp += w[i][k]
            tmp = torch.true_divide(tmp, len(w))
            w_avg[k].copy_(tmp)
        return w_avg  

class Worker:
    def __init__(self,id,lr=0.002,b1=0.9,b2=0.999,dev='cuda:0',load_saved=True):
        self.id = id
        self.dev = dev
        self.x_data = []
        self.y_data = []
        self.discriminator = Discriminator().to(self.dev)
        if load_saved:
            self.discriminator.load_state_dict(torch.load('init_models/D_init'))
        self.d_optimizer = torch.optim.Adam(self.discriminator.parameters(),lr=lr,betas=(b1, b2))
        self.loss_disc_real = None
        self.loss_disc_fake = None
        self.loss_disc = None
    def load_worker_data(self,x,y):
        self.x_data = x
        self.y_data = y
    

if __name__ == "__main__":
    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"
    dev = torch.device(dev)

    server = Server(0)
    clinet = Worker(0)
    print(server.id)