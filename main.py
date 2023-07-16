#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'


# In[ ]:


import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import spectral_norm
from torch.autograd import Variable
from torchvision import datasets, transforms, utils
# import seaborn as sns
import matplotlib.pyplot as plt
from dataTransformation import labels4clients, distribute_data_labels4clients, distribute_data_per_client_edited
from gan_model import Discriminator, Generator, initialize_weights
from network import Server, Worker
from torch.utils.data import Dataset, DataLoader
from utils import Logger
from fid_score import *
from inception import *
import math


# In[ ]:


seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
os.environ['PYTHONHASHSEED']=str(seed)
torch.use_deterministic_algorithms(True)
# torch.backends.cudnn.deterministic = True


# In[ ]:


if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
dev = torch.device(dev)


# In[ ]:


NUM_WORKERS = 2
CLASSES_PER_USER = 2
WORKER_OVERRIDE = False # overrides other workers weights with the chosen worker
NUM_EPOCHS = 200
BATCH_SIZE = 16

LEARNING_RATE_D = 2e-4
LEARNING_RATE_G = 2e-4
B1 = 0.5
B2 = 0.999

NOISE_DIM = 128
FID_BATCH_SIZE = 20
NUM_UNIQUE_USERS = NUM_WORKERS


# In[ ]:


limit_classes = True
Discriminator_trainig_iterations = 1
chosen_strategy = 4
trainig_strategies = {0:'fed_avg', 1:'weighted_avg_most',2:'weighted_avg_least', 3:'most_forgiving', 4:'least_forgiving'}
trainig_strategies = trainig_strategies[chosen_strategy]


# In[ ]:


trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
# trans_cifar = transforms.Compose([transforms.ToTensor()])
dataset = datasets.CIFAR10(root='./datasets/cifar/', train=True, download=True, transform=trans_cifar)
dataset_test = datasets.CIFAR10(root='./datasets/cifar/', train=False, download=True, transform=trans_cifar)
dataloader_one = torch.utils.data.DataLoader(dataset, shuffle = True,batch_size=BATCH_SIZE)
dataloader_test = torch.utils.data.DataLoader(dataset_test, shuffle = True,batch_size=10000)

MAX_WORKER_SAMPLE = len(dataset)/NUM_WORKERS


# In[ ]:


num_classes = 10


dictionary = labels4clients(num_classes,CLASSES_PER_USER,NUM_WORKERS,NUM_UNIQUE_USERS,random_seed=False)

if limit_classes:
    worker1_classes = []
    worker1_classes.append(dataset.class_to_idx['truck'])
    worker1_classes.append(dataset.class_to_idx['automobile'])
    print(worker1_classes)

    worker2_classes = []
    worker2_classes.append(dataset.class_to_idx['dog'])
    worker2_classes.append(dataset.class_to_idx['cat'])
    print(worker2_classes)

    dictionary = {}
    for i in worker1_classes:
        if i in dictionary:
            dictionary[i].append(0)
        else:
            dictionary[i]= [0]
    for i in worker2_classes:
        if i in dictionary:
            dictionary[i].append(1)
        else:
            dictionary[i]= [1]

print(dictionary)


# In[ ]:


for img in dataloader_test:
    test_imgs=img[0].to(dev)


# In[ ]:


# print(dataset.data[0])
# print(dataset.transforms(dataset.data[0],transforms.ToTensor()))
# print(dataset.transforms(dataset.data[0],trans_cifar))


# In[ ]:


transformed = trans_cifar(dataset.data[0]).cpu().detach().numpy()
print("transformed shape:", transformed.shape)
plt.figure('normalized data')
plt.hist(transformed.ravel(), bins=50, density=False)
plt.xlabel("pixel values")
plt.ylabel("frequency")
plt.show()


# In[ ]:


# print(dataset.data.shape)
# print(type(dataset))
# print(dataloader_one.dataset.data.shape)
# x,_ = dataloader_one.dataset[0]
# print(x.shape)
# print(x)


# In[ ]:


x_train_normalized_np = np.empty((dataset.data.shape[0],dataset.data.shape[3],dataset.data.shape[1],dataset.data.shape[2]))
print("train datatset shape:",x_train_normalized_np.shape)
for i in range(len(dataset)):
    x_train_normalized_np[i] = trans_cifar(dataset.data[i])


# In[ ]:


x_train_normalized_np[0][0].shape


# In[ ]:


plt.figure('normalized data')
bin_size = 60
plt.hist(x_train_normalized_np[:][0].ravel(),color='r', bins=bin_size, density=False)
plt.hist(x_train_normalized_np[:][1].ravel(),color='g', bins=bin_size, density=False)
plt.hist(x_train_normalized_np[:][2].ravel(),color='b', bins=bin_size, density=False)
plt.xlabel("pixel values")
plt.ylabel("frequency")
plt.show()


# In[ ]:


# x_train = np.asarray(dataset.data)
y_train = np.asarray(dataset.targets)
x_client_list, y_client_list = distribute_data_per_client_edited(x_train_normalized_np,y_train,dictionary,CLASSES_PER_USER,random_seed = False, max_samples_per_client = MAX_WORKER_SAMPLE)


# In[ ]:


for i, x in enumerate(x_client_list):
    print(f'worker#{i} data count: {len(x)}')


# In[ ]:


# def getDist(y,class_list,user_num):
#     ax = sns.countplot(x=y)
#     ax.set(title="Count of data classes for %s" %user_num)
#     plt.show()


# In[ ]:


# def getDist(y,class_list,user_num):
#     # ax = sns.barplot(x=class_list,y=y)
#     ax.set(title="Count of data classes for %s" %user_num)
#     plt.show()


# In[ ]:


total_data = 0
class_list = [i for i in range(num_classes)]
for i in range (len(x_client_list)):
    length = len(y_client_list[i])
    total_data+= length
    y_list = np.bincount(y_client_list[i],minlength=num_classes)
    # getDist(y_list,class_list,i)
print("total used data", total_data)


# In[ ]:


fic_model = InceptionV3().to(dev)


# In[ ]:


main_server = Server(0,LEARNING_RATE_G,B1,B2,dev)
# initialize_weights(main_server.generator)
# initialize_weights(main_server.global_disc)
main_server.generator.train()
main_server.global_disc.train()
workers = []
workers_weights= []
copy_workers = False
for i in range(NUM_WORKERS):
    worker = Worker(i,LEARNING_RATE_D,B1,B2,dev)
    # x_clinet_list[i] = np.transpose(x_clinet_list[i],(0, 3, 1, 2))
    if copy_workers:
        worker.load_worker_data(x_client_list[0], y_client_list[0])
    else:
        worker.load_worker_data(x_client_list[i], y_client_list[i]) 
    # initialize_weights(worker.discriminator)
    worker.discriminator.train()
    workers.append(worker)
    workers_weights.append(worker.discriminator.state_dict())
    
# summary(main_server.generator,(128,1,1))
# summary(workers[0].discriminator,(3,32,32))


# In[ ]:


# # code to make all the workers the same
# workers_weights= []
# for worker in workers:
#     worker.discriminator = workers[-1].discriminator
#     workers_weights.append(worker.discriminator.state_dict())


# In[ ]:


criterion = nn.MSELoss()
fixed_noise = torch.randn(36, NOISE_DIM, 1, 1).to(dev) # to use for generating output images

worker_loaders = []

for worker in workers:
    # print(worker.x_data.shape)
    worker_loaders.append([])
    for batch_id, real in enumerate(DataLoader(dataset=worker.x_data,batch_size=BATCH_SIZE)):
        worker_loaders[-1].append(real)


# In[ ]:


for worker in worker_loaders:
    plt.figure('normalized data')
    plt.hist(worker[:][1].ravel(),color='r', bins=bin_size, density=False)
    # plt.hist(x_train_normalized_np[:][1].ravel(),color='g', bins=bin_size, density=False)
    # plt.hist(x_train_normalized_np[:][2].ravel(),color='b', bins=bin_size, density=False)
    plt.xlabel("pixel values")
    plt.ylabel("frequency")
    plt.show()


# In[ ]:


logger = Logger(model_name='F2U',data_name='CIFAR10')
param_list = ["NUM_WORKERS", "CLASSES_PER_USER", "NUM_EPOCHS", "BATCH_SIZE",
"Discriminator_trainig_iterations","LEARNING_RATE_D", "LEARNING_RATE_G", "trainig_strategies", "WORKER_OVERRIDE","limit_classes"]
global_params = list(globals().items()) 
logger.log_params(param_list,global_params)


# In[ ]:


# main training loop for F2U (trial == FALSE)
start = 0
end = start + NUM_EPOCHS
end = NUM_EPOCHS

soft_max = nn.Softmax(dim=0)
worker_contribution_cum = torch.zeros(NUM_WORKERS)

for epoch in range(start,end):
    for batch_id in range(len(worker_loaders[0])):

        noise = torch.randn(BATCH_SIZE, NOISE_DIM, 1,1).to(dev)
        fake = main_server.generator(noise)
        
        for disc_train_iter in range(Discriminator_trainig_iterations):
            highest_loss = 0
            lowest_loss = math.inf
            chosen_discriminator = None
            worker_losses = []
            for worker_id, worker in enumerate(workers):
                current_worker_real = worker_loaders[worker_id][batch_id].float().to(dev)

                # print('worker ({}) datasum:'.format(worker_id),sum(current_worker_real.flatten()).item())
                # print(current_worker_real.shape)

                worker.d_optimizer.zero_grad()
            
                ### Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
                current_disc_real = worker.discriminator(current_worker_real).reshape(-1)
                worker.loss_disc_real = criterion(current_disc_real, torch.ones_like(current_disc_real))
                # print('real_classification:', round(sum(current_disc_real).item(),6),'real_loss:',round(worker.loss_disc_real.item(),6))
                current_disc_fake = worker.discriminator(fake.detach()).reshape(-1)
                worker.loss_disc_fake = criterion(current_disc_fake, torch.zeros_like(current_disc_fake))
                # print('fake_classification:', round(sum(current_disc_fake).item(),6),'fake_loss:',round(worker.loss_disc_fake.item(),6))
                worker.loss_disc = (worker.loss_disc_real + worker.loss_disc_fake) / 2
                
                worker.loss_disc.backward()
                worker.d_optimizer.step()

                workers_weights[worker_id] = worker.discriminator.state_dict()
                worker_losses.append(worker.loss_disc_fake.detach())
                # print(worker.loss_disc_fake, i)
                if highest_loss < worker.loss_disc_fake and trainig_strategies == 'most_forgiving':
                    highest_loss = worker.loss_disc_fake
                    chosen_discriminator = worker_id
                    chosen_worker = workers[chosen_discriminator]
                    
                    

                if lowest_loss > worker.loss_disc_fake and trainig_strategies == 'least_forgiving' :
                    lowest_loss = worker.loss_disc_fake
                    chosen_discriminator = worker_id
                    chosen_worker = workers[chosen_discriminator]
                   
            # print(f"chosen worker is {chosen_discriminator} with loss of: {highest_loss.item():.4f}")
        
        ### Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))
        if trainig_strategies in ['fed_avg', 'weighted_avg_most']:
            worker_losses = torch.FloatTensor(worker_losses)
            # print(f"worker_losses before: {worker_losses}")
            worker_losses_sm = soft_max(worker_losses)
            # print(worker_losses_sm)
            worker_contribution_cum += worker_losses_sm
        elif trainig_strategies == 'weighted_avg_least':
            # print(f"worker_losses before: {worker_losses}")
            worker_losses = -1*torch.FloatTensor(worker_losses)
            # print(f"worker_losses after: {worker_losses}")
            worker_losses_sm = soft_max(worker_losses)
            # print(f"worker_losses_sm: {worker_losses_sm}")
            worker_contribution_cum += worker_losses_sm

        # print(f"worker_losses after softmax: {worker_losses_sm}")

        # worker_total_weights1 = []
        # for worker in workers:
        #     weight_sum = 0
        #     for i, p in enumerate(worker.discriminator.parameters()):
        #         output = sum(p.detach().cpu().numpy().flatten())
        #         weight_sum += output
        #     worker_total_weights1.append(round(weight_sum,1))

        main_server.g_optimizer.zero_grad()

        if trainig_strategies == 'fed_avg':
            avg_w = main_server.fed_average(workers_weights)
            main_server.global_disc.load_state_dict(avg_w)
            for worker in workers:
                worker.discriminator.load_state_dict(avg_w)
            output = main_server.global_disc(fake).reshape(-1)
            chosen_worker = workers[0]
        elif trainig_strategies in ['weighted_avg_most','weighted_avg_least']:
            avg_w = main_server.weighted_fed_average(workers_weights,worker_losses_sm)
            main_server.global_disc.load_state_dict(avg_w)
            for worker in workers:
                worker.discriminator.load_state_dict(avg_w)
            output = main_server.global_disc(fake).reshape(-1)
            chosen_worker = workers[0]
        else:
            worker_contribution_cum[chosen_discriminator]+=1
            if WORKER_OVERRIDE:
                for worker in workers:
                    worker.discriminator.load_state_dict(chosen_worker.discriminator.state_dict())
            output = chosen_worker.discriminator(fake).reshape(-1)
        main_server.loss_gen = criterion(output, torch.ones_like(output))
        
        main_server.loss_gen.backward()
        # check weights of all workers before and after

        main_server.g_optimizer.step()

        # worker_total_weights2 = []
        # for worker in workers:
        #     weight_sum = 0
        #     for i, p in enumerate(worker.discriminator.parameters()):
        #         output = sum(p.detach().cpu().numpy().flatten())
        #         weight_sum += output
        #     worker_total_weights2.append(round(weight_sum,1))
        
        # diff = []
        # for i, curr_weight in enumerate(worker_total_weights2):
        #     diff.append(abs(curr_weight-worker_total_weights1[i]))
        # print("before g_optimizer:", worker_total_weights1, "after g_optimizer:", worker_total_weights2, "diff:", diff)

        with torch.no_grad():
            logger.log_workers(workers,epoch,batch_id,len(worker_loaders[0]))
            logger.log(chosen_worker.loss_disc.item(),main_server.loss_gen.item(),chosen_worker.loss_disc_real, chosen_worker.loss_disc_fake,epoch,batch_id,len(worker_loaders[0]))
        # Print loss
        if batch_id % 100 == 0:
            fid_z = torch.randn(FID_BATCH_SIZE, NOISE_DIM, 1,1).to(dev)
            gen_imgs = main_server.generator(fid_z.detach())
            mu_gen, sigma_gen = calculate_activation_statistics(gen_imgs, fic_model, batch_size=FID_BATCH_SIZE,cuda=True,verbose=False)
            mu_test, sigma_test = calculate_activation_statistics(test_imgs[:FID_BATCH_SIZE], fic_model, batch_size=FID_BATCH_SIZE,cuda=True,verbose=False)
            fid = calculate_frechet_distance(mu_gen, sigma_gen, mu_test, sigma_test)
            logger.log_fid(fid,epoch,batch_id,len(worker_loaders[0]))

            print(
                f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {batch_id}/{len(worker_loaders[0])} \
                Loss D: {chosen_worker.loss_disc:.4f}, loss G: {main_server.loss_gen:.4f}, FID Score: {fid:.1f}"
            )
        
    with torch.no_grad():
        main_server.generator.eval()
        fake = main_server.generator(fixed_noise)
        main_server.generator.train()
        logger.log_images(fake,len(fake), epoch, batch_id, len(worker_loaders[0]))

    chosen_w_np = worker_contribution_cum.detach().cpu().numpy()
    ra = range(1,len(chosen_w_np)+1)
    plt.bar(ra,chosen_w_np)
    plt.xticks(ra)
    plt.xlabel('worker number')
    plt.ylabel('contribution')
    plt.savefig('{}/worker_cont.png'.format(logger.writer.logdir))
    plt.show()

    if (epoch+1) % 50 == 0 and epoch != 0:
        logger.save_models(main_server,workers,epoch)


# In[ ]:


# plt.bar(range(len(worker_chosen_counter)),worker_chosen_counter)
# plt.xlabel('worker number')
# plt.ylabel('chosen counter')
# plt.show()


# In[ ]:


# for testing the total weights
# worker_total_weights1 = []
# for worker in workers:
#     weight_sum = 0
#     for i, p in enumerate(worker.discriminator.parameters()):
#         output = sum(p.detach().cpu().numpy().flatten())
#         weight_sum += output
#     worker_total_weights1.append(round(weight_sum,1))

# worker_total_weights2 = []
# for worker in workers:
#     weight_sum = 0
#     for i, p in enumerate(worker.discriminator.parameters()):
#         output = sum(p.detach().cpu().numpy().flatten())
#         weight_sum += output
#     worker_total_weights2.append(round(weight_sum,1))

# diff = []
# for i, curr_weight in enumerate(worker_total_weights2):
#     diff.append(abs(curr_weight-worker_total_weights1[i]))
# print("before g_optimizer:", worker_total_weights1, "after g_optimizer:", worker_total_weights2, "diff:", diff)


# In[ ]:


# # GAN archicture trial (trial == TRUE)
# if trial:
#     start = 0
#     end = start + NUM_EPOCHS
#     for epoch in range(start,end):
#         for i, data in enumerate(dataloader_one):
#             worker = workers[0]
#             noise = torch.randn(BATCH_SIZE, NOISE_DIM, 1,1).to(dev)
#             fake = main_server.generator(noise)
#             real, _ = data

#             ### Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
            
#             current_disc_real = worker.discriminator(real).reshape(-1)
#             # print('current discriminator real output', current_disc_real)
#             worker.loss_disc_real = criterion(current_disc_real, torch.ones_like(current_disc_real))
#             # print('worker loss_disc_real output', current_disc_real)
#             current_disc_fake = worker.discriminator(fake.detach()).reshape(-1)
#             worker.loss_disc_fake = criterion(current_disc_fake, torch.zeros_like(current_disc_fake))
#             worker.loss_disc = (worker.loss_disc_real + worker.loss_disc_fake) / 2
#             worker.discriminator.zero_grad()
#             worker.loss_disc.backward()
#             # total_norm_d =0
#             # for p in list(filter(lambda p: p.grad is not None, worker.discriminator.parameters())):
#             #     total_norm_d += p.grad.detach().data.norm(2).item()** 2
#             # total_norm_d = total_norm_d ** 0.5

#             worker.d_optimizer.step()

#             ### Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))
            
#             output = worker.discriminator(fake).reshape(-1)
#             main_server.loss_gen = criterion(output, torch.ones_like(output))
#             main_server.generator.zero_grad()
#             main_server.loss_gen.backward()

#             # total_norm_g =0
#             # for p in list(filter(lambda p: p.grad is not None, main_server.generator.parameters())):
#             #     total_norm_g += p.grad.detach().data.norm(2).item()** 2
#             # total_norm_g = total_norm_g ** 0.5

#             main_server.g_optimizer.step()


#             logger.log(worker.loss_disc.item(),main_server.loss_gen.item(),worker.loss_disc_real, worker.loss_disc_fake,epoch,i,len(dataloader_one))

#             # Print loss
#             if i % 100 == 0:    
#                 fid_z = torch.randn(FID_BATCH_SIZE, NOISE_DIM, 1,1).to(dev)
#                 gen_imgs = main_server.generator(fid_z.detach())
#                 mu_gen, sigma_gen = calculate_activation_statistics(gen_imgs, fic_model, batch_size=FID_BATCH_SIZE,cuda=True)
#                 mu_test, sigma_test = calculate_activation_statistics(test_imgs[:FID_BATCH_SIZE], fic_model, batch_size=FID_BATCH_SIZE,cuda=True)
#                 fid = calculate_frechet_distance(mu_gen, sigma_gen, mu_test, sigma_test)
#                 logger.log_fid(fid,epoch,i,len(dataloader_one))

#                 print(
#                     f"Epoch [{epoch}/{end}] Batch {i}/{len(dataloader_one)} \
#                     Loss D: {worker.loss_disc:.4f}, loss G: {main_server.loss_gen:.4f}, FID Score: {fid:.1f}"
#                 )

#             if i% 500 == 0:
#                 with torch.no_grad():
#                     fake = main_server.generator(fixed_noise)
#                     logger.log_images(fake,len(fake), epoch, i, len(dataloader_one))
#         if epoch % 50 == 0 and epoch !=0:
#             logger.save_models(main_server,workers,epoch)

