import os
import numpy as np
import errno
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
from IPython import display
from matplotlib import pyplot as plt
import torch

'''
    TensorBoard Data will be stored in './runs' path
'''


class Logger:

    def __init__(self, model_name, data_name):
        self.model_name = model_name
        self.data_name = data_name

        self.comment = '{}_{}'.format(model_name, data_name)
        self.data_subdir = '{}/{}'.format(model_name, data_name)

        # TensorBoard
        self.writer = SummaryWriter(comment=self.comment)
        self.full_name = '{}{}'.format(self.writer.logdir,self.comment)

    def log(self, d_error, g_error,prediction_real, prediction_fake, epoch, n_batch, num_batches):
       
        # var_class = torch.autograd.variable.Variable
        if isinstance(d_error, torch.autograd.Variable):
            d_error = d_error.data.cpu().numpy()
        if isinstance(g_error, torch.autograd.Variable):
            g_error = g_error.data.cpu().numpy()

        if isinstance(prediction_real, torch.autograd.Variable):
            prediction_real = prediction_real.data
        if isinstance(prediction_fake, torch.autograd.Variable):
            prediction_fake = prediction_fake.data


        step = Logger._step(epoch, n_batch, num_batches)
        self.writer.add_scalar(
            '{}/D_error'.format(self.comment), d_error, step)
        self.writer.add_scalar(
            '{}/G_error'.format(self.comment), g_error, step)
        self.writer.add_scalar(
            '{}/real_pred'.format(self.comment), prediction_real.mean(), step)
        self.writer.add_scalar(
            '{}/fake_pred'.format(self.comment), prediction_fake.mean(), step)


    def log_fid(self, fid_score, epoch, n_batch, num_batches):
       
        # var_class = torch.autograd.variable.Variable
        if isinstance(fid_score, torch.autograd.Variable):
            fid_score = fid_score.data.cpu().numpy()

        step = Logger._step(epoch, n_batch, num_batches)
        self.writer.add_scalar(
            '{}/fid_score'.format(self.comment), fid_score, step)

    def log_images(self, images, num_images, epoch, n_batch, num_batches,save_imgs=False, format='NCHW', normalize=True):
        '''
        input images are expected in format (NCHW)
        '''
        if type(images) == np.ndarray:
            images = torch.from_numpy(images)
        
        if format=='NHWC':
            images = images.transpose(1,3)
        

        step = Logger._step(epoch, n_batch, num_batches)
        img_name = '{}/images{}'.format(self.comment, '')

        # Make vertical grid from image tensor
        nrows = int(np.sqrt(num_images))
        grid = vutils.make_grid(
            images, nrow=nrows, normalize=True, scale_each=True)

        # Add horizontal images to tensorboard
        self.writer.add_image(img_name, grid, step)

        # Save plots
        self.save_torch_images(grid, epoch, n_batch)

    def save_torch_images(self, grid, epoch, n_batch, plot_=False):
       
        fig = plt.figure()
        plt.imshow(np.moveaxis(grid.cpu().numpy(), 0, -1))
        plt.axis('off')
        if plot_:
            display.display(plt.gcf())
        self._save_images(fig, epoch, n_batch)
        plt.close()

    def _save_images(self, fig, epoch, n_batch, comment=''):
        out_dir = '{}/images'.format(self.writer.logdir)
        Logger._make_dir(out_dir)
        fig.savefig('{}/{}_epoch_{}_batch_{}.png'.format(out_dir,
                                                         comment, epoch, n_batch), transparent=True)

    def display_status(self, epoch, num_epochs, n_batch, num_batches, d_error, g_error, d_pred_real, d_pred_fake):
        
        # var_class = torch.autograd.variable.Variable
        if isinstance(d_error, torch.autograd.Variable):
            d_error = d_error.data.cpu().numpy()
        if isinstance(g_error, torch.autograd.Variable):
            g_error = g_error.data.cpu().numpy()
        if isinstance(d_pred_real, torch.autograd.Variable):
            d_pred_real = d_pred_real.data
        if isinstance(d_pred_fake, torch.autograd.Variable):
            d_pred_fake = d_pred_fake.data
        
        
        print('Epoch: [{}/{}], Batch Num: [{}/{}]'.format(
            epoch,num_epochs, n_batch, num_batches)
             )
        print('Discriminator Loss: {:.4f}, Generator Loss: {:.4f}'.format(d_error, g_error))
        print('D(x): {:.4f}, D(G(z)): {:.4f}'.format(d_pred_real.mean(), d_pred_fake.mean()))

    def save_models(self, server, workers, epoch):
        out_dir = '{}/models'.format(self.writer.logdir)
        Logger._make_dir(out_dir)
        torch.save(server.generator.state_dict(),
                   '{}/G_epoch_{}'.format(out_dir, epoch))
        for i, worker in enumerate(workers):
            torch.save(worker.discriminator.state_dict(),
                   '{}/D{}_epoch_{}'.format(out_dir,i, epoch))

    def close(self):
        self.writer.close()

    # Private Functionality

    @staticmethod
    def _step(epoch, n_batch, num_batches):
        return epoch * num_batches + n_batch

    @staticmethod
    def _make_dir(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise