import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import make_grid
from torchvision.utils import save_image
from modules import Discriminator, Generator
import os
import time
import datetime
import numpy as np

class Solver(object):
    def __init__(self, config, dataloader):
        self.dataloader = dataloader
        self.data_size = config.data_size
        # self.iters = config.iters
        self.loss_type = config.loss_type
        self.G_lr = config.G_lr
        self.D_lr = config.D_lr
        self.beta1 = config.momentum
        self.batch_size = config.batch_size
        self.max_epoch = config.max_epoch
        self.z_dim = config.z_dim
        self.lr_update_step = config.lr_update_step
        self.lr_decay_after = config.lr_decay_after
        self.lr_decay = config.lr_decay
        # path
        self.sample_path = os.path.join(config.main_path, 'samples')
        self.ckpt_path = os.path.join(config.main_path, 'checkpoints')
        # misc
        self.log_step = config.log_step
        self.eval_step = config.eval_step
        self.save_step = config.save_step

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.build_model()

    def build_model(self):
        self.G = Generator()
        self.D = Discriminator()

        self.G_optim = optim.Adam(self.G.parameters(), self.G_lr, (self.beta1, 0.999))
        self.D_optim = optim.Adam(self.D.parameters(), self.D_lr, (self.beta1, 0.999))

        if self.loss_type == 'BCEwL':
            self.criterion = nn.BCEWithLogitsLoss()
        elif self.loss_type == 'WGAN':
            pass
        elif self.loss_type == 'WGAN+':
            pass

        self.fixed_sample = None
        self.fixed_noise = None

        # self.true = torch.ones([self.batch_size, 1, 1, 1], requires_grad=False).to(self.device)
        # self.false = torch.zeros([self.batch_size, 1, 1, 1], requires_grad=False).to(self.device)

        # Change to GPU mode
        print('Change CPU mode to GPU mode...')
        self.G.to(self.device)
        self.D.to(self.device)
        print('Creating models are success...')

    def restore_model(self, resume_iters):
        print('Load the trained models from step {}...'.format(resume_iters))
        G_path = os.path.join(self.ckpt_path, '{}-G.ckpt'.format(resume_iters))
        D_path = os.path.join(self.ckpt_path, '{}-D.ckpt'.format(resume_iters))
        self.G.load_state_dict(torch.load(G_path))
        self.D.load_state_dict(torch.load(D_path))

    def train(self):
        iters = self.max_epoch * len(self.dataloader)
        data_iter = iter(self.dataloader)
        self.fixed_sample = next(data_iter)
        self.fixed_noise = torch.randn(self.batch_size, self.z_dim).to(self.device)
        num_data = 0
        start_time = time.time()
        print('Start training...')
        for i in range(iters):
            # try:
            #     sample = next(data_iter)
            # except:
            #     print('error occur')
            #     data_iter = iter(self.dataloader)
            #     sample = next(data_iter)
            sample = next(data_iter)
            if i % len(self.dataloader) == 0:
                data_iter = iter(self.dataloader)
            # Load data.
            right_embd = sample['right_embd'].to(self.device)
            wrong_embd = sample['wrong_embd'].to(self.device)
            z_noise = torch.randn(right_embd.size(0), self.z_dim).to(self.device)
            real_img = sample['real_img'].to(self.device)
            fake_img = self.G(right_embd, z_noise)
            # print('right_embd size: {}'.format(right_embd.size()))
            # print('wrong_embd size: {}'.format(wrong_embd.size()))
            # print('real_img size: {}'.format(real_img.size()))
            num_data += right_embd.size(0)
            T = torch.ones([right_embd.size(0), 1, 1, 1], requires_grad=False).to(self.device)
            F = torch.zeros([right_embd.size(0), 1, 1, 1], requires_grad=False).to(self.device)
            ## Train Discriminator.
            sr = self.D(real_img, right_embd) # {real image, right text}
            rr_loss = self.criterion(sr, T)
            sw = self.D(real_img, wrong_embd) # {real image, wrong text}
            rw_loss = self.criterion(sw, F)
            sf = self.D(fake_img.detach(), right_embd) # {fake image, right text}
            fr_loss = self.criterion(sf, F)
            d_loss = rr_loss + rw_loss + fr_loss
            ## Backward and optimize for D.
            self.D_optim.zero_grad()
            d_loss.backward()
            self.D_optim.step()
            # For logs
            loss = {}
            loss['D/rr_loss'] = rr_loss.item()
            loss['D/rw_loss'] = rw_loss.item()
            loss['D/fr_loss'] = fr_loss.item()
            loss['D/d_loss'] = d_loss.item()

            ## Train Generator.
            sf = self.D(fake_img, right_embd)
            g_loss = self.criterion(sf, T)
            ## Backward and optimize for G.
            self.G_optim.zero_grad()
            g_loss.backward()
            self.G_optim.step()
            loss['G/g_loss'] = g_loss.item()

            ## Print training information.
            if (i+1) % self.log_step == 0:
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                logs = "Elapsed [{}], Iter [{}/{}], Epoch [{}/{}]".format(et, i+1, iters, (i+1)/len(self.dataloader), self.max_epoch)
                logs += ", Dataset [{}/{}]".format(num_data%self.data_size, self.data_size)
                for tag, value in loss.items():
                    logs += ', {} [{:.4f}]'.format(tag, value)
                print(logs)
            ## Debug sample images.
            if (i+1) % self.eval_step == 0: #will be modified.
                with torch.no_grad():
                    image_path = os.path.join(self.sample_path, '{}.jpg'.format(i+1))
                    fake_img = self.G(self.fixed_sample['right_embd'].to(self.device), self.fixed_noise)#size: [B, 3, 64, 64]
                    real_img = self.fixed_sample['real_img']
                    img_list = []
                    for row in range(int(self.batch_size/8)): #print multiple of 8 samples
                        img_list += [real_img[row*8 + col] for col in range(8)]
                        img_list += [fake_img[row*8 + col].to('cpu') for col in range(8)]
                    sample_name = os.path.join(self.sample_path, '{}iter.jpg'.format(i+1))
                    save_image(make_grid(img_list), sample_name)
                print('Save generated sample results {}iter.jpg into {}...'.format(i+1, self.sample_path))
            ## Save model checkpoints.
            if (i+1) % self.save_step == 0:
                G_path = os.path.join(self.ckpt_path, '{}-G.ckpt'.format(i+1))
                D_path = os.path.join(self.ckpt_path, '{}-D.ckpt'.format(i+1))
                torch.save(self.G.state_dict(), G_path)
                torch.save(self.D.state_dict(), D_path)
                print('Save model checkpoints into {}...'.format(self.ckpt_path))
            ## Decay learning rates.
            if (i+1) % self.lr_update_step == 0:
                if (i+1) >= self.lr_decay_after:
                    self.G_lr = self.G_lr * self.lr_decay
                    self.D_lr = self.D_lr * self.lr_decay
                    for param_group in self.G_optim.param_groups:
                        param_group['lr'] = self.G_lr
                    for param_group in self.D_optim.param_groups:
                        param_group['lr'] = self.D_lr
                print('Decay learning rates, g_lr: {}, d_lr: {}...'.format(self.G_lr, self.D_lr))

    def test(self):
        pass


















