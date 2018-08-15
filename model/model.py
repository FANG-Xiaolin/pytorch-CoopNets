import os
import time

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import math

# from model.utils.data_io import *
from model.utils.data_io import DataSet,saveSampleResults

class Descriptor(nn.Module):
    def __init__(self):
        super(Descriptor,self).__init__()
        self.conv1=nn.Conv2d(3,64,kernel_size=5,stride=2,padding=1)
        self.conv2=nn.Conv2d(64,128,kernel_size=3,stride=2,padding=1)
        self.conv3=nn.Conv2d(128,256,kernel_size=3,stride=1,padding=1)
        self.fc=nn.Linear(16*16*256,100)
        self.leakyrelu=nn.LeakyReLU()

    def forward(self,x):
        self.x=x
        out=self.conv1(x)
        out=self.leakyrelu(out)
        out=self.conv2(out)
        out=self.leakyrelu(out)
        out=self.conv3(out)
        out=self.leakyrelu(out)
        out=out.view(out.size(0),-1)
        out=self.fc(out)
        return out

class Generator(nn.Module):
    def __init__(self):
        super(Generator,self).__init__()
        self.convt1=nn.ConvTranspose2d(100,512,kernel_size=4,stride=1,padding=0)
        self.convt2=nn.ConvTranspose2d(512,256,kernel_size=5,stride=2,padding=2,output_padding=1)
        self.convt3=nn.ConvTranspose2d(256,128,kernel_size=5,stride=2,padding=2,output_padding=1)
        self.convt4=nn.ConvTranspose2d(128,64,kernel_size=5,stride=2,padding=2,output_padding=1)
        self.convt5=nn.ConvTranspose2d(64,3,kernel_size=5,stride=2,padding=2,output_padding=1)
        self.bn1=nn.BatchNorm2d(512)
        self.bn2=nn.BatchNorm2d(256)
        self.bn3=nn.BatchNorm2d(128)
        self.bn4=nn.BatchNorm2d(64)
        self.leakyrelu=nn.LeakyReLU()
        self.tanh=nn.Tanh()

    def forward(self,z):
        self.z=z
        out=self.convt1(z)
        out=self.bn1(out)
        out=self.leakyrelu(out)
        out=self.convt2(out)
        out=self.bn2(out)
        out=self.leakyrelu(out)
        out=self.convt3(out)
        out=self.bn3(out)
        out=self.leakyrelu(out)
        out=self.convt4(out)
        out=self.bn4(out)
        out=self.leakyrelu(out)
        out=self.convt5(out)
        out=self.tanh(out)
        return out

class Descriptor_cifar(nn.Module):
    def __init__(self):
        super(Descriptor_cifar,self).__init__()
#        self.conv1=nn.Conv2d(3,64,kernel_size=5,stride=2,padding=1)
#        self.conv2=nn.Conv2d(64,128,kernel_size=3,stride=2,padding=1)
#        self.conv3=nn.Conv2d(128,256,kernel_size=3,stride=1,padding=1)
#        self.fc=nn.Linear(8*8*256,100)
#        self.leakyrelu=nn.LeakyReLU()
       
        self.conv1=nn.Conv2d(3,64,kernel_size=5,stride=2,padding=2)
        self.conv2=nn.Conv2d(64,128,kernel_size=5,stride=2,padding=2)
        self.conv3=nn.Conv2d(128,256,kernel_size=5,stride=2,padding=2)
        self.conv4=nn.Conv2d(256,512,kernel_size=5,stride=2,padding=2)
        self.fc=nn.Linear(2*2*512,100)
        self.leakyrelu=nn.LeakyReLU()


    def forward(self,x):
        self.x=x
        out=self.conv1(x)
        out=self.leakyrelu(out)
        out=self.conv2(out)
        out=self.leakyrelu(out)
        out=self.conv3(out)
        out=self.leakyrelu(out)

        out=self.conv4(out)
        out=self.leakyrelu(out)

        out=out.view(out.size(0),-1)
        out=self.fc(out)
        return out

class Generator_cifar(nn.Module):
    def __init__(self):
        super(Generator_cifar,self).__init__()
#        self.convt1=nn.ConvTranspose2d(100,256,kernel_size=4,stride=1,padding=0)
#        self.convt2=nn.ConvTranspose2d(256,128,kernel_size=5,stride=2,padding=2,output_padding=1)
#        self.convt3=nn.ConvTranspose2d(128,64,kernel_size=5,stride=2,padding=2,output_padding=1)
#        self.convt4=nn.ConvTranspose2d(64,3,kernel_size=5,stride=2,padding=2,output_padding=1)
#        self.bn1=nn.BatchNorm2d(256)
#        self.bn2=nn.BatchNorm2d(128)
#        self.bn3=nn.BatchNorm2d(64)
#        self.leakyrelu=nn.LeakyReLU()
#        self.tanh=nn.Tanh()

        self.fc=nn.Linear(100,2048)
        self.convt1=nn.ConvTranspose2d(512,256,kernel_size=5,stride=2,padding=2,output_padding=1)
        self.convt2=nn.ConvTranspose2d(256,128,kernel_size=5,stride=2,padding=2,output_padding=1)
        self.convt3=nn.ConvTranspose2d(128,64,kernel_size=5,stride=2,padding=2,output_padding=1)
        self.convt4=nn.ConvTranspose2d(64,3,kernel_size=5,stride=2,padding=2,output_padding=1)
        self.bn1=nn.BatchNorm2d(256)
        self.bn2=nn.BatchNorm2d(128)
        self.bn3=nn.BatchNorm2d(64)
        self.leakyrelu=nn.LeakyReLU()
        self.tanh=nn.Tanh()

    def forward(self,z):
        self.z=z

        z=z.view(-1,100)        
        out=self.fc(z)
        out=out.reshape(-1,512,2,2)

        out=self.convt1(out)
        out=self.bn1(out)
        out=self.leakyrelu(out)
        out=self.convt2(out)
        out=self.bn2(out)
        out=self.leakyrelu(out)
        out=self.convt3(out)
        out=self.bn3(out)
        out=self.leakyrelu(out)
        out=self.convt4(out)
        out=self.tanh(out)
        return out


class CoopNets(nn.Module):
    def __init__(self,opts):
        super(CoopNets,self).__init__()
        self.img_size=opts.img_size
        self.num_chain=opts.nRow*opts.nCol
        self.opts=opts
        if opts.with_noise==True:
            print ('Do Langevin with noise')
        if opts.set == 'cifar':
            opts.img_size = 32
            print ('train on cifar. img_size: {:d}'.format(opts.img_size))

    def langevin_dynamics_generator(self,z):
        criterian=nn.MSELoss()
        for i in range(self.opts.langevin_step_num_gen):
            noise=torch.randn(size=(self.num_chain,self.opts.z_size)).cuda()
            z=Variable(z,requires_grad=True)
            gen_res=self.generator(z)
            gen_loss=1.0/(2.0* self.opts.sigma_gen*self.opts.sigma_gen)*criterian(gen_res,z)
            gen_loss.backward()
            # nn.utils.clip_grad_norm_(self.generator.parameters(), 0.5)
            grad=self.generator.z.grad 
            z=z-0.5*self.opts.langevin_step_size_gen*self.opts.langevin_step_size_gen*(z+grad)
            if self.opts.with_noise == True:
               z+=self.opts.langevin_step_size_gen*noise

        return z

    def langevin_dynamics_descriptor(self,x):
        for i in range(self.opts.langevin_step_num_des):
            noise=Variable(torch.randn(self.num_chain,3,self.opts.img_size,self.opts.img_size).cuda())
            #clone it and turn x into a leaf variable so the grad won't be thrown away
            x=Variable(x.data,requires_grad=True)
            x_feature=self.descriptor(x)
            x_feature.backward(torch.ones(144,100).cuda())
            grad=x.grad
            x=x-0.5*self.opts.langevin_step_size_des*self.opts.langevin_step_size_des*\
                (x/self.opts.sigma_des/self.opts.sigma_des-grad)
            if self.opts.with_noise:
                x+=self.opts.langevin_step_size_des*noise
        return x

    def train(self):
        if self.opts.ckpt_des!=None and self.opts.ckpt_des!='None':
            self.descriptor=torch.load(self.opts.ckpt_des)
            print ('Loading Descriptor from '+self.opts.ckpt_des+'...')
        else:
            if self.opts.set == 'scene':
                self.descriptor=Descriptor().cuda()
                print ('Loading Descriptor without initialization...')
            elif self.opts.set=='cifar':
                self.descriptor=Descriptor_cifar().cuda()
                print ('Loading Descriptor_cifar without initialization...')
            else:
                raise NotImplementedError('The set should be either scene or cifar currently')

        if self.opts.ckpt_gen!=None and self.opts.ckpt_gen!='None':
            self.generator=torch.load(self.opts.ckpt_gen)
            print ('Loading Generator from '+ self.opts.ckpt_gen+'...')
        else:
            if self.opts.set=='scene':
                self.generator=Generator().cuda()
                print ('Loading Generator without initialization...')
            elif self.opts.set=='cifar':
                self.generator=Generator_cifar().cuda()
                print ('Loading Generator_cifar without initialization...')
            else:
                raise NotImplementedError('The set should be either scene or cifar currently')



        batch_size=self.opts.batch_size
        train_data=DataSet(os.path.join(self.opts.data_path,self.opts.category),image_size=self.opts.img_size)
        num_batches=int(math.ceil(len(train_data)/batch_size))

        # sample_results = np.random.randn(self.num_chain * num_batches, self.opts.img_size, self.opts.img_size, 3)
        des_optimizer=torch.optim.Adam(self.descriptor.parameters(),lr=self.opts.lr_des,betas=[self.opts.beta1_des,0.999])
        gen_optimizer=torch.optim.Adam(self.generator.parameters(), lr=self.opts.lr_gen,betas=[self.opts.beta1_gen,0.999])

        if not os.path.exists(self.opts.ckpt_dir):
            os.makedirs(self.opts.ckpt_dir)
        if not os.path.exists(self.opts.output_dir):
            os.makedirs(self.opts.output_dir)
        logfile=open(self.opts.ckpt_dir+'/log','w+')


        for epoch in range(self.opts.num_epoch):
            start_time=time.time()
            gen_loss_epoch,des_loss_epoch,recon_loss_epoch=[],[],[]
            for i in range(num_batches):
                if (i+1)*batch_size>len(train_data):
                    continue
                obs_data=train_data[i*batch_size:min((i+1)*batch_size,len(train_data))]
                obs_data=Variable(torch.Tensor(obs_data).cuda(),requires_grad=True)

                #G0
                z=torch.randn(self.num_chain,self.opts.z_size)
                z=Variable(z.cuda(),requires_grad=True)
                #NCHW
                z=z.view(-1,self.opts.z_size,1,1)
                gen_res=self.generator(z)

                #D1
                if self.opts.langevin_step_num_des>0:
                    revised=self.langevin_dynamics_descriptor(gen_res)
                #G1
                if self.opts.langevin_step_num_gen>0:
                    z=self.langevin_dynamics_generator(z)

                #D2
                obs_feature=self.descriptor(obs_data)
                revised_feature=self.descriptor(revised)

                des_loss=(revised_feature.mean(0)-obs_feature.mean(0)).sum()

                des_optimizer.zero_grad()
                des_loss.backward()
                des_optimizer.step()


                #G2
                if self.opts.langevin_step_num_gen>0:
                    gen_res=self.generator(z)
                # gen_res=gen_res.detach()
                gen_loss=0.5*self.opts.sigma_gen*self.opts.sigma_gen*((revised-gen_res)**2).sum()

                gen_optimizer.zero_grad()
                gen_loss.backward()
                gen_optimizer.step()

                #Compute reconstruction loss
                recon_loss=((revised-gen_res)**2).mean()

                gen_loss_epoch.append(gen_loss.cpu().data)
                des_loss_epoch.append(des_loss.cpu().data)
                recon_loss_epoch.append(recon_loss.cpu().data)

                # sample_results[i * batch_size:min((i + 1) * batch_size, len(train_data))] = syn


            # saveSampleResults(obs_data.cpu().data, "%s/observed.png", col_num=self.opts.nCol)
            saveSampleResults(revised.cpu().data, "%s/des_%03d.png" % (self.opts.output_dir, epoch+1), col_num=self.opts.nCol)
            saveSampleResults(gen_res.cpu().data, "%s/gen_%03d.png" % (self.opts.output_dir, epoch+1), col_num=self.opts.nCol)

            end_time = time.time()
            print('Epoch #{:d}/{:d}, des_loss: {:.4f}, gen_loss: {:.4f}, recon_loss: {:.4f}, '
                  'time: {:.2f}s'.format(epoch+1,self.opts.num_epoch, np.mean(des_loss_epoch), np.mean(gen_loss_epoch), np.mean(recon_loss_epoch),
                                         end_time - start_time))


            #python 3
            print ('Epoch #{:d}/{:d}, des_loss: {:.4f}, gen_loss: {:.4f}, recon_loss: {:.4f}, '
                    'time: {:.2f}s'.format(epoch,self.opts.num_epoch, np.mean(des_loss_epoch), np.mean(gen_loss_epoch), np.mean(recon_loss_epoch),
                                           end_time - start_time),file=logfile)
            #python 2.7
            #print >> logfile, ('Epoch #{:d}/{:d}, des_loss: {:.4f}, gen_loss: {:.4f}, recon_loss: {:.4f}, '
             #    'time: {:.2f}s'.format(epoch,self.opts.num_epoch, np.mean(des_loss_epoch), np.mean(gen_loss_epoch), np.mean(recon_loss_epoch),
              #                          end_time - start_time))


            if epoch%self.opts.log_epoch==0:
                torch.save(self.descriptor, self.opts.ckpt_dir+'/des_ckpt_{}.pth'.format(epoch))
                torch.save(self.generator, self.opts.ckpt_dir+'/gen_ckpt_{}.pth'.format(epoch))
        logfile.close()


    # def test(self):





