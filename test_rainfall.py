#encoding=utf8

from __future__ import print_function
import argparse
import os
import sys
import random
import time
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
cudnn.fastest = True
import torch.optim as optim
import torchvision.utils as vutils
from torch.autograd import Variable

# from misc import *
from misc_zx import getLoader, create_exp_dir, weights_init
# import models.derain_residual  as net2
# import models.derain_dense  as net1
import models.derain_dense
import models.derain_residual_zx

from myutils.vgg16 import Vgg16
# from myutils import utils
import pdb


def norm_ip(img, min, max):
    img.clamp_(min=min, max=max)
    img.add_(-min).div_(max - min)

    return img


def norm_range(t, range):
    if range is not None:
        norm_ip(t, range[0], range[1])
    else:
        norm_ip(t, t.min(), t.max())
    return norm_ip(t, t.min(), t.max())

def get_opt():
    # Pre-defined Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=False, default='pix2pix_val',  help='')
    parser.add_argument('--out_directory', default='./result_all/new_model_data/testing_our_our/')
    # parser.add_argument('--dataroot', required=False, default='', help='path to trn dataset')
    parser.add_argument('--valDataroot', required=False, default='', help='path to val dataset')
    # parser.add_argument('--mode', type=str, default='B2A', help='B2A: facade, A2B: edges2shoes')
    # parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
    parser.add_argument('--valBatchSize', type=int, default=1, help='input batch size')
    parser.add_argument('--originalSize', type=int, default=512, help='the height / width of the original input image')
    parser.add_argument('--imageSize', type=int, default=512, help='the height / width of the cropped input image to network')
    parser.add_argument('--inputChannelSize', type=int, default=3, help='size of the input channels')
    parser.add_argument('--outputChannelSize', type=int, default=3, help='size of the output channels')
    # parser.add_argument('--ngf', type=int, default=64)
    # parser.add_argument('--ndf', type=int, default=64)
    # parser.add_argument('--niter', type=int, default=400, help='number of epochs to train for')
    # parser.add_argument('--lrD', type=float, default=0.0002, help='learning rate, default=0.0002')
    # parser.add_argument('--lrG', type=float, default=0.0002, help='learning rate, default=0.0002')
    # parser.add_argument('--annealStart', type=int, default=0, help='annealing learning rate start to')
    # parser.add_argument('--annealEvery', type=int, default=400, help='epoch to reaching at learning rate of 0')
    # parser.add_argument('--lambdaGAN', type=float, default=0.01, help='lambdaGAN')
    # parser.add_argument('--lambdaIMG', type=float, default=1, help='lambdaIMG')
    # parser.add_argument('--poolSize', type=int, default=50, help='Buffer size for storing previously generated samples from G')
    # parser.add_argument('--wd', type=float, default=0.0000, help='weight decay in D')
    # parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam')
    parser.add_argument('--netG', default='./pre_trained/netG_epoch_9.pth', help="path to netG (to continue training)")
    # parser.add_argument('--netD', default='', help="path to netD (to continue training)")
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=1)
    # parser.add_argument('--exp', default='sample', help='folder to output images and model checkpoints')
    # parser.add_argument('--display', type=int, default=5, help='interval for displaying train-logs')
    # parser.add_argument('--evalIter', type=int, default=500, help='interval for evauating(generating) images from valDataroot')
    opt = parser.parse_args()
    return opt

opt = get_opt()
print(opt)

# create_exp_dir(opt.exp)
opt.manualSeed = random.randint(1, 10000)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
torch.cuda.manual_seed_all(opt.manualSeed)
print("Random Seed: ", opt.manualSeed)


# val_target = torch.FloatTensor(opt.valBatchSize, opt.outputChannelSize, opt.imageSize, opt.imageSize)
val_input = torch.FloatTensor(opt.valBatchSize, opt.inputChannelSize, opt.imageSize, opt.imageSize)
# val_depth = torch.FloatTensor(opt.valBatchSize, opt.inputChannelSize, opt.imageSize, opt.imageSize)
# val_ato = torch.FloatTensor(opt.valBatchSize, opt.inputChannelSize, opt.imageSize, opt.imageSize)
# val_target = val_target.cuda()
val_input = val_input.cuda()
# val_depth, val_ato = val_depth.cuda(), val_ato.cuda()
# label_d = torch.FloatTensor(opt.batchSize)
# label_d = Variable(label_d.cuda())

# Load pre-trained density-classification network
# net_label=net1.vgg19ca()
net_label = models.derain_dense.vgg19ca()
net_label.load_state_dict(torch.load('./classification/netG_epoch_9.pth'))
net_label=net_label.cuda()

# Load pre-trained residual-getting network
# residue_net=net2.Dense_rain_residual()
residue_net = models.derain_residual_zx.Dense_rain_residual()
residue_net.load_state_dict(torch.load('./residual_heavy/netG_epoch_6.pth'))
residue_net=residue_net.cuda()

# Load Pre-trained derain model
# netG=net1.Dense_rain()
netG = models.derain_dense.Dense_rain()
netG.apply(weights_init)
if opt.netG != '':
    # 加载模型 ./pre_trained/netG_epoch_9.pth
    netG.load_state_dict(torch.load(opt.netG))
print(netG)
# netG.train()
netG.cuda()
# get optimizer
# optimizerG = optim.Adam(netG.parameters(), lr = opt.lrG, betas = (opt.beta1, 0.999), weight_decay=0.00005)
# opt.dataset='pix2pix_val'

valDataloader = getLoader(opt.dataset,
                          opt.valDataroot,
                          opt.imageSize, #opt.originalSize,
                          opt.imageSize,
                          opt.valBatchSize,
                          opt.workers,
                          mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
                          split='val',
                          shuffle=False,
                          seed=opt.manualSeed)

for i, data_val in enumerate(valDataloader, 0):
    # if 1:
    print('Image:'+str(i))
    t0 = time.time()

    val_input_cpu, val_target_cpu, label = data_val
    val_input_cpu = val_input_cpu.float().cuda()
    # val_target_cpu = val_target_cpu.float().cuda()

    #   val_batch_output = torch.FloatTensor(val_input.size()).fill_(0)
    val_input.resize_as_(val_input_cpu).copy_(val_input_cpu)
    # val_target=Variable(val_target_cpu, volatile=True)

    label_cpu = torch.FloatTensor(opt.valBatchSize).fill_(0)
    label_cpu = label_cpu.long().cuda()
    label_cpu = Variable(label_cpu)

    for idx in range(val_input.size(0)):
        single_img = val_input[idx,:,:,:].unsqueeze(0)
        val_inputv = Variable(single_img, volatile=True)
        # ********residual-getting network********
        output = residue_net(val_inputv, label_cpu)

        ## Get the residual ##
        residue = val_inputv - output

        # ********Get the density-label using residual********
        label = net_label(residue)
        softmax=nn.Softmax()
        label=softmax(label)

        label_final2 = label.max(1)[1]
        label_final=label_final2+1
        print(label, label_final2, label_final)

        # ********Get de-rained results ********
        x_hat_val = netG(val_inputv, label_final)

        t1 = time.time()
        print('running time:'+str(t1 - t0))

    residual, resukt=x_hat_val
    tensor = resukt.data.cpu()


    ###   Save the de-rained results #####
    # directory = './result_all/new_model_data/testing_our_our/'
    if not os.path.exists(opt.out_directory):
        os.makedirs(opt.out_directory)

    label_final2=label_final2.data.cpu().numpy()
    tensor = torch.squeeze(tensor)
    tensor=norm_range(tensor, None)
    ndarr = tensor.mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
    im = Image.fromarray(ndarr)
    filename=os.path.join(opt.out_directory, str(i)+'.jpg')
    im.save(filename)
