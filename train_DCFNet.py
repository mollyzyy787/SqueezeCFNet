import os
from os.path import join, isdir, isfile
from os import makedirs
import sys
import argparse
import time
import shutil

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms


from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from models.DCFnet import DCFNet
from dataset import FNTDataset


def gaussian_shaped_labels(sigma, sz):
    x, y = np.meshgrid(np.arange(1, sz[0]+1) - np.floor(float(sz[0]) / 2), np.arange(1, sz[1]+1) - np.floor(float(sz[1]) / 2))
    d = x ** 2 + y ** 2
    g = np.exp(-0.5 / (sigma ** 2) * d)
    g = np.roll(g, int(-np.floor(float(sz[0]) / 2.) + 1), axis=0)
    g = np.roll(g, int(-np.floor(float(sz[1]) / 2.) + 1), axis=1)
    return g.astype(np.float32)

class TrackerConfig(object):
    crop_sz = 200 #network input size is 200x200
    output_sz = 196 #feature map size is 48x48

    lambda0 = 1e-4
    padding = 2.0
    output_sigma_factor = 0.1

    output_sigma = crop_sz / (1 + padding) * output_sigma_factor
    y = gaussian_shaped_labels(output_sigma, [output_sz, output_sz])
    yf = torch.view_as_real(torch.fft.rfft2(torch.Tensor(y).view(1, 1, output_sz, output_sz).cuda(), norm="ortho"))
    # cos_window = torch.Tensor(np.outer(np.hanning(crop_sz), np.hanning(crop_sz))).cuda()  # train without cos window

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def save_checkpoint(state, is_best, save_path):
    filename = join(save_path, 'checkpoint_DCFnet.pt')
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, join(save_path, 'model_best_DCFnet.pt'))


def train(train_loader, model, optimizer, target, epoch):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    loc_losses = AverageMeter()

    loc_loss = nn.MSELoss(reduction='sum').cuda()
    model.train()

    end = time.time()
    for i, (template, search, negative) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        template = template.cuda(non_blocking=True)
        search = search.cuda(non_blocking=True)

        optimizer.zero_grad()

        #compute outputs
        response = model(template, search)
        #compute losses
        loss = loc_loss(response, target)/template.size(0)
        loss.backward()

        optimizer.step()
        # measure accuracy and record loss
        loc_losses.update(loss.item())

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=loc_losses))
    writer.add_scalar('Train/map MSE loss', loc_losses.avg, epoch)

def validate(val_loader, model, target, epoch):
    batch_time = AverageMeter()
    losses_total = AverageMeter()

    loc_loss = nn.MSELoss(reduction='sum').cuda()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (template, search, negative) in enumerate(val_loader):

            template = template.cuda(non_blocking=True)
            search = search.cuda(non_blocking=True)

            # compute output
            response = model(template, search)
            # compute loss
            lossmap = loc_loss(response, target)/template.size(0)
            losses_total.update(lossmap.item())

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses_total))
        print(' * Loss {loss.val:.4f} ({loss.avg:.4f})'.format(loss=losses_total))
        writer.add_scalar('Validate/total loss', losses_total.avg, epoch)
    return losses_total.avg

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Training SqueezeCFNet in Pytorch 1.12.1')
    parser.add_argument('--dataset', dest='dataset', default='', type=str, help='path to dataset *.json file')
    parser.add_argument('--save_path', dest='save_path', default='', type=str, help='path to save trained model parameters')
    parser.add_argument('--input_sz', dest='input_sz', default=200, type=int, help='crop input size')
    parser.add_argument('--padding', dest='padding', default=2.0, type=float, help='crop padding size')
    parser.add_argument('--epochs', default=50, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--print-freq', '-p', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                        help='number of data loading workers (default: 2)')
    parser.add_argument('-b', '--batch-size', default=8, type=int,
                        metavar='N', help='mini-batch size (default: 8)')
    parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=5e-5, type=float,
                        metavar='W', help='weight decay (default: 5e-5)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
    parser.add_argument('--pretrained', default=False, type=bool, help='whether the checkpoint resumed from is a pretrained model')

    args = parser.parse_args()

    print(args)
    torch.cuda.empty_cache()
    best_loss = 1e6

    # load data
    # use dataset, DataLoader to parse data into training_data, test_data etc.
    # set up loss
    config = TrackerConfig()
    # set up net to train
    model = DCFNet(config=config)
    model.cuda()
    gpu_num = torch.cuda.device_count()
    print('GPU NUM: {:2d}'.format(gpu_num))
    if gpu_num > 1:
        model = nn.DataParallel(model, list(range(gpu_num))).cuda()

    # set up optimizer
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True)

    target = torch.Tensor(config.y).cuda().unsqueeze(0).unsqueeze(0).repeat(args.batch_size * gpu_num, 1, 1, 1)  # for training
    #target_n = torch.Tensor(config.yn).cuda().unsqueeze(0).unsqueeze(0).repeat(args.batch_size * gpu_num, 1, 1, 1)  # for training

    # set up tensorboard
    writer = SummaryWriter()

    # optionally resume from a checkpoint
    if args.resume:
        if isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            if args.pretrained:
                # args.start_epoch = 0 #already set to zero in default
                # best_loss = 1e6
                print("Using the checkpoint as pretrained model")
            else:
                args.start_epoch = checkpoint['epoch']
                print("Resuming training from the checkpoint")
                best_loss = checkpoint['best_loss']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    if not args.dataset:
        # FathomNet_mini
        json_file_path = '~/DCNN_CF/curate_dataset/data_sample/FathomNet_wrap.json'
        json_file_path = os.path.expanduser(json_file_path)
    else:
        json_file_path = args.dataset


    if not args.save_path:
        save_param_path = join(os.getcwd(), "checkpoints")
        if not os.path.exists(save_param_path):
            makedirs(save_param_path)
    else:
        save_param_path = args.save_path

    train_dataset = FNTDataset(json_file_path, train=True, normalize=True)
    print("total number of train images: ", len(train_dataset)) #7427 for regular FathomNet, 828 for mini
    val_dataset = FNTDataset(json_file_path, train=False, normalize=True)
    print("total number of val images: ", len(val_dataset)) #818 for regular FathomNet, 87 for mini

    train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size*gpu_num, shuffle=True,
            num_workers=args.workers, pin_memory=True, drop_last=True)

    val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size*gpu_num, shuffle=False,
            num_workers=args.workers, pin_memory=True, drop_last=True)

    for epoch in range(args.start_epoch, args.epochs):
        train(train_loader, model, optimizer, target, epoch)
        loss = validate(val_loader, model, target, epoch)
        lr_scheduler.step(loss)

        writer.add_scalar('Train/lr', optimizer.param_groups[0]['lr'], epoch)

        # remember best loss and save checkpoint
        is_best = loss < best_loss
        best_loss = min(best_loss, loss)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_loss': best_loss,
            'optimizer': optimizer.state_dict(),
            'lr_scheduler' : lr_scheduler.state_dict(),
        }, is_best, save_param_path)

    writer.close()
