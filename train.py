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

from models.squeezeCFnet import FeatSqueezeNet, SqueezeCFNet
from dataset import FNTDataset
from utils import gaussian_shaped_labels, unravel_index

class TrackerConfig(object):
    crop_sz = 200 #network input size is 200x200
    output_sz = 48 #feature map size is 48x48

    lambda0 = 1e-4
    padding = 2.0
    output_sigma_factor = 0.1

    output_sigma = crop_sz / (1 + padding) * output_sigma_factor
    y = gaussian_shaped_labels(output_sigma, [output_sz, output_sz])
    yf = torch.view_as_real(torch.fft.rfft2(torch.Tensor(y).view(1, 1, output_sz, output_sz).cuda(), norm="ortho"))
    #ynf = torch.view_as_real(torch.fft.rfft2(torch.Tensor(yn).view(1, 1, output_sz, output_sz).cuda(), norm="ortho")) #ynf not used in model
    use_fire_layer = "all"
    # cos_window = torch.Tensor(np.outer(np.hanning(crop_sz), np.hanning(crop_sz))).cuda()  # train without cos window


class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def calc_euclidean(self, x1, x2):
        return (x1 - x2).pow(2).sum(1)

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        distance_positive = self.calc_euclidean(anchor, positive)
        distance_negative = self.calc_euclidean(anchor, negative)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean()

class PSRLoss(nn.Module):
    def __init__(self):
        super(PSRLoss, self).__init__()
    def forward(self, response):
        response_flat = torch.flatten(response, start_dim=1) #(N, H*W)
        h = response.shape[-2]
        w = response.shape[-1]
        max_coords = unravel_index(torch.argmax(response_flat, dim=1), (h, w)) #argmax is (N, 1), coords is (N, 2)
        y_coords = max_coords[:,0]
        x_coords = max_coords[:,1]
        b_sz = response_flat.shape[0]
        mask = torch.ones(response.size()).cuda()
        for i in range(b_sz):
            cy = y_coords[i]
            cx = x_coords[i]
            mask[i,:, max(0,cy-5):min(cy+6,h), max(0,cx-5):min(cx+6, w)] = 0
        response_masked = response*mask
        mean = torch.ones(8).cuda()
        std = torch.ones(8).cuda()
        for i in range(b_sz):
            response_i = response_masked[i,:,:,:]
            mean[i] = torch.mean(response_i[response_i>0])
            std[i] = torch.std(response_i[response_i>0])
        F_max = torch.max(response_flat, dim=1)
        psr = (F_max.values-mean)/(std+1e-5)
        return psr.mean()

class APCELoss(nn.Module):
    def __init__(self):
        super(APCELoss, self).__init__()
    def forward(self, response):
        response_flat = torch.flatten(response, start_dim=1) #(N, H*W)
        Fmax = torch.max(response_flat, dim=1)
        Fmin = torch.min(response_flat, dim=1)
        apce = (Fmax.values-Fmin.values)**2/(torch.mean((response_flat-Fmin.values[:,None])**2, dim=1))
        return apce.mean()

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
    filename = join(save_path, 'checkpoint.pt')
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, join(save_path, 'model_best.pt'))


def train(train_loader, model, loss_mode, optimizer, target, epoch):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    loc_losses = AverageMeter()
    cf_losses = AverageMeter()
    enc_losses = AverageMeter()
    losses = AverageMeter()

    loc_loss = nn.MSELoss(reduction='sum').cuda()
    enc_loss = TripletLoss().cuda()
    psr_loss = PSRLoss().cuda()
    apce_loss = APCELoss().cuda()

    model.train()

    end = time.time()
    for i, (template, search, negative) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        template = template.cuda(non_blocking=True)
        search = search.cuda(non_blocking=True)
        negative = negative.cuda(non_blocking=True)

        #grid1 = torchvision.utils.make_grid(template)
        #writer.add_image("templates", grid1)
        #grid2 = torchvision.utils.make_grid(search)
        #writer.add_image("searches", grid2)
        #grid3 = torchvision.utils.make_grid(negative)
        #writer.add_image("negatives", grid3)
        #writer.add_graph(model, (template, search, negative))
        # zero the parameter gradients
        optimizer.zero_grad()

        #compute outputs
        p_response, n_response, z_encode, x_encode, n_encode = model(template, search, negative)
        #compute losses
        lossmap = loc_loss(p_response, target)/template.size(0)
        #print("p_response PSR: ", psr_loss(p_response).item())
        #print("n_response PSR: ", psr_loss(n_response).item())
        #lossPSR = - psr_loss(p_response) + psr_loss(n_response)
        lossAPCE = - apce_loss(p_response) + apce_loss(n_response)
        lossCF = lossAPCE
        lossENC = enc_loss(x_encode, z_encode, n_encode)
        total_loss = lossCF+lossENC
        if loss_mode == "cf":
             # compute gradient and do SGD step
            lossCF.backward()
        elif loss_mode == "enc":
            lossENC.backward()
        elif loss_mode == 'map':
            lossmap.backward()
        else:
            total_loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0, norm_type=2)
        optimizer.step()
        # measure accuracy and record loss
        losses.update(total_loss.item())
        loc_losses.update(lossmap.item())
        cf_losses.update(lossCF.item())
        enc_losses.update(lossENC.item())

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))
    writer.add_scalar('Train/map MSE loss', loc_losses.avg, epoch)
    writer.add_scalar('Train/CF loss', cf_losses.avg, epoch )
    writer.add_scalar('Train/encoding loss', enc_losses.avg, epoch)
    writer.add_scalar('Train/total loss', losses.avg, epoch)

def validate(val_loader, model, loss_mode, target, epoch):
    batch_time = AverageMeter()
    losses_focus = AverageMeter()
    losses_total = AverageMeter()

    loc_loss = nn.MSELoss(reduction='sum').cuda()
    enc_loss = TripletLoss().cuda()
    psr_loss = PSRLoss().cuda()
    apce_loss = APCELoss().cuda()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (template, search, negative) in enumerate(val_loader):

            template = template.cuda(non_blocking=True)
            search = search.cuda(non_blocking=True)
            negative = negative.cuda(non_blocking=True)

            # compute output
            p_response, n_response, z_encode, x_encode, n_encode = model(template, search, negative)
            # compute loss
            #lossmap = loc_loss(p_response, target)/template.size(0)
            #lossPSR = - psr_loss(p_response) + psr_loss(n_response)
            lossAPCE = -apce_loss(p_response) + apce_loss(n_response)
            #loss1 = lossmap + lossPSR
            lossCF=lossAPCE
            lossENC = enc_loss(x_encode, z_encode, n_encode)
            loss_total=lossCF+lossENC

            if loss_mode == "cf":
                loss_focus = lossCF
            elif loss_mode == "enc":
                loss_focus = lossENC
            else:
                loss_focus = loss_total
            losses_focus.update(loss_focus.item())
            losses_total.update(loss_total.item())

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses_focus))
        print(' * Loss {loss.val:.4f} ({loss.avg:.4f})'.format(loss=losses_focus))
        writer.add_scalar('Validate/focused loss', losses_focus.avg, epoch)
        writer.add_scalar('Validate/total loss', losses_total.avg, epoch)
    return losses_focus.avg, losses_total.avg

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
    parser.add_argument('--train_mode', default=0, type=int, metavar='N',
                        help='training mode, default 0: train with both cf and encoding loss; 1: train with cf loss only')


    args = parser.parse_args()

    print(args)
    torch.cuda.empty_cache()
    best_loss = 1e6

    # load data
    # use dataset, DataLoader to parse data into training_data, test_data etc.
    # set up loss
    config = TrackerConfig()
    # set up net to train
    model = SqueezeCFNet(config=config, kernel='linear')
    model.cuda()
    gpu_num = torch.cuda.device_count()
    print('GPU NUM: {:2d}'.format(gpu_num))
    if gpu_num > 1:
        model = nn.DataParallel(model, list(range(gpu_num))).cuda()

    # set up optimizer
    optimizer_enc = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    optimizer_cf = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    lr_scheduler_enc = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_enc, verbose=True)
    lr_scheduler_cf = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_cf, verbose=True)

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
            optimizer_enc.load_state_dict(checkpoint['optimizer_enc'])
            optimizer_cf.load_state_dict(checkpoint['optimizer_cf'])
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

        if args.train_mode == 0:
            train_cf = (int(epoch/10)%2 == 0) #staggering training for two types of losses
        else:
            train_cf = True #train on cf loss only

        if train_cf: 
            # train for one epoch
            print("training for CF error")
            train(train_loader, model, "cf", optimizer_cf, target, epoch)
            # evaluate on validation set
            loss, total_loss = validate(val_loader, model, "cf", target, epoch)
            lr_scheduler_cf.step(loss)
        else:
            print("training for encoding error")
            train(train_loader, model, "enc", optimizer_enc, target, epoch)
            loss, total_loss = validate(val_loader, model, "enc", target, epoch)
            lr_scheduler_enc.step(loss)

        writer.add_scalar('Train/cf_lr', optimizer_cf.param_groups[0]['lr'], epoch)
        writer.add_scalar('Train/enc_lr', optimizer_enc.param_groups[0]['lr'], epoch)

        # remember best loss and save checkpoint
        is_best = loss < best_loss
        best_loss = min(best_loss, loss)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_loss': best_loss,
            'optimizer_enc': optimizer_enc.state_dict(),
            'optimizer_psr': optimizer_cf.state_dict(),
            'lr_scheduler_psr' : lr_scheduler_cf.state_dict(), #pretrained parameter in checkpoints/ are uunder the name "psr" for "cf"
            'lr_scheduler_enc' : lr_scheduler_enc.state_dict(),
        }, is_best, save_param_path)

    writer.close()
