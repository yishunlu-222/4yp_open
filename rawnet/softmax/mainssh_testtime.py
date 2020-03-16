# -*- coding: utf-8 -*-
import toolkits
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
# import dataloader
import dataset
import os
from model_RawNet import RawNet
from tqdm import tqdm
import pdb
from tensorboardX import SummaryWriter
from dataset import voxceleb_dataset
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator
import time
parser = argparse.ArgumentParser()
# set up the training hyperparameters and configuration
# parser.add_argument('--gpu', default='', type=str)

parser.add_argument('--resume', default='', type=str)
parser.add_argument('--batch_size', default=256, type=int)  # 256 -- 5G  256+128
parser.add_argument('--data_path', default='/mnt/ssd/voxceleb2/aac', type=str)
parser.add_argument('--multiprocess', default=4, type=int)
# set up network configuration.
parser.add_argument('--net', default='resnet34s', choices=['resnet34s', 'resnet34l'], type=str)
parser.add_argument('--kernel_size', default=100, type=int,
                    help='the kernel_size of the 1D CNN')
parser.add_argument('--ghost_cluster', default=2, type=int)
parser.add_argument('--vlad_cluster', default=10, type=int)
parser.add_argument('--bottleneck_dim', default=512, type=int)
parser.add_argument('--aggregation_mode', default='gvlad', choices=['avg', 'vlad', 'gvlad'], type=str)
# set up learning rate, training loss and optimizer.
parser.add_argument('--epochs', default=56, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--warmup_ratio', default=0, type=float)
parser.add_argument('--loss', default='softmax', choices=['softmax', 'amsoftmax'], type=str)
parser.add_argument('--optimizer', default='adam', choices=['adam', 'sgd'], type=str)
parser.add_argument('--ohem_level', default=0, type=int,
                    help='pick hard samples from (ohem_level * batch_size) proposals, must be > 1')
global args
args = parser.parse_args()

# class DataLoaderX(DataLoader):
        # # speed up dataloader so workers don't wait
    # def __iter__(self):
        # return BackgroundGenerator(super().__iter__())

def main():
    # no need to initialize the gpu as pytorch would use the gpu as it needs to use
    # toolkits.initialize_GPU(args)
    # enable GPU
    class DataLoaderX(DataLoader):

        # speed up dataloader so workers don't wait
        def __iter__(self):
            return BackgroundGenerator(super(DataLoaderX,self).__iter__())


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # import model
    # ==================================
    #       Get correpsonding names of Train/Val.
    # ==================================
    trnlist, trnlb = toolkits.get_voxceleb2_datalist(args,
                                                     path='/home/weidi/yishun/vox2/voxlb2_train.txt')
    vallist, vallb = toolkits.get_voxceleb2_datalist(args,
                                                     path='/home/weidi/yishun/vox2/vox2_val.txt')
    #obtain the full path of the dataset

    # construct the parameters for data loader.

    params_1D ={'batch_size': args.batch_size,
                'n_classes': 5994,
                'sampling_rate': 16000,
                'time_len': 3,
     }
    partition = {'train': trnlist.flatten(), 'val': vallist.flatten()}  # ,
    labels = {'train': trnlb.flatten(), 'val': vallb.flatten()}  #

    dir_yaml = '/home/weidi/yishun/project/ssh/network2/train_RawNet.yaml'
    with open(dir_yaml, 'r') as f_yaml:
        rnparser = yaml.load(f_yaml)
    rnparser['model']['nb_classes'] = max(trnlb)+1

    print("batch size is {0}".format(args.batch_size))
    print("num of classes is {0}".format(rnparser['model']['nb_classes']))
    train_dataset = voxceleb_dataset(partition['train'], labels['train'],
                                     params_1D['sampling_rate'],time_len=params_1D['time_len'])
    # trn_gen = dataloader.DataLoader(partition['train'], labels['train'], **params)
    # val_gen = dataloader.DataLoader(partition['val'], labels['val'], **params)
    val_dataset = voxceleb_dataset(partition['val'], labels['val'],
                                     params_1D['sampling_rate'],time_len=params_1D['time_len'])
    # pdb.set_trace()
    trn_gen = DataLoader(dataset=train_dataset,
                            batch_size=args.batch_size,
                             num_workers=8,
							pin_memory= True,
                             shuffle=True)
    val_gen = DataLoader(dataset=val_dataset,
                            batch_size=args.batch_size,
                             num_workers=8,
				pin_memory= True,
                             shuffle=True)
    print(
        "finish loading dataloader"
    )
    # trainset = dataset.Voxdataset
    # train_loader = torch.utils.data.DataLoader(train_dir, batch_size=args.batch_size, shuffle=False, **kwargs)

    rawnet = RawNet(rnparser['model'],device)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        rawnet = nn.DataParallel(rawnet, device_ids=[0, 1, 2, 3])
    else:
        print("Let's use", torch.cuda.device_count(), "GPU!")  #

    # load pretrained model
    checkpoint = torch.load('/mnt/4TB/yishun_4yp/result/rawnetfull2/checkpoint_40.pth')
    rawnet.load_state_dict(checkpoint['state_dict'])
    print("finish loading model and the epoch is {}".format(checkpoint['epoch']))
    rawnet.to(device)
    # pdb.set_trace()

    # optimizer
    optimizer = create_optimizer(rawnet, args.lr)
    pdb.set_trace()
    optimizer.load_state_dict(checkpoint['optimizer'])
    for param_group in optimizer.param_groups:
       print(param_group['lr'])
    
    pdb.set_trace()

    # tensorboard
    # writer_val = SummaryWriter('runs/network_rawnetfull/val')
    # writer_train = SummaryWriter('runs/network_rawnetfull/train')
    print("start training")
    train(trn_gen, val_gen, rawnet, device, optimizer)
    # , load_epoch = checkpoint['epoch']+1)
    # torch.save(rawnet, '/mnt/4TB/yishun_4yp/result/rawnetfull/networkfinish.pth')
    print("finish training")


def train(train_loader, val_loader, model, device, optimizer, writer_train=None,
          writer_val=None,load_epoch = 0):
    # torch.save({'epoch': 0, 'state_dict': model.state_dict(),
    #             'optimizer': optimizer.state_dict()},
    #            '/mnt/4TB/yishun_4yp/result/network2/checkpoint_{}.pth'.format(0))
    # torch.save(model, '/mnt/4TB/yishun_4yp/result/network2/network2.pth')
    train_num_iter = 0
    val_num_iter = 0
    print('training the {}th epoch'.format(load_epoch))

    for epoch in tqdm(range(45-load_epoch)):
        epoch += load_epoch
        train_correct = 0
        train_total = 0
        train_running_loss = 0
        val_correct = 0
        val_total = 0
        val_running_loss = 0
        start = time.time()  # test the running time
        model.train()
        for trn in enumerate(BackgroundGenerator(train_loader)):
            train_num_iter += 1
            batch_idx, (x_data, y_label) = trn
            if batch_idx == len(train_loader):
                break
            # #("batch_idx is {0}".format(batch_idx))
            # print#("x_data shape is ", x_data.shape)
            # print(x_data.shape)

            x_data = torch.unsqueeze(x_data,1).float().to(device, non_blocking=True)
            y_label = y_label.to(device, non_blocking=True)

            output_x = model(x_data)

            optimizer.zero_grad()
            criterion = nn.CrossEntropyLoss()
            # pdb.set_trace()
            loss = criterion(output_x, y_label)
            loss.backward()
            optimizer.step()
            # Accuracy
            _, predicted = torch.max(output_x.data, 1)
            train_total += y_label.size(0)
            train_correct += (predicted == y_label).sum().item()
            # Loss
            train_running_loss += float(loss)
            if batch_idx ==100:
                end = time.time()
                print('running time for 100 batches: ', end - start)
                pdb.set_trace()
            if batch_idx > 2:
                print('epoch: {}, i: {}, train_Accuracy: {:.4f}, train_Loss: {:.4f}'.format(epoch+1, train_num_iter,
                                                                                            100 * train_correct / train_total,
                                                                                            train_running_loss / batch_idx))
                # if writer_train is not None:
                    # writer_train.add_scalar('training loss', train_running_loss / batch_idx,train_num_iter)
                    # writer_train.add_scalar('train/val accuracy', 100 * train_correct / train_total, epoch+1)
                    # writer_train.add_scalar('training accuracy', 100 * train_correct / train_total, train_num_iter)

        # torch.save({'epoch': epoch + 1, 'state_dict': model.state_dict(), 'train_num_iter':train_num_iter,
                    # 'optimizer': optimizer.state_dict(),'val_num_iter':val_num_iter},
                   # '/mnt/4TB/yishun_4yp/result/rawnetfull/checkpoint_{}.pth'.format(epoch + 1))

        val_num_iter = validation(val_loader, model, device, writer_val, epoch=epoch,
                                  correct=val_correct, total=val_total, running_loss=val_running_loss,
                                  num_iter=val_num_iter)


def validation(val_loader, model, device, writer_val,epoch, correct, total, running_loss, num_iter):
    with torch.no_grad():
        model.eval()
        for trn in enumerate(BackgroundGenerator(val_loader)):
            num_iter += 1
            batch_idx, (x_data, y_label) = trn
            if batch_idx == len(val_loader):
                break
            # #("batch_idx is {0}".format(batch_idx))
            # print#("x_data shape is ", x_data.shape)
            x_data = torch.unsqueeze(x_data,1).to(device)
            y_label = y_label.to(device)

            output_x = model(x_data)

            criterion = nn.CrossEntropyLoss()
            # pdb.set_trace()
            loss = criterion(output_x, y_label)
            # Accuracy
            _, predicted = torch.max(output_x.data, 1)

            total += y_label.size(0)
            correct += (predicted == y_label).sum().item()
            # Loss
            running_loss += float(loss)
            # pdb.set_trace()
            if batch_idx > 2:  # print every 2 mini-batches
                print('epoch: {}, iter: {}, val_Accuracy: {:.4f}, val_Loss: {:.4f}'.format(epoch+1,num_iter,
                                                                                100 * correct / total,
                                                                                running_loss / batch_idx))
                if writer_val is not None:
                    writer_val.add_scalar('val loss', running_loss / batch_idx, epoch+1)
                    writer_val.add_scalar('train/val accuracy', 100 * correct / total, epoch+1)
                    writer_val.add_scalar('validation accuracy', 100 * correct / total, num_iter)
    return num_iter


def create_optimizer(model, new_lr):
    # setup optimizer
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=new_lr,
                              momentum=0.9, dampening=0.9,
                              weight_decay=args.wd)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=new_lr
                               )
        # weight_decay=args.wd
    elif args.optimizer == 'adagrad':
        optimizer = optim.Adagrad(model.parameters(),
                                  lr=new_lr,
                                  lr_decay=args.lr_decay,
                                  weight_decay=args.wd)
    return optimizer


if __name__ == '__main__':
    main()
