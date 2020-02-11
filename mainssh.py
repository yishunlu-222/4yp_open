# -*- coding: utf-8 -*-
import toolkits
import argparse
import torchvision
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import dataloader
import dataset
import os
from backbone_class_2_ssh import resnet_1D_34
from VLADssh2 import model
from tqdm import tqdm
import pdb
from tensorboardX import SummaryWriter



parser = argparse.ArgumentParser()
# set up the training hyperparameters and configuration
# parser.add_argument('--gpu', default='', type=str)

parser.add_argument('--resume', default='', type=str)
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--data_path', default='/mnt/2TB-1/datasets/voxceleb1/wav', type=str)
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

def main():
    # no need to initialize the gpu as pytorch would use the gpu as it needs to use
    # toolkits.initialize_GPU(args)
    # enable GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # import model
    # ==================================
    #       Get correpsonding names of Train/Val.
    # ==================================
    trnlist, trnlb = toolkits.get_voxceleb2_datalist(args,
                                                     path='/home/weidi/yishun/vox1/vox120train.txt')
    vallist, vallb = toolkits.get_voxceleb2_datalist(args,
                                                     path='/home/weidi/yishun/vox1/vox120test.txt')
    # val is not right at this moment

    # construct the parameters for data loader.
    params = {'dim': (257, 250, 1),
              'mp_pooler': toolkits.set_mp(processes=args.multiprocess),
              'nfft': 512,
              'spec_len': 250,
              'win_length': 400,
              'hop_length': 160,
              'n_classes': 200,
              'sampling_rate': 10000,
              'batch_size': args.batch_size,
              'shuffle': True,
              'normalize': True,
              }

    partition = {'train': trnlist.flatten(),'val': vallist.flatten()}  #,
    labels = {'train': trnlb.flatten(), 'val': vallb.flatten()}  #

    print("batch size is {0}".format(args.batch_size))
    trn_gen = dataloader.DataLoader(partition['train'], labels['train'], **params)
    val_gen =dataloader.DataLoader(partition['val'], labels['val'], **params)

    # pdb.set_trace()
    print(
        "finish loading dataloader"
    )
    # trainset = dataset.Voxdataset
    # train_loader = torch.utils.data.DataLoader(train_dir, batch_size=args.batch_size, shuffle=False, **kwargs)

    my_tensor = torch.randn(1, 1,  params["sampling_rate"]*3)
    my_tensor = my_tensor.to(device)
    kernel_size = args.kernel_size
    resnet  = resnet_1D_34(kernel_size=kernel_size)
    resnet.to(device)

    y = resnet(my_tensor)
    # torch.cuda.empty_cache()
    vladmodel = model(mode='gvlad', k_centers=8,num_classes=params['n_classes'],g_centers=2, dim=256,
                      input_shape=y.shape[-1],kernel_size=kernel_size)
    del y
    del resnet
    #load pretrained model
    #checkpoint = torch.load('/mnt/4TB/yishun_4yp/result/network2/checkpoint_50.pth')
    #vladmodel.load_state_dict(checkpoint['state_dict'])

    # if torch.cuda.device_count() > 1:
        # ("Let's use", torch.cuda.device_count(), "GPUs!")
        # # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        # vladmodel = nn.DataParallel(vladmodel,device_ids=[0,1,2,3])
    # else:
        # ("Let's use", torch.cuda.device_count(), "GPU!")#
    vladmodel.to(device)
    # pdb.set_trace()
    print("finish loading model")
    #tensorboard
    # writer_net = SummaryWriter('/mnt/4TB/yishun_4yp/result/network2/runs/network2')
    # pdb

    # optimizer
    optimizer = create_optimizer(vladmodel, args.lr)

    # pdb.set_trace()

    # training
    print("start training")
    writer_val = SummaryWriter('runs/network2_10/val20')
    writer_train = SummaryWriter('runs/network2_10/train20')
        # validation(val_gen, vladmodel, device,writer_val)
    train(trn_gen,val_gen,vladmodel, device, optimizer,writer_train=writer_train,writer_val=writer_val)
    torch.save(vladmodel, '/mnt/4TB/yishun_4yp/result/network21/network2.pth')
    print("finish training")


def train(train_loader,val_loader,model,device,optimizer,writer_train,writer_val):

    # torch.save({'epoch': 0, 'state_dict': model.state_dict(),
    #             'optimizer': optimizer.state_dict()},
    #            '/mnt/4TB/yishun_4yp/result/network2/checkpoint_{}.pth'.format(0))
    # torch.save(model, '/mnt/4TB/yishun_4yp/result/network2/network2.pth')

    for epoch in range(45):
        train_num_iter = 0
        train_correct = 0
        train_total = 0
        train_running_loss = 0
        val_correct = 0
        val_total = 0
        val_running_loss = 0
        val_num_iter = 0

        model.train()
        for trn in tqdm(enumerate(train_loader)):
            train_num_iter +=1
            batch_idx, (x_data, y_label) = trn
            if batch_idx == len(train_loader):
                break
            # #("batch_idx is {0}".format(batch_idx))
            # print#("x_data shape is ", x_data.shape)
            x_data = torch.from_numpy(x_data).float().to(device)
            # print(x_data.shape)
            x_data = x_data.permute(0,2,1)
            y_label = torch.from_numpy(y_label).long().to(device)
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

            if batch_idx % 2 == 1:    # print every 2 mini-batches
                print('epoch: {}, i: {}, train_Accuracy: {:.4f}, train_Loss: {:.4f}'.format (epoch,batch_idx,
                                                                      100 * train_correct / train_total,
                                                                      train_running_loss / batch_idx))

                writer_train.add_scalar('training loss', train_running_loss / batch_idx, train_num_iter)
                writer_train.add_scalar('training accuracy', 100 * train_correct / train_total, train_num_iter)

        torch.save({'epoch': epoch+1, 'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()},
               '/mnt/4TB/yishun_4yp/result/network21/checkpoint_{}.pth'.format(epoch+1))

        
        validation(val_loader, model, device, writer_val,
                  correct=val_correct,total=val_total,running_loss=val_running_loss,num_iter=val_num_iter)




def validation(val_loader,model,device,writer_val,correct,total,running_loss,num_iter):

    with torch.no_grad():
        model.eval()
        for trn in tqdm(enumerate(val_loader)):
            num_iter += 1
            batch_idx, (x_data, y_label) = trn
            if batch_idx == len(val_loader):
                break
            # #("batch_idx is {0}".format(batch_idx))
            # print#("x_data shape is ", x_data.shape)
            x_data = torch.from_numpy(x_data).float().to(device)

            # print(x_data.shape)
            x_data = x_data.permute(0, 2, 1)
            # pdb.set_trace()
            y_label = torch.from_numpy(y_label).long().to(device)
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
            if batch_idx % 2 == 1:  # print every 2 mini-batches
                print('iter: {}, val_Accuracy: {:.4f}, val_Loss: {:.4f}'.format( num_iter,
                                                                                   100 * correct / total,
                                                                                   running_loss / batch_idx))
                writer_val.add_scalar('val loss', running_loss / batch_idx, num_iter)
                writer_val.add_scalar('val accuracy', 100 * correct / total, num_iter)

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
