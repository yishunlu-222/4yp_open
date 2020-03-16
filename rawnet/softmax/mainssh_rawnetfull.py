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
import numpy as np
import util as ut
parser = argparse.ArgumentParser()
# set up the training hyperparameters and configuration
# parser.add_argument('--gpu', default='', type=str)

parser.add_argument('--resume', default='', type=str)
parser.add_argument('--batch_size', default=256, type=int)  # 256 -- 5G  256+128
parser.add_argument('--data_path', default='/mnt/ssd/voxceleb2/aac', type=str)
parser.add_argument('--val_data_path', default='/mnt/2TB-1/datasets/voxceleb1/wav', type=str)
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
#         # speed up dataloader so workers don't wait
#     def __iter__(self):
#         return BackgroundGenerator(super().__iter__())

def main():
    # no need to initialize the gpu as pytorch would use the gpu as it needs to use
    # toolkits.initialize_GPU(args)
    # enable GPU



    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # import model
    # ==================================
    #       Get correpsonding names of Train/Val.
    # ==================================
    trnlist, trnlb = toolkits.get_voxceleb2_datalist(args.data_path,
                                                     path='/home/weidi/yishun/vox2/vox2train95.txt')
    vallist, vallb = toolkits.get_voxceleb2_datalist(args.data_path,
                                                     path='/home/weidi/yishun/vox2/vox2val5.txt')
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
                                     params_1D['sampling_rate'],time_len=params_1D['time_len'],mode='train')
    # trn_gen = dataloader.DataLoader(partition['train'], labels['train'], **params)
    # val_gen = dataloader.DataLoader(partition['val'], labels['val'], **params)
    val_dataset = voxceleb_dataset(partition['val'], labels['val'],
                                     params_1D['sampling_rate'],time_len=params_1D['time_len'])
    # pdb.set_trace()
    trn_gen = DataLoader(dataset=train_dataset,
                            batch_size=args.batch_size,
                             num_workers=16,
                            pin_memory=True,
                             shuffle=True)
    val_gen = DataLoader(dataset=val_dataset,
                            batch_size=args.batch_size,
                             num_workers=16,
                            pin_memory=True,
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
    # rawnet.apply(init_weights)
    


    # load and optimizer
    
    rnparser['lr'] = 0.0006
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(rawnet.parameters(), 
								lr = rnparser['lr'],
								momentum = rnparser['opt_mom'],
								weight_decay = rnparser['wd'],
								nesterov = bool(rnparser['nesterov']))
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(rawnet.parameters(),
								lr = rnparser['lr'],
								weight_decay = rnparser['wd'],
								amsgrad = bool(rnparser['amsgrad']))

    else:
        raise NotImplementedError('Add other optimizers if needed')
		
    # optimizer.load_state_dict(checkpoint['optimizer'])
    print('finish loading optimizer')
    if bool(rnparser['do_lr_decay']):
         if rnparser['lr_decay'] == 'keras':
            lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = lambda step: keras_lr_decay(step))
			
    # optimizer = create_optimizer(rawnet, args.lr)

    # tensorboard
    writer_val = SummaryWriter('runs/network_rawnetfull2/traintotal_val')
    writer_train = SummaryWriter('runs/network_rawnetfull2/traintotal_train')
    print("start training")
    torch.save(rawnet, '/mnt/4TB/yishun_4yp/result/rawnetfull2/newest_network.pth')
    
    train(trn_gen, val_gen, rawnet, device, optimizer,writer_train=writer_train, 
          writer_val=writer_val,rnparser=rnparser,lr_scheduler=lr_scheduler,
          load_epoch = checkpoint['epoch'],train_num_iter =checkpoint['train_num_iter'],
          val_num_iter=checkpoint['val_num_iter'])
          
    torch.save(rawnet, '/mnt/4TB/yishun_4yp/result/rawnetfull2/networkfinish.pth')
    print("finish training")


def train(train_loader, val_loader, model, device, optimizer, rnparser,lr_scheduler,writer_train=None,
          writer_val=None,load_epoch = 0,train_num_iter = 0,val_num_iter = 0, ):
    # torch.save({'epoch': 0, 'state_dict': model.state_dict(),
    #             'optimizer': optimizer.state_dict()},
    #            '/mnt/4TB/yishun_4yp/result/network2/checkpoint_{}.pth'.format(0))
    # torch.save(model, '/mnt/4TB/yishun_4yp/result/network2/network2.pth')
    
    f_eer = open('/mnt/4TB/yishun_4yp/result/rawnetfull2/eerstest.txt', 'a', buffering = 1)
    print('training the {}th epoch'.format(load_epoch+1))
    time_for_lastepoch = 0
    lr_cur=0.0006
    for epoch in tqdm(range(55-load_epoch)):
        epoch += load_epoch
        train_correct = 0
        train_total = 0
        train_running_loss = 0
        val_correct = 0
        val_total = 0
        val_running_loss = 0
        startfull = time.time()  # test the running time
        model.train()
        
        for trn in enumerate(BackgroundGenerator(train_loader)):
            if bool(rnparser['do_lr_decay']):
                    if rnparser['lr_decay'] == 'keras': lr_scheduler.step()
            train_num_iter += 1
            batch_idx, (x_data, y_label) = trn
            if batch_idx == len(train_loader):
                break
            # #("batch_idx is {0}".format(batch_idx))
            # print#("x_data shape is ", x_data.shape)
            # print(x_data.shape)

            x_data = torch.unsqueeze(x_data,1).to(device)
            y_label = y_label.to(device)

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
            # pdb.set_trace()
            
            if batch_idx > 2:
				
                if  batch_idx % 100 == 0:
                  for p in optimizer.param_groups:
                     lr_cur = p['lr']
					#print('lr_cur', lr_cur)
                     break
                print('epoch: {}, i: {}, Accuracy: {:.4f}, train_Loss: {:.4f}, nloss: {:.4f}, running time: {:.6f}, lr: {:.8f}'.format(
                                                                                                epoch+1, train_num_iter,
                                                                                            100 * train_correct / train_total,
                                                                                            train_running_loss / batch_idx, loss,
                                                                                            time_for_lastepoch,lr_cur))
                if writer_train is not None:
                    writer_train.add_scalar('training loss', train_running_loss / batch_idx,train_num_iter)
                    writer_train.add_scalar('train/val accuracy', 100 * train_correct / train_total, epoch+1)
                    writer_train.add_scalar('training accuracy', 100 * train_correct / train_total, train_num_iter)
                    writer_train.add_scalar('train normal loss',loss , train_num_iter)
                    
        val_num_iter,val_acc, val_loss = validation(val_loader, model, device, writer_val, epoch=epoch,
                                  correct=val_correct, total=val_total, running_loss=val_running_loss,
                                  num_iter=val_num_iter)
        eer =compute_eer(model ,device, epoch)
        # ~ f_eer.write('epoch:%d' %(epoch))
        f_eer.write('epoch:%d,train_accuracy:%d, val_accuracy:%d, train_loss:%f ,val_loss:%f, test_eer:%f\n'%(epoch+1, 100 * train_correct / train_total,
                                                                                                              val_acc,train_running_loss / batch_idx, val_loss, eer))
        
        endfull = time.time()
        time_for_lastepoch = endfull - startfull
        torch.save({'epoch': epoch + 1, 'state_dict': model.state_dict(), 'train_num_iter':train_num_iter,
                    'optimizer': optimizer.state_dict(),'val_num_iter':val_num_iter
                    },
                   '/mnt/4TB/yishun_4yp/result/rawnetfull2/checkpoint_{}.pth'.format(epoch + 1))	
    f_eer.close()
	
	
	
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
    return num_iter, 100 * correct / total, running_loss / batch_idx
    
def compute_eer(model,device, epoch,test_type='normal'):
    if test_type == 'normal':
       verify_list = np.loadtxt('/home/weidi/yishun/vox2/voxceleb1_veri_test.txt', str)
    elif test_type == 'hard':
       verify_list = np.loadtxt('/home/weidi/yishun/vox2/voxceleb1_veri_test_hard.txt', str)
    elif test_type == 'extend':
       verify_list = np.loadtxt('/home/weidi/yishun/vox2/voxceleb1_veri_test_extended.txt', str)
    else:
        raise IOError('==> unknown test type.')  
       # from weidi's work
    verify_lb = np.array([int(i[0]) for i in verify_list])
    list1 = np.array([os.path.join('/mnt/2TB-1/datasets/wav', i[1]) for i in verify_list])
    list2 = np.array([os.path.join('/mnt/2TB-1/datasets/wav', i[2]) for i in verify_list])

    total_list = np.concatenate((list1, list2))
    unique_list = np.unique(total_list)
    print('path is {} ==> start testing.'.format(epoch))
    
    total_length = len(unique_list)
    feats, scores, labels = [], [], []
	
    for c, ID in tqdm(enumerate(unique_list)):
        with torch.no_grad():
            model.eval()
            if c % 50 == 0: print('Finish extracting features for {}/{}th wav.'.format(c, total_length))
            audio = ut.load_data(ID, sr=16000, mode='eval')
            input = np.expand_dims(np.expand_dims(audio, 0), -2)
            input = torch.from_numpy(input).float().to(device)
            # pdb.set_trace()
            v =model(input)
            v1 = v.cpu().numpy()
            # pdb.set_trace()
            feats += [v1]
            # torch.cuda.empty_cache()



    feats =np.array(feats)
    np.save('/mnt/4TB/yishun_4yp/result/rawnetfull2/features/{}_thpredictfeats.npy'.format(epoch+1),feats)
    
    for c, (p1, p2) in enumerate(zip(list1, list2)):
        ind1 = np.where(unique_list == p1)[0][0]
        ind2 = np.where(unique_list == p2)[0][0]

        v1 = feats[ind1, 0]
        v2 = feats[ind2, 0]

        scores += [np.sum(v1 * v2)]
        labels += [verify_lb[c]]
        # print('scores : {}, gt : {}'.format(scores[-1], verify_lb[c]))

    scores = np.array(scores)
    labels = np.array(labels)
    eer, thresh = toolkits.calculate_eer(labels, scores)
    print('==> model epoch: {}, EER: {}'.format(epoch+1, eer))
    
    return eer
    
    
def init_weights(m):
	print(m)
	if isinstance(m, nn.Linear):
		torch.nn.init.xavier_uniform(m.weight)
		m.bias.data.fill_(0.0001)
	elif isinstance(m, nn.BatchNorm1d):
		pass
	else:
		if hasattr(m, 'weight'):
			torch.nn.init.kaiming_normal_(m.weight, a=0.01)
		else:
			print('no weight',m)

def keras_lr_decay(step, decay = 0.00005):
	return 1./(1.+decay*step)


def create_optimizer(model, new_lr):
    # setup optimizer
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), 
								lr = rnparser['lr'],
								momentum = rnparser['opt_mom'],
								weight_decay = rnparser['wd'],
								nesterov = bool(rnparser['nesterov']))
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(),
								lr = rnparser['lr'],
								weight_decay = rnparser['wd'],
								amsgrad = bool(rnparser['amsgrad']))
        # weight_decay=args.wd
    elif args.optimizer == 'adagrad':
        optimizer = optim.Adagrad(model.parameters(),
                                  lr=new_lr,
                                  lr_decay=args.lr_decay,
                                  weight_decay=args.wd)
    return optimizer


if __name__ == '__main__':
    main()
