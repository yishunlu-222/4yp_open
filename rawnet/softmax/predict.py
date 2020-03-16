from __future__ import absolute_import
from __future__ import print_function
import os
import sys
import numpy as np
from model_RawNet import RawNet
import yaml
import toolkits
import util as ut
import torch
import torch.nn as nn
import pdb
from tqdm import tqdm
# ===========================================
#        Parse the argument
# ===========================================
import argparse

parser = argparse.ArgumentParser()
# set up training configuration.
parser.add_argument('--gpu', default='', type=str)
parser.add_argument('--resume', default='', type=str)
parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--sample_rate', default=16000, type=int)
parser.add_argument('--data_path', default='/mnt/2TB-1/datasets/wav', type=str)
# set up network configuration.
parser.add_argument('--net', default='resnet34s', choices=['resnet34s', 'resnet34l'], type=str)
parser.add_argument('--ghost_cluster', default=2, type=int)
parser.add_argument('--vlad_cluster', default=8, type=int)
parser.add_argument('--bottleneck_dim', default=512, type=int)
parser.add_argument('--aggregation_mode', default='gvlad', choices=['avg', 'vlad', 'gvlad'], type=str)
# set up learning rate, training loss and optimizer.
parser.add_argument('--loss', default='softmax', choices=['softmax', 'amsoftmax'], type=str)
parser.add_argument('--test_type', default='normal', choices=['normal', 'hard', 'extend'], type=str)

global args
args = parser.parse_args()


def main():
    # gpu configuration
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    # ==================================
    #       Get Train/Val.
    # ==================================
    print('==> calculating test({}) data lists...'.format(args.test_type))

    if args.test_type == 'normal':
        verify_list = np.loadtxt('/home/weidi/yishun/vox2/voxceleb1_veri_test.txt', str)
    elif args.test_type == 'hard':
        verify_list = np.loadtxt('/home/weidi/yishun/vox2/voxceleb1_veri_test_hard.txt', str)
    elif args.test_type == 'extend':
        verify_list = np.loadtxt('/home/weidi/yishun/vox2/voxceleb1_veri_test_extended.txt', str)
    else:
        raise IOError('==> unknown test type.')
   # from weidi's work
    verify_lb = np.array([int(i[0]) for i in verify_list])
    list1 = np.array([os.path.join(args.data_path, i[1]) for i in verify_list])
    list2 = np.array([os.path.join(args.data_path, i[2]) for i in verify_list])

    total_list = np.concatenate((list1, list2))
    unique_list = np.unique(total_list)
    pdb.set_trace()
    # ==================================
    #       Get Model
    # ==================================
    # construct the data generator.
    dir_yaml = '/home/weidi/yishun/project/ssh/network2/train_RawNet.yaml'
    with open(dir_yaml, 'r') as f_yaml:
        rnparser = yaml.load(f_yaml)
    rnparser['model']['nb_classes'] = 5994
    rawnet = RawNet(rnparser['model'],device)

    print("batch size is {0}".format(args.batch_size))
    print("num of classes is {0}".format(rnparser['model']['nb_classes']))

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        rawnet = nn.DataParallel(rawnet, device_ids=[0, 1, 2, 3])
    else:
        print("Let's use", torch.cuda.device_count(), "GPU!")
    # ==> load pre-trained model ???
    
    number = 17
    
    checkpoint = torch.load('/mnt/4TB/yishun_4yp/result/rawnetorigin/checkpoint_{}.pth'.format(number))
    rawnet.load_state_dict(checkpoint['state_dict'])
    rawnet.to(device)
    print('path is {} ==> start testing.'.format(number))

    # The feature extraction process has to be done sample-by-sample,
    # because each sample is of different lengths.
    total_length = len(unique_list)
    feats, scores, labels = [], [], []


    for c, ID in tqdm(enumerate(unique_list)):
        with torch.no_grad():
            rawnet.eval()
            if c % 50 == 0: print('Finish extracting features for {}/{}th wav.'.format(c, total_length))
            audio = ut.load_data(ID, sr=args.sample_rate, mode='eval')
            input = np.expand_dims(np.expand_dims(audio, 0), -2)
            input = torch.from_numpy(input).float().to(device)
            # pdb.set_trace()
            v =rawnet(input)
            v1 = v.cpu().numpy()
            # pdb.set_trace()
            feats += [v1]
            # torch.cuda.empty_cache()



    feats =np.array(feats)
    np.save('/mnt/4TB/yishun_4yp/result/rawnetorigin/{}thpredictfeats.npy'.format(number),feats)
    # feats = np.load('/mnt/4TB/yishun_4yp/result/rawnetfull2/features/{}_thpredictfeats.npy'.format(number))
    # ==> compute the pair-wise similarity.
    for c, (p1, p2) in enumerate(zip(list1, list2)):
        ind1 = np.where(unique_list == p1)[0][0]
        ind2 = np.where(unique_list == p2)[0][0]

        v1 = feats[ind1, 0]
        v2 = feats[ind2, 0]
		# weidi:np.sum(v1 * v2)
        scores += [cos_sim(v1,v2)]
        labels += [verify_lb[c]]
        # print('scores : {}, gt : {}'.format(scores[-1], verify_lb[c]))

    scores = np.array(scores)
    labels = np.array(labels)

    # np.save('/mnt/4TB/yishun_4yp/result/rawnetfull/prediction_scores.npy', scores)
    # np.save('/mnt/4TB/yishun_4yp/result/rawnetfull/groundtruth_labels.npy', labels)

    eer, thresh = toolkits.calculate_eer(labels, scores)
    print('==> model : {}, EER: {}'.format(args.resume, eer))
    
def cos_sim(a,b):
	return np.dot(a,b) / (np.linalg.norm(a) * np.linalg.norm(b))

if __name__ == "__main__":
    main()
