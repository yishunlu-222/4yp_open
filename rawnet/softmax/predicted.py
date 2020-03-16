##predicted
from tqdm import tqdm
from collections import OrderedDict
from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import librosa
import os
import yaml
import numpy as np
import pdb
import torch
import torch.nn as nn
from torch.utils import data
from model_RawNet import RawNet
from prefetch_generator import BackgroundGenerator
import metrics
import time


def cos_sim(a,b):
	return np.dot(a,b) / (np.linalg.norm(a) * np.linalg.norm(b))
	
class Dataset_VoxCeleb2(data.Dataset):
	def __init__(self, list_IDs, base_dir, nb_time = 0, labels = {}, cut = True, return_label = True, pre_emp = True):
		'''
		self.list_IDs	: list of strings (each string: utt key)
		self.labels		: dictionary (key: utt key, value: label integer)
		self.nb_time	: integer, the number of timesteps for each mini-batch
		cut				: (boolean) adjust utterance duration for mini-batch construction
		return_label	: (boolean) 
		pre_emp			: (boolean) do pre-emphasis with coefficient = 0.97
		'''
		self.list_IDs = list_IDs
		self.nb_time = nb_time
		self.base_dir = base_dir
		self.labels = labels
		self.cut = cut
		self.return_label = return_label
		self.pre_emp = pre_emp
		if self.cut and self.nb_time == 0: raise ValueError('when adjusting utterance length, "nb_time" should be input')

	def __len__(self):
		return len(self.list_IDs)

	def __getitem__(self, index):
		ID = self.list_IDs[index]
		sr =16000

		# X = np.load(self.base_dir+ID+'.wav')
		a, sr_ret = librosa.load(self.base_dir+ID+'.wav', sr)
		X = np.expand_dims(a, axis = 0)
		if self.pre_emp: X = self._pre_emphasis(X)
		if self.cut:
			nb_time = X.shape[1]
			if nb_time > self.nb_time:
				start_idx = np.random.randint(low = 0,
					high = nb_time - self.nb_time)
				X = X[:, start_idx:start_idx+self.nb_time]
			elif nb_time < self.nb_time:
				nb_dup = int(self.nb_time / nb_time) + 1
				X = np.tile(X, (1, nb_dup))[:, :self.nb_time]
			else:
				X = X
		if not self.return_label:
			return X
		y = self.labels[ID.split('/')[0]]
		return X, y

	def _pre_emphasis(self, x):
		'''
		Pre-emphasis for single channel input
		'''
		return np.asarray(x[:,1:] - 0.97 * x[:, :-1], dtype=np.float32) 

def get_utt_list(src_dir):
	'''
	Designed for VoxCeleb
	'''
	l_utt = []
	for r, ds, fs in os.walk(src_dir):
		base = '/'.join(r.split('/')[-2:])+'/'
		# pdb.set_trace()
		for f in fs:
 
			if f[-3:] != 'wav':
				continue
			l_utt.append(base+f[:-4])

	return l_utt

def evaluate_model(mode, db_gen, device, l_utt, l_trial, model='eval'):
	if mode not in ['val', 'eval']: raise ValueError('mode should be either "val" or "eval"')
	# l_utt is the unique list and l_trial is the full list of  voxceleb1_veri_test.txt
	model.eval()
	with torch.set_grad_enabled(False):
		#1st, extract speaker embeddings.
		l_embeddings = []
		with tqdm(total = len(db_gen), ncols = 70) as pbar:
			for m_batch in BackgroundGenerator(db_gen):
				code = model(x = m_batch, is_test=True)
				l_embeddings.extend(code.cpu().numpy()) #>>> (batchsize, codeDim)
				pbar.update(1)
		d_embeddings = {}
		
		if not len(l_utt) == len(l_embeddings):
			print(len(l_utt), len(l_embeddings))
			exit()
		for k, v in zip(l_utt, l_embeddings):
			d_embeddings[k] = v

		#2nd, calculate EER
		y_score = [] # score for each sample
		y = [] # label for each sample 

		for line in l_trial:
			trg, utt_a, utt_b = line.strip().split(' ')
			y.append(int(trg))
			if mode =='val':
				y_score.append(cos_sim(d_embeddings[utt_a], d_embeddings[utt_b]))
			else:
				y_score.append(cos_sim(d_embeddings[utt_a[:-4]], d_embeddings[utt_b[:-4]]))
		fpr, tpr, thresholds = roc_curve(y, y_score, pos_label=1)
		eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
		print('eer', eer)
		
	return eer

if __name__ == '__main__':
	## load the parameters
	dir_yaml =  './train_RawNetssh.yaml'
	with open(dir_yaml, 'r') as f_yaml:
		parser = yaml.load(f_yaml)
		
	cuda = torch.cuda.is_available()
	device = torch.device('cuda' if cuda else 'cpu')
	print(device)
	# load the voxceleb_veri.text
	l_eval  = sorted(get_utt_list('/mnt/2TB-1/datasets/wav/'))
	
	with open(parser['DB']+'voxceleb1_veri_test.txt', 'r') as f:
		l_eval_trial = f.readlines()
	# dataloader of the voxceleb_veri.text
	evalset = Dataset_VoxCeleb2(list_IDs = l_eval,
								cut = False,
								return_label = False,
								base_dir ='/mnt/2TB-1/datasets/wav/')
	
	evalset_gen = data.DataLoader(evalset,
		batch_size = 1, #because for evaluation, we do not modify its duration, thus cannot compose mini-batches
		shuffle = False,
		drop_last = False,
		num_workers = parser['nb_proc_db'])
		
	# model = RawNet(parser['model'], device)
	PATH ='./17rawnet.pth'
	model = torch.load(PATH)
	model.to(device)
	eval_eer = evaluate_model(mode = 'eval',
				model = model,
				db_gen = evalset_gen, 
				device = device,
				l_utt = l_eval,
				l_trial = l_eval_trial)
