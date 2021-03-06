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
from tensorboardX import SummaryWriter
import metrics
import time
def keras_lr_decay(step, decay = 0.00005):
	return 1./(1.+decay*step)
	
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


def train_model(model, device, db_gen, optimizer, epoch,train_num_iter,writer_train=None,metric_fc=None):
	model.train()
	train_running_loss = 0
	train_total = 0
	train_correct = 0
	
	for idx_ct, (m_batch, m_label) in enumerate(BackgroundGenerator(db_gen)):
		if epoch != 0 :
			if bool(parser['do_lr_decay']):
				if parser['lr_decay'] == 'keras': lr_scheduler.step()
		train_num_iter += 1
		m_batch = m_batch.to(device)
		m_label= m_label.to(device)

		feature = model(m_batch) #output
		output = metric_fc(feature, m_label)

		loss = criterion(output, m_label)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		_, predicted = torch.max(output.data, 1)
		train_total += m_label.size(0)
		train_correct += (predicted == m_label).sum().item()
		# Loss
		# train_running_loss += float(loss)
		train_running_loss += float(loss)
		if idx_ct % 100 == 0:
			for p in optimizer.param_groups:
				lr_cur = p['lr']
				#print('lr_cur', lr_cur)
				break
		if idx_ct >2:
			print('epoch: {}, i: {}, Accuracy: {:.4f}, train_Loss: {:.4f}, nloss: {:.4f},  lr: {:.8f}'.format(
																			epoch+1, 
																			train_num_iter,
																			100 * train_correct / train_total,
																			train_running_loss / idx_ct, loss,
																			lr_cur))
			writer_train.add_scalar('training loss', train_running_loss / idx_ct,train_num_iter)
			writer_train.add_scalar('train/val accuracy', 100 * train_correct / train_total, epoch+1)
			writer_train.add_scalar('training accuracy', 100 * train_correct / train_total, train_num_iter)
			writer_train.add_scalar('train normal loss',loss , train_num_iter)

	torch.save({'epoch': epoch + 1, 'state_dict': model.state_dict(), 'train_num_iter':train_num_iter,
			'optimizer': optimizer.state_dict()
			},
		   '/mnt/4TB/yishun_4yp/result/rawnetarcface0.55/checkpoint_{}.pth'.format(epoch + 1))
	return train_num_iter,100 * train_correct / train_total

def evaluate_model(mode, model, db_gen, device, l_utt, save_dir, epoch, l_trial):
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
		if mode == 'val':
			f_res = open(save_dir + 'results/epoch%s.txt'%(epoch), 'w')
		else:
			f_res = open(save_dir + 'results/eval.txt', 'w')

		for line in l_trial:
			trg, utt_a, utt_b = line.strip().split(' ')
			y.append(int(trg))
			if mode =='val':
				y_score.append(cos_sim(d_embeddings[utt_a], d_embeddings[utt_b]))
			else:
				y_score.append(cos_sim(d_embeddings[utt_a[:-4]], d_embeddings[utt_b[:-4]]))
			f_res.write('{score} {target}\n'.format(score=y_score[-1],target=y[-1]))
			
		f_res.close()
		fpr, tpr, thresholds = roc_curve(y, y_score, pos_label=1)
		eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
		
		if mode =='val':
			writer_val = SummaryWriter('runs/rawnetarcface0.55/traintotal_val')
			writer_val.add_scalar('val_eer', eer, epoch+1)
			# writer_val.add_scalar('train/val accuracy', 100 * correct / total, epoch+1)
			# writer_val.add_scalar('validation accuracy', 100 * correct / total, num_iter)
		else:
			writer_eer = SummaryWriter('runs/rawnetarcface0.0.55/traintotal_eer')
			writer_eer.add_scalar('test_eer', eer, epoch+1)
		print('eer', eer)
		
	return eer

def cos_sim(a,b):
	return np.dot(a,b) / (np.linalg.norm(a) * np.linalg.norm(b))

def get_val_utts(l_val_trial):
	l_utt = []
	for line in l_val_trial:
		_, utt_a, utt_b = line.strip().split(' ')
		if utt_a not in l_utt: l_utt.append(utt_a)
		if utt_b not in l_utt: l_utt.append(utt_b)
	return l_utt

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
			
def get_label_dic_Voxceleb(l_utt):
	d_label = {}
	idx_counter = 0
	for utt in l_utt:
		spk = utt.split('/')[0]
		if spk not in d_label:
			d_label[spk] = idx_counter
			idx_counter += 1 
	return d_label

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

def make_validation_trial(l_utt, nb_trial, dir_val_trial):
	f_val_trial = open(dir_val_trial, 'w')
	#trg trial: 1, non-trg: 0
	nb_trg_trl = int(nb_trial / 2)
	d_spk_utt = {}
	#make a dictionary that has keys as speakers 
	for utt in l_utt:
		spk = utt.split('/')[0]
		if spk not in d_spk_utt: d_spk_utt[spk] = []
		d_spk_utt[spk].append(utt)

	l_spk = list(d_spk_utt.keys())
	#print('nb_spk: %d'%len(l_spk))
	#compose trg trials
	selected_spks = np.random.choice(l_spk, size=nb_trg_trl, replace=True) 
	for spk in selected_spks:
		l_cur = d_spk_utt[spk]
		utt_a, utt_b = np.random.choice(l_cur, size=2, replace=False)
		f_val_trial.write('1 %s %s\n'%(utt_a, utt_b))
	#compose non-trg trials
	for i in range(nb_trg_trl):
		spks_cur = np.random.choice(l_spk, size=2, replace = False)
		utt_a = np.random.choice(d_spk_utt[spks_cur[0]], size=1)[0]
		utt_b = np.random.choice(d_spk_utt[spks_cur[1]], size=1)[0]
		f_val_trial.write('0 %s %s\n'%(utt_a, utt_b))
	f_val_trial.close()
	return

if __name__ == '__main__':
	dir_yaml =  './train_RawNetssh.yaml'
	with open(dir_yaml, 'r') as f_yaml:
		parser = yaml.load(f_yaml)
	np.random.seed(parser['seed'])
	
	#device setting
	cuda = torch.cuda.is_available()
	device = torch.device('cuda' if cuda else 'cpu')
	print(device)

	#get utt_lists & define labels
	l_dev  = sorted(get_utt_list(parser['DB_vox2']+parser['dev_wav']))
	# pdb.set_trace()
	l_val  = sorted(get_utt_list(parser['DB']+parser['val_wav']))

	l_eval  = sorted(get_utt_list('/mnt/2TB-1/datasets/wav/'))
	
	d_label_vox2 = get_label_dic_Voxceleb(l_dev)
	parser['model']['nb_classes'] = len(list(d_label_vox2.keys()))


	#def make_validation_trial(l_utt, nb_trial, dir_val_trial):
	if bool(parser['make_val_trial']):
		make_validation_trial(l_utt=l_val, nb_trial=parser['nb_val_trial'], dir_val_trial=parser['DB']+'val_trial.txt')
	with open(parser['DB']+'val_trial.txt', 'r') as f:
		l_val_trial = f.readlines()
	with open(parser['DB']+'voxceleb1_veri_test.txt', 'r') as f:
		l_eval_trial = f.readlines()

	
	#define dataset generators
	l_val = get_val_utts(l_val_trial)

	devset = Dataset_VoxCeleb2(list_IDs = l_dev,
		labels = d_label_vox2,
		nb_time = parser['nb_time'],
		base_dir = parser['DB_vox2']+parser['dev_wav'])
	# devset = Dataset_VoxCeleb2(list_IDs =trnlist,
		# labels = trnlb,
		# nb_time = parser['nb_time'],
		# base_dir = parser['DB_vox2']+parser['dev_wav'])
	devset_gen = data.DataLoader(devset,
		batch_size = parser['batch_size'],
		shuffle = True,
		drop_last = True,
		num_workers = parser['nb_proc_db'])
		
	valset = Dataset_VoxCeleb2(list_IDs = l_val,
		return_label = False,
		nb_time = parser['nb_time'],
		base_dir = parser['DB']+parser['val_wav'])
	valset_gen = data.DataLoader(valset,
		batch_size = parser['batch_size'],
		shuffle = False,
		drop_last = False,
		num_workers = parser['nb_proc_db'])
	evalset = Dataset_VoxCeleb2(list_IDs = l_eval,
		cut = False,
		return_label = False,
		base_dir ='/mnt/2TB-1/datasets/wav/')
	evalset_gen = data.DataLoader(evalset,
		batch_size = 1, #because for evaluation, we do not modify its duration, thus cannot compose mini-batches
		shuffle = False,
		drop_last = False,
		num_workers = parser['nb_proc_db'])

	#set save directory
	save_dir = parser['save_dir'] + parser['name'] + '/'
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)
	if not os.path.exists(save_dir  + 'results/'):
		os.makedirs(save_dir + 'results/')
	if not os.path.exists(save_dir  + 'models/'):
		os.makedirs(save_dir + 'models/')
	
	
	#define model
	if bool(parser['mg']):
		model_1gpu = RawNet(parser['model'], device)
		nb_params = sum([param.view(-1).size()[0] for param in model_1gpu.parameters()])
		model = nn.DataParallel(model_1gpu)
	else:
		model = RawNet(parser['model'], device).to(device)
		nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
	
	load_path = True
	
	if load_path is not True:
		model.apply(init_weights)
		print('start from initial weights')
	print('nb_params: {}'.format(nb_params))
	checkpoint = torch.load('/mnt/4TB/yishun_4yp/result/rawnetorigin/checkpoint_{}.pth'.format(17))
	model.load_state_dict(checkpoint['state_dict'])
	#
	print(model)
	# delete the last layer of pretrained
	model.module.fc2_gru= torch.nn.Sequential(*(list(model.module.fc2_gru.children())[:-1]))
	model.to(device)
	# torch.save(model, '/mnt/4TB/yishun_4yp/result/rawnetorigin/17rawnet.pth')
	
	# model.to(device)
	#set ojbective funtions
	criterion = nn.CrossEntropyLoss()
	metric_fc = metrics.ArcMarginProduct(1024, parser['model']['nb_classes'], s=64, m=0.5)
	metric_fc = nn.DataParallel(metric_fc)
	metric_fc.to(device)
	#set optimizer
	params = [
		{'params': [param for name, param in model.named_parameters()
						if 'bn' not in name]},
		{'params': [param for name, param in model.named_parameters()
				if 'bn' in name],
			'weight_decay': 0},   # weight decay 0 for batch normalization 
		{'params': metric_fc.parameters()}
	]
	
		##########################################
	#Train####################################
	##########################################
	#tensorboard
	writer_train = SummaryWriter('runs/rawnetarcface0.55/traintotal_train')
	f_eer = open(save_dir + 'eers.txt', 'a', buffering = 1)
	train_num_iter = 0
	for epoch in tqdm(range(parser['epoch'])):
		if load_path is True:  # fisrt has lr=0.001,then decay
			if epoch == 0:
				
				optimizer = torch.optim.Adam(params,
				lr = parser['lr'],
				weight_decay = parser['wd'],
				amsgrad = bool(parser['amsgrad']))
				
			elif epoch == 1:
				params = [
					{'params': [param for name, param in model.named_parameters()
									if 'bn' not in name]},
					{'params': [param for name, param in model.named_parameters()
							if 'bn' in name],
						'weight_decay': 0},   # weight decay 0 for batch normalization 
					{'params': metric_fc.parameters()}]
				
				optimizer = torch.optim.Adam(params,
				lr = parser['lr'],
				weight_decay = parser['wd'],
				amsgrad = bool(parser['amsgrad']))
				
				if bool(parser['do_lr_decay']):
					if parser['lr_decay'] == 'keras':
						lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = lambda step: keras_lr_decay(step))
		#train phase
		startfull = time.time()
		train_num_iter,accuracy=train_model(model = model,
			device = device,
			db_gen = devset_gen,
			optimizer = optimizer,
			epoch = epoch,
			writer_train = writer_train,
			train_num_iter=train_num_iter,metric_fc=metric_fc)
		endtrain = time.time()
		time_for_training = endtrain - startfull
		print('finish training and the time spent is {}'.format(time_for_training))
		#validation phase
		val_eer = evaluate_model(mode = 'val',
			model = model,
			db_gen = valset_gen, 
			device = device,
			l_utt = l_val,
			save_dir = save_dir,
			epoch = epoch,
			l_trial = l_val_trial)
		f_eer.write('epoch:%d,val_eer:%f\n'%(epoch+1, val_eer))

		eval_eer = evaluate_model(mode = 'eval',
			model = model,
			db_gen = evalset_gen, 
			device = device,
			l_utt = l_eval,
			save_dir = save_dir,
			epoch = epoch,
			l_trial = l_eval_trial)
		endfull = time.time()
		time_for_lastepoch = endfull - startfull
		f_eer.write('epoch:%d,Eval eer:%f\n'%(epoch+1, eval_eer))
		f_eer.write('epoch:%d,accuracy:%f\n'%(epoch+1, accuracy))
		f_eer.write('epoch:%d,time spent:%f\n'%(epoch+1, time_for_lastepoch))
		print('epoch time is {}'.format(time_for_lastepoch))

	f_eer.close()












