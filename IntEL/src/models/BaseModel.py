import logging
from pyexpat import model
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset as BaseDataset
from helpers.BaseReader import BaseReader
from utils import utils
from tqdm import tqdm
import numpy as np
import pandas as pd
import gc


class BaseModel(nn.Module):
	reader="BaseReader"
	extra_log_args = []

	@staticmethod
	def parse_model_args(parser):
		parser.add_argument('--model_path', type=str, default='',
							help='Model save path.')
		parser.add_argument('--buffer', type=int, default=1,
							help='Whether to buffer feed dicts for dev/test')
		parser.add_argument('--model_num',type=int,default=2,help='Number of base models.')
		
		return parser

	@staticmethod
	def init_weights(m):
		if 'Linear' in str(type(m)):
			nn.init.normal_(m.weight, mean=0.0, std=0.01)
			if m.bias is not None:
				nn.init.normal_(m.bias, mean=0.0, std=0.01)
		elif 'Embedding' in str(type(m)):
			nn.init.normal_(m.weight, mean=0.0, std=0.01)

	def __init__(self,args, corpus: BaseReader):
		super(BaseModel, self).__init__()
		self.intent_num = len(corpus.zero_int)
		self.device = args.device
		self.model_path = args.model_path
		self.buffer = args.buffer
		self.optimizer = None
		self.check_list = list()

	def forward(self, data):
		pass

	"""
	Auxiliary Methods
	"""
	def customize_parameters(self,define_dict={}) -> list:
		# customize optimizer settings for different parameters
		weight_p, bias_p = [], []
		for name, p in filter(lambda x: x[1].requires_grad, self.named_parameters()):
			if 'bias' in name:
				bias_p.append(p)
			else:
				weight_p.append(p)
		optimize_dict = [{'params': weight_p}, {'params': bias_p, 'weight_decay': 0}]
		return optimize_dict

	def save_model(self, model_path=None):
		if model_path is None:
			model_path = self.model_path
		utils.check_dir(model_path)
		torch.save(self.state_dict(), model_path)

	def load_model(self, model_path=None):
		if model_path is None:
			model_path = self.model_path
		self.load_state_dict(torch.load(model_path))
		logging.info('Load model from ' + model_path)

	def count_variables(self) -> int:
		total_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
		return total_parameters

	"""
	Define Dataset Class
	"""	
	class Dataset(BaseDataset):
		def __init__(self,model,corpus,phase:str):
			self.model = model
			self.corpus = corpus
			self.phase = phase
			self.buffer_dict = dict()
			self.data = corpus.interactions[phase]

		def __len__(self):
			if type(self.data) == dict:
				for key in self.data:
					return len(self.data[key])
			return len(self.data)

		def __getitem__(self, index: int) -> dict:
			if self.model.buffer and self.phase != 'train':
				return self.buffer_dict[index]
			return self._get_feed_dict(index)

		# ! Key method to construct input data for a single instance
		def _get_feed_dict(self, index: int) -> dict:
			pass

		# Called after initialization
		def prepare(self):
			if self.model.buffer and self.phase != 'train':
				for i in tqdm(range(len(self)), leave=False, desc=('Prepare ' + self.phase)):
					self.buffer_dict[i] = self._get_feed_dict(i)
				
				for key in ['i_id_s']+self.corpus.basic_scores:
					self.data.pop(key)
				gc.collect()

		# Collate a batch according to the list of feed dicts
		def collate_batch(self, feed_dicts) -> dict:
			feed_dict = dict()
			for key in feed_dicts[0]:
				if isinstance(feed_dicts[0][key], np.ndarray):
					tmp_list = [len(d[key]) for d in feed_dicts]
					if any([tmp_list[0] != l for l in tmp_list]):
						stack_val = np.array([d[key] for d in feed_dicts], dtype=np.object)
					else:
						stack_val = np.array([d[key] for d in feed_dicts])
				else:
					stack_val = np.array([d[key] for d in feed_dicts])
				if stack_val.dtype == np.object:  # inconsistent length (e.g., history)
					feed_dict[key] = nn.utils.rnn.pad_sequence([torch.from_numpy(x) for x in stack_val], batch_first=True)
				else:
					feed_dict[key] = torch.from_numpy(stack_val)
			
			feed_dict['scores'] = torch.stack([feed_dict[basic] for basic in self.corpus.basic_scores],dim=2)
			for basic in self.corpus.basic_scores:
				feed_dict.pop(basic)
			feed_dict['batch_size'] = len(feed_dicts)
			feed_dict['phase'] = self.phase
			return feed_dict
		
		def actions_after_epoch(self):
			pass


class GeneralModel(BaseModel):
	reader="BaseReader"
	extra_log_args = []

	def __init__(self,args,corpus):
		super().__init__(args,corpus)
		self.user_num = int(corpus.max_uid+1)
		self.item_num = int(corpus.max_iid+1)

	class Dataset(BaseModel.Dataset):
		def _get_feed_dict(self,index):
			uid = self.data['u_id_c'][index]
			feed_dict = {}
			for key in ['u_id_c','c_id_c']+self.corpus.pos_types:
				feed_dict[key] = self.data[key][index]
			feed_dict['context_mh'], feed_dict['user_mh'] = 0, 0
			for i,key in enumerate(self.corpus.cfeatures): # context features
				feed_dict['context_mh'] = feed_dict['context_mh']*self.corpus.contextfnum[i] + self.data[key][index]
			for i,key in enumerate(self.corpus.ufeatures): # user features
				feed_dict['user_mh'] = feed_dict['user_mh']*self.corpus.userfnum[i] + self.corpus.usermeta[uid][i]
			for i,key in enumerate(self.corpus.ifeatures): # item features
				feed_dict[key] = np.array([self.corpus.itemmeta[iid][i] for iid in self.data['i_id_s'][index]])
			for i,key in enumerate(['i_id_s']+self.corpus.basic_scores): # sequence of item and base scores
				feed_dict[key] = np.array(self.data[key][index])
				if key != 'i_id_s': # normalize the scores
					feed_dict[key] = (feed_dict[key] - feed_dict[key].min()) / (feed_dict[key].max()-feed_dict[key].min()+1e-6)
			feed_dict['session_len'] = len(feed_dict['i_id_s'])
			feed_dict['intents'] = self.corpus.intents.get(feed_dict['c_id_c'],self.corpus.zero_int)
			
			# get the rankings
			max_ranking = len(self.corpus.pos_types)
			feed_dict['ranking'] = []
			for type_idx, pos_type in enumerate(self.corpus.pos_types):
				feed_dict['ranking'] += [max_ranking-type_idx]*feed_dict[pos_type]
			feed_dict['ranking'] += [0]*self.data['c_trueneg_i'][index] 
			feed_dict['ranking'] = np.array(feed_dict['ranking'] + [-1]*(feed_dict['session_len']-len(feed_dict['ranking']))) # padding the rankings
			if len(feed_dict['ranking']) > feed_dict['session_len']: # limited by max_session_len
				feed_dict['ranking'] = feed_dict['ranking'][:feed_dict['session_len']]
			return feed_dict
				

class GeneralShuffleModel(GeneralModel):
	
	class Dataset(GeneralModel.Dataset):
		def _get_feed_dict(self, index: int) -> dict:
			feed_dict = super()._get_feed_dict(index)
			idxs = np.random.choice(np.arange(feed_dict['session_len']),feed_dict['session_len'],replace=False).astype(int)
			for i,key in enumerate(['i_id_s','ranking',]+self.corpus.basic_scores+self.corpus.ifeatures): # shuffle all sequential features
				feed_dict[key] = feed_dict[key][idxs]
			return feed_dict
