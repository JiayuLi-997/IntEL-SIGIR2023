'''
Reference:
	Samuel Oliveira, Victor Diniz, Anisio Lacerda, and Gisele L Pappa. 2016. Evolu-tionary rank aggregation for recommender systems. 
	IEEE Congress on Evolutionary Computation (CEC). IEEE, 255–262.
'''
import torch
from torch import nn
import torch.nn.functional as F
from modules.attention import *
from modules import layers
from models import BaseModel
from utils import utils

class ERA(BaseModel.GeneralShuffleModel):
	reader, runner ="BaseReader", "BaseRunner"
	extra_log_args = ['hidden_sizes']
	
	@staticmethod
	def parse_model_args(parser):
		parser.add_argument('--window_size',type=int,default=10,help='mAgr Window size.')
		parser.add_argument('--hidden_sizes',type=str,default='16')
		return BaseModel.GeneralShuffleModel.parse_model_args(parser)
	
	def __init__(self,args,corpus):
		super().__init__(args,corpus)
		self.model_num = args.model_num
		self.window_size = args.window_size
		n_features = 5
		hidden_sizes = [int(x) for x in args.hidden_sizes.split(",")]
		self.hidden_sizes = [n_features]+hidden_sizes
		self.mlp = nn.Sequential()
		for i in range(len(self.hidden_sizes)-1):
			self.mlp.add_module('Linear-%d'%(i),
					nn.Linear(self.hidden_sizes[i],self.hidden_sizes[i+1]))
			self.mlp.add_module('Activate-%d'%(i),nn.ReLU())
		self.mlp.add_module('Linear-%d'%(len(self.hidden_sizes)-1),
					nn.Linear(self.hidden_sizes[-1],1))
	
	def forward(self, data):
		features = [data['p10'],data['mAgr']]
		for m in range(self.model_num):
			features.append(data['psc_%d'%(m)])
		features = torch.stack(features,dim=2).float()
		ens_scores = self.mlp(features).squeeze()
		return ens_scores

	
	class Dataset(BaseModel.GeneralShuffleModel.Dataset):
		
		def __init__(self, model, corpus, phase):
			super().__init__(model,corpus,phase)
			self.window_size = model.window_size

		def _get_feed_dict(self, index):
			feed_dict = super()._get_feed_dict(index)
			all_rankings = []
			for i, key in enumerate(['c_pCTR_s','c_pCVR_s','c_pFVR_s']):
				ranked_idx = np.argsort(feed_dict[key])[::-1]
				rankings = np.argsort(ranked_idx) + 1 # larger score, smaller ranking
				psc = 1- (rankings-1)/len(rankings)
				feed_dict['psc_%d'%(i)] = psc
				all_rankings.append(rankings)
			all_rankings = np.array(all_rankings)
			feed_dict['p10'] = (all_rankings<=10).sum(axis=0)
			feed_dict['mAgr'] = 1/2 * (np.abs(all_rankings[1]-all_rankings[0])<=self.window_size)

			return feed_dict