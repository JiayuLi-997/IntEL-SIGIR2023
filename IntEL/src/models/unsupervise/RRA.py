'''RRA
Reference:
	Kolde R, Laur S, Adler P, et al. Robust rank aggregation for gene list integration and meta-analysis[J]. 
	Bioinformatics, 2012, 28(4): 573-580.
'''

import torch
import logging
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from scipy.stats import beta

from models import BaseModel
from utils import utils

class RRA(BaseModel.GeneralShuffleModel):
	reader, runner ="BaseReader", "BaseRunner"
	extra_log_args = []

	def __init__(self,args,corpus):
		super().__init__(args,corpus)
		self.model_num = args.model_num
	
	def forward(self, data):
		score_list= data['scores'].float() # batch size * model num * length
		session_len = data['session_len']
		# score list +
		for batch in range(score_list.shape[0]):
			score_list[batch,:session_len[batch],:] += 1e-4
		ranked_idx = torch.argsort(score_list,dim=1,descending=True)
		rankings = torch.argsort(ranked_idx,dim=1) + 1 # larger score, smaller ranking
		rankings_norm = rankings / session_len[:,None,None] # normalize

		beta_distribution = self.get_binomial_prob(rankings_norm, session_len, rankings)
		score_rra = beta_distribution.min(axis=-1)[0] * self.model_num # smaller rra indicates better results
		score_ens = score_rra

		return {"weights":torch.zeros_like(score_list),"ens_score":score_ens}
	
	def get_binomial_prob(self, rankings_norm_t, session_lens_t, rankings_t):
		rankings_norm = rankings_norm_t.cpu().numpy()
		session_lens = session_lens_t.cpu().numpy()
		rankings = rankings_t.cpu().numpy()
		rankings_new = np.zeros_like(rankings)
		rankings_new_list = []
		batch_size, max_length, model_num = rankings.shape
		for batch in range(batch_size):
			L = []
			for pos in range(max_length):
				if pos < session_lens[batch]:
					L2 = []
					for mo in range(model_num):
						beta_prob = self.beta_calculator(rankings_norm[batch,pos,mo],session_lens[batch],rankings[batch,pos,mo])
						rankings_new[batch,pos,mo] = beta_prob
						L2.append(beta_prob)
					L.append(L2)
				else:
					L.append([0]*model_num)
			rankings_new_list.append(L)
		rankings_new_np = np.array(rankings_new_list)
		return torch.Tensor(rankings_new_np).to(self.device)
	
	def beta_calculator(self,x,n,k):
		if x==0 or n == k:
			return 1
		return beta.cdf(x,k,n-k)

