'''LambdaRank
Reference:
	Christopher Burges, Robert Ragno, and Quoc Le. 2006. Learning to rank with nonsmooth cost functions. 
	Advances in neural information processing systems 19 (2006)
'''

from utils import utils
import numpy as np
import json
import logging
from tqdm import tqdm
import torch
from torch import nn
from models import BaseModel

class LambdaRank(BaseModel.GeneralShuffleModel):
	extra_log_args = []
	
	@staticmethod
	def parse_model_args(parser):
		parser.add_argument('--hidden_size',type=str,default='32')
		parser.add_argument('--i_emb_size',type=int,default=32,help='Embedding size for item id.')
		return BaseModel.GeneralShuffleModel.parse_model_args(parser)

	def __init__(self,args,corpus):
		super().__init__(args,corpus)
		self.iid_embeddings = nn.Embedding(self.item_num,args.i_emb_size)
		n_features = args.model_num + args.i_emb_size + 1 # model num + id embedding + category id
		hidden_sizes = [int(x) for x in args.hidden_size.split(',')]
		self.hidden_sizes = [n_features]+hidden_sizes
		self.mlp = nn.Sequential()
		for i in range(len(self.hidden_sizes)-1):
			self.mlp.add_module('Linear-%d'%(i),
					nn.Linear(self.hidden_sizes[i],self.hidden_sizes[i+1]))
			self.mlp.add_module('Activate-%d'%(i),nn.ReLU())
		self.mlp.add_module('Linear-%d'%(len(self.hidden_sizes)-1),
					nn.Linear(self.hidden_sizes[-1],1))

	def forward(self,data):
		item_list = data['i_id_s']
		item_metadata, score_list = data['i_class_c'], data['scores'].float()
		lengths = data['session_len']
		valid_mask = (torch.arange(item_list.size(1)).to(self.device)[None,:] < lengths[:,None])

		h_iid = self.iid_embeddings(item_list)
		h_input = torch.cat([h_iid,item_metadata.unsqueeze(2),score_list],dim=2)
		ens_scores = self.mlp(h_input).squeeze().softmax(dim=-1)	

		return {"weights":torch.zeros_like(score_list),"ens_score":ens_scores}
