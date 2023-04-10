'''Borda
Reference:
	JC de Borda. 1784. Mémoire sur les élections au scrutin[J]. 
	Histoire de l’Academie Royale des Sciences pour 1781 (Paris, 1784)
'''

import torch.nn as nn
import torch
import torch.nn.functional as F
from models import BaseModel

'''
Average of ranking
'''

class Borda(BaseModel.GeneralShuffleModel):
	reader, runner ="SeqReader", "BaseRunner"
	extra_log_args = []

	def __init__(self,args,corpus,item_f_num=0):
		super().__init__(args,corpus)

	def forward(self,data):
		score_list = data['scores'].float() # batch_size * length * model_num
		ranked_idx = torch.argsort(score_list,dim=1)
		ranking = torch.argsort(ranked_idx,dim=1)
		w = torch.ones_like(score_list).to(self.device) / score_list.size(2)
		ens_score = torch.mul(w,ranking).sum(dim=2)

		return {"weights":w,"ens_score":ens_score}

