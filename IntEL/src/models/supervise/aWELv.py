'''aWELv
Reference:
	Hongzhi Liu, Yingpeng Du, and Zhonghai Wu. 2022. Generalized Ambiguity Decomposition for Ranking Ensemble Learning. 
	Journal of Machine Learning, Research 23, 88 (2022), 1–36.
'''
import torch.nn as nn
import torch
import torch.nn.functional as F
from models import BaseModel

class aWELv(BaseModel.GeneralShuffleModel):
	reader, runner ="BaseReader", "BaseRunner"
	extra_log_args = []
	
	@staticmethod
	def parse_model_args(parser):
		parser.add_argument('--hidden_size',type=int,default=32)
		return BaseModel.GeneralShuffleModel.parse_model_args(parser)
		

	def __init__(self,args, corpus):
		super().__init__(args,corpus)
		self.uid_embeddings = torch.nn.Embedding(self.user_num,args.hidden_size)
		self.model_embeddings = torch.nn.Embedding(args.model_num,args.hidden_size)

	def forward(self,data):
		score_list, intent = data['scores'].float(), data['intents'].float()
		user_id = data['u_id_c']
		model_num = score_list.size(2) # batch * list_length * base_num

		h_u = self.uid_embeddings(user_id) # batch * hidden_size
		weights = []
		for m in range(model_num):
			h_m = self.model_embeddings(torch.LongTensor([m]).to(self.device)) # 1 * hidden_size
			weights.append(torch.mul(h_u,h_m.squeeze()[None,:]).sum(dim=1).unsqueeze(1)) # batch * hidden_size
		w = torch.cat(weights,dim=1).unsqueeze(1).repeat(1,score_list.size(1),1).softmax(dim=-1)
		ens_score = torch.mul(w,score_list).sum(dim=2) # batch * list_length

		out_dict = {"weights":w,"ens_score":ens_score}
		return out_dict 