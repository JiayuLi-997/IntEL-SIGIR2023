from email.policy import default
import torch.nn as nn
import torch
import torch.nn.functional as F
from models import BaseModel

class SingleSort(BaseModel.GeneralShuffleModel):
	reader, runner ="BaseReader", "BaseRunner"
	extra_log_args = []
	
	@staticmethod
	def parse_model_args(parser):
		parser.add_argument('--choose_list',type=str,default='pCTR',
							help='Choose pCTR, pCVR, or pFVR for ranking.')
		return BaseModel.GeneralShuffleModel.parse_model_args(parser)
	
	def __init__(self,args, corpus):
		self.choose_list = args.choose_list
		super().__init__(args,corpus)

	def forward(self,data):
		score_list = data['scores'].float()
		if self.choose_list == 'pCTR':
			ens_score = score_list[:,:,0].squeeze()
		elif self.choose_list == 'pCVR':
			ens_score = score_list[:,:,1].squeeze()
		else:
			ens_score = score_list[:,:,2].squeeze()

		return {"weights":torch.zeros_like(score_list),"ens_score":ens_score}