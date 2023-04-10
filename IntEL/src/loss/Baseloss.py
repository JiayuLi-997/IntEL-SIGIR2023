from sklearn import ensemble
import torch
import torch.nn as nn
import torch.nn.functional as F

class Baseloss(nn.Module):

	@staticmethod
	def parse_loss_args(parser):
		parser.add_argument('--cal_diversity',type=int,default=0)
		parser.add_argument('--diversity_alpha',type=float,default=0.01)
		return parser

	def __init__(self,args):
		self.cal_diversity = args.cal_diversity
		self.diversity_alpha = args.diversity_alpha
		super().__init__()
	
	def forward(self,out_dict,in_batch):
		pass