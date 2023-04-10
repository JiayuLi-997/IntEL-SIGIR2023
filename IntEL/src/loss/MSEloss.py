import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from loss.BaseIntloss import BaseIntloss

class MSEloss(BaseIntloss):

	def __init__(self,args):
		super().__init__(args)

	def diversity(self,out_dict,in_batch,valid):
		weights = out_dict['weights']
		ens_scores = out_dict['ens_score']
		base_scores = in_batch['scores']
		diversity_loss = weights * ((base_scores-ens_scores.unsqueeze(2))**2)
		loss = (diversity_loss*valid.unsqueeze(2)).sum(dim=-1).sum(dim=-1) / valid.sum(dim=-1)
		return -loss.mean()

	def forward(self,out_dict,in_batch):
		ens_scores = out_dict['ens_score']
		device = ens_scores.device
		valid = (torch.arange(ens_scores.size(1)).to(device)[None,:] < in_batch['session_len'][:,None])
		rankings = torch.clamp(in_batch['ranking'],0,max=in_batch['ranking'].max())
		loss_list = (((ens_scores - rankings)**2)*valid).sum(dim=-1) / valid.sum(dim=-1)
		loss = loss_list.mean()
		if self.cal_diversity:
			diversity_loss = self.diversity(out_dict,in_batch,valid)
			loss += diversity_loss*self.diversity_alpha
		return loss, loss, loss

