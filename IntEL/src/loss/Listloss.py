import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from loss.BaseIntloss import BaseIntloss

class Listloss(BaseIntloss):

	def __init__(self,args):
		super().__init__(args)
	
	def list_loss(self,loss_matrix,valid_mask,rankings):
		loss_all = torch.exp(-loss_matrix)*valid_mask
		loss_list = ((loss_all.sum(dim=2)+1)*(rankings>0)).clamp(1).log().sum(dim=1) / (rankings>0).sum(dim=-1)
		return loss_list.mean()
	
	def diversity(self,diff_matrix,base_diff,weights,valid_mask,rankings):
		diff_exp = torch.exp(-diff_matrix)
		A_nk_up = (( diff_exp.unsqueeze(3) * (base_diff-diff_matrix.unsqueeze(3)) *valid_mask.unsqueeze(3)).sum(dim=2))**2
		A_nk_w = (weights * A_nk_up).sum(-1)
		A_nk_bo = 2*(1+(diff_exp*valid_mask).sum(dim=2))**2
		diversity_loss = (A_nk_w / A_nk_bo * (rankings>0) ).sum(dim=-1) / (rankings>0).sum(dim=-1)
		return -diversity_loss.mean()

	def forward(self,out_dict,in_batch):
		ens_scores = out_dict['ens_score']
		device = ens_scores.device
		batch_size = in_batch['batch_size']
		valid = (torch.arange(ens_scores.size(1)).to(device)[None,:] < in_batch['session_len'][:,None])
		valid_mask = valid.unsqueeze(2) * valid.unsqueeze(2).transpose(1,2)
		rankings = torch.clamp(in_batch['ranking'],0,max=in_batch['ranking'].max())

		ens_diff = ens_scores.unsqueeze(2) - ens_scores.unsqueeze(2).transpose(1,2)
		diff_mask = (rankings.unsqueeze(2) > rankings.unsqueeze(2).transpose(1,2))*valid_mask
		loss = self.list_loss(ens_diff,diff_mask,rankings)
		
		if self.cal_diversity:
			base_scores = in_batch['scores']
			base_diff = base_scores.unsqueeze(2) - base_scores.unsqueeze(2).transpose(1,2)
			diversity_loss = self.diversity(ens_diff,base_diff,out_dict["weights"],diff_mask,rankings)
			loss += diversity_loss * self.diversity_alpha

		return loss, loss, loss