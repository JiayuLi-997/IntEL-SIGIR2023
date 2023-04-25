import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from loss.BaseIntloss import BaseIntloss

class BPRloss(BaseIntloss):

	def __init__(self,args):
		super().__init__(args)

	def diversity(self,ens_diff,base_diff,weights,select_mask,rankings):
		# calculate diversity for BPR loss
		sig_z = ens_diff.sigmoid() * (1-ens_diff.sigmoid())
		z_diff = ( sig_z.unsqueeze(3) * (base_diff - ens_diff.unsqueeze(3))**2 * select_mask.unsqueeze(3) ).sum(dim=2)
		A_mn = (z_diff * weights).sum(dim=-1) * (rankings>0)
		loss = A_mn.sum(dim=-1) / (rankings>0).sum(dim=-1)
		return -loss.mean()

	def bpr_loss(self,ens_diff,diff_mask,rankings):
		max_rank = diff_mask.max()

		rank_similarity = (max_rank+1-diff_mask)*(diff_mask>0)
		select_similarity = rank_similarity.max(dim=-1)[0]
		possible_mask = ((rank_similarity==select_similarity[:,:,None])*(diff_mask>0)).int()
		random_mask = torch.rand_like(possible_mask,dtype=torch.float32)/10

		select_index = (possible_mask+random_mask).argmax(dim=-1)
		select_mask = F.one_hot(select_index)
		select_mask = F.pad(select_mask,(0,ens_diff.size(-1)-select_mask.size(-1)),'constant',0)

		loss_item = (- ens_diff.sigmoid().log() * select_mask).sum(dim=-1) * (rankings>0)
		loss_list = loss_item.sum(dim=-1) / (rankings>0).sum(dim=-1) # average of the list
		return loss_list.mean(), select_mask


	def forward(self,out_dict,in_batch):
		ens_scores = out_dict['ens_score']
		device = ens_scores.device
		batch_size = in_batch['batch_size']
		valid = (torch.arange(ens_scores.size(1)).to(device)[None,:] < in_batch['session_len'][:,None])
		valid_mask = valid.unsqueeze(2) * valid.unsqueeze(2).transpose(1,2)
		rankings = torch.clamp(in_batch['ranking'],0,max=in_batch['ranking'].max())

		ens_diff = ens_scores.unsqueeze(2) - ens_scores.unsqueeze(2).transpose(1,2)
		diff_mask = ((rankings.unsqueeze(2) - rankings.unsqueeze(2).transpose(1,2)))*valid_mask

		loss, select_mask = self.bpr_loss(ens_diff,diff_mask,rankings)

		if self.cal_diversity:
			base_scores = in_batch['scores']
			base_diff = base_scores.unsqueeze(2) - base_scores.unsqueeze(2).transpose(1,2)
			diversity_loss = self.diversity(ens_diff,base_diff,out_dict['weights'],select_mask,rankings)
			loss += diversity_loss*self.diversity_alpha
		
		return loss, loss, loss
			
