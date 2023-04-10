import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from loss.Baseloss import Baseloss
import numpy as np
import pandas as pd
import logging

class BaseIntloss(Baseloss):

	@staticmethod
	def parse_loss_args(parser):
		parser.add_argument('--intent_weight',type=float,default=0.1,
					help='Weight for intent loss.')
		parser.add_argument('--ensemble_weight',type=float,default=1,
					help='Weight for ensemble loss.')
		parser.add_argument('--kl_temp',type=float,default=2)
		parser.add_argument('--kl_weight',type=float,default=0.5)
		return Baseloss.parse_loss_args(parser)
	
	def __init__(self,args):
		self.intent_weight = args.intent_weight
		self.ensemble_weight = args.ensemble_weight
		self.kl_weight, self.T = args.kl_weight, args.kl_temp

		super().__init__(args)
		self.loss_kl = nn.KLDivLoss(reduction="none")

	def ce_loss(self,true_labels,predict_labels,weights):
		# true_labels, predict_labels: batch size * class num
		if predict_labels.min()==0:
			# if zero exists, make soft
			predict_soft = (predict_labels+1e-6)
			predict_soft /= (predict_soft.sum(dim=-1)[:,None])
		else:
			predict_soft = predict_labels
		positive_masks = true_labels > 0
		negative_masks = true_labels == 0
		positive_loss = positive_masks * true_labels * predict_soft.log() 
		negative_loss = negative_masks * (1-predict_soft).log()
		loss = -( ( positive_loss + negative_loss ) * weights).sum(dim=-1)
		return loss.mean()
	
	def kl_loss(self,true_labels,predict_labels,weights):
		# true_labels, predict_labels: batch size * class num
		if predict_labels.min()==0:
			# if zero exists, make soft
			predict_soft = (predict_labels+1e-6)
			predict_soft /= (predict_soft.sum(dim=-1)[:,None])
		else:
			predict_soft = predict_labels
		loss = self.loss_kl(predict_soft.log(),true_labels).double()
		loss_mean = (loss*weights).sum(dim=-1)
		return loss_mean.mean()
	
	def get_intloss(self,out_dict,in_batch):
		batch_size = in_batch['intents'].size(0)
		ensemble_loss = super().forward(out_dict,in_batch)
		intent_num = in_batch['intents'].size(-1)
		labels = in_batch['intents'].float()
		masks = torch.ones(batch_size).to(in_batch['intents'].device).float()
		ce_loss = self.ce_loss(in_batch['intents'],out_dict['intents'],masks.unsqueeze(1).repeat(1,intent_num))
		kl_loss =  self.kl_loss(labels,out_dict['intents'],
						masks.unsqueeze(1).repeat(1,intent_num))* self.T * self.T
		intent_loss = ce_loss*(1-self.kl_weight) + kl_loss*self.kl_weight
		return intent_loss, ce_loss,kl_loss


	def forward(self,out_dict,in_batch):
		pass