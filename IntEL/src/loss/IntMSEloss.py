import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from loss.MSEloss import MSEloss
import numpy as np
import pandas as pd
import logging

class IntMSEloss(MSEloss):

	def __init__(self,args):
		super().__init__(args)
	
	def forward(self,out_dict,in_batch):
		intent_loss, ce_loss, kl_loss = self.get_intloss(out_dict,in_batch)
		ensemble_loss, _, _ = super().forward(out_dict,in_batch)
		loss = ensemble_loss * self.ensemble_weight + intent_loss * self.intent_weight

		return loss, ensemble_loss, intent_loss