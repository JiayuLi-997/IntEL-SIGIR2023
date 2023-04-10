from email.policy import default
import numpy as np
import pandas as pd
import torch.nn as nn
import torch
import torch.nn.functional as F
from models import BaseModel
from modules import layers

class GeneralSeq(BaseModel.GeneralShuffleModel):
	reader, runner ="SeqReader", "BaseRunner"
	extra_log_args = []

	@staticmethod
	def parse_model_args(parser):
		parser.add_argument('--history_max',type=int,default=20)
		return BaseModel.GeneralShuffleModel.parse_model_args(parser)

	def __init__(self,args,corpus):
		self.max_his = args.history_max
		super().__init__(args,corpus)
	
	def forward(self, data):
		user_id= data['u_id_c']
		device = user_id.device
		score_list = data['scores'].float().to(device)
		weights = torch.rand(score_list.shape).to(device)
		w = F.softmax(weights,dim=2)
		ens_score = torch.mul(w,score_list).sum(dim=2) # batch * list_length

		out_dict = {"weights":w,"ens_score":ens_score}
		return out_dict
	
	class Dataset(BaseModel.GeneralShuffleModel.Dataset):
		def _get_feed_dict(self, index):
			feed_dict = super()._get_feed_dict(index)
			# add historical session intents, session context
			position = self.data['position'][index]
			if position:
				history = self.corpus.user_his[feed_dict['u_id_c']][:position]
				if self.model.max_his>0:
					history = history[-self.model.max_his:]
				feed_dict['his_intents'] = np.array([self.corpus.intents[his[0]] for his in history])
				feed_dict['his_context_mh'] = [0 for i in range(position)]
				for i,key in enumerate(self.corpus.cfeatures):
					feed_dict['his_context_mh'] = [feed_dict['his_context_mh'][his]*self.corpus.contextfnum[i] + history[his][i+1] for his in range(len(history))] # i+1: skip c_id_c
				feed_dict['his_context_mh'] = np.array(feed_dict['his_context_mh'])
			else: # without history
				feed_dict['his_intents'] = np.zeros([1,self.model.intent_num])
				feed_dict['his_context_mh'] = np.array([0])
			feed_dict['position'] = position
			feed_dict['history_len'] = len(feed_dict['his_context_mh'])
			feed_dict['intentloss_w'] = self.corpus.intentloss_w
			return feed_dict


""" Encoder Layers """
class GRU4RecEncoder(nn.Module):
	def __init__(self,emb_size, hidden_size=128):
		super().__init__()
		self.rnn = nn.GRU(input_size=emb_size, hidden_size=hidden_size, batch_first=True)
		self.out = nn.Linear(hidden_size, emb_size, bias=False)

	def forward(self, seq, lengths):
		# Sort and Pack
		sort_lengths, sort_idx = torch.topk(lengths, k=len(lengths))
		sort_seq = seq.index_select(dim=0, index=sort_idx)
		seq_packed = torch.nn.utils.rnn.pack_padded_sequence(sort_seq, sort_lengths.cpu(), batch_first=True)

		# RNN
		output, hidden = self.rnn(seq_packed, None)

		# Unsort
		sort_rnn_vector = self.out(hidden[-1])
		unsort_idx = torch.topk(sort_idx, k=len(lengths), largest=False)[1]
		rnn_vector = sort_rnn_vector.index_select(dim=0, index=unsort_idx)

		return rnn_vector

class BERT4RecEncoder(nn.Module):
	def __init__(self, emb_size, max_his, num_layers=2, num_heads=2):
		super().__init__()
		self.p_embeddings = nn.Embedding(max_his + 1, emb_size)
		self.transformer_block = nn.ModuleList([
			layers.TransformerLayer(d_model=emb_size, d_ff=emb_size, n_heads=num_heads)
			for _ in range(num_layers)
		])

	def forward(self, seq, lengths):
		batch_size, seq_len = seq.size(0), seq.size(1)
		len_range = torch.from_numpy(np.arange(seq_len)).to(seq.device)
		valid_mask = len_range[None, :] < lengths[:, None]

		# Position embedding
		position = len_range[None, :] * valid_mask.long()
		pos_vectors = self.p_embeddings(position)
		seq = seq + pos_vectors

		# Self-attention
		attn_mask = valid_mask.view(batch_size, 1, 1, seq_len)
		for block in self.transformer_block:
			seq = block(seq, attn_mask)
		seq = seq * valid_mask[:, :, None].float()

		his_vector = seq[torch.arange(batch_size), lengths - 1]
		return his_vector
