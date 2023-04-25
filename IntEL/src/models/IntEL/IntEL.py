'''
Implementation of IntEL
[SIGIR2023] Intent-aware Ranking Ensemble for Personalized Recommendation
'''
import torch.nn as nn
import torch
import torch.nn.functional as F
from models import GeneralSeq
from modules import layers
from modules.attention import *
from utils import utils

class IntEL(GeneralSeq.GeneralSeq):
	extra_log_args = ['cross_attn_qsize','num_heads','num_layers','encoder','intent_emb_size',]

	@staticmethod
	def parse_model_args(parser):
		parser.add_argument('--encoder',type=str,default='BERT4Rec',
				help='A sequence encoder for intent prediction.')
		parser.add_argument('--context_emb_size',type=int,default=16,help='Embedding size for context.')
		parser.add_argument('--i_emb_size',type=int,default=16,help='Embedding size for item id.')
		parser.add_argument('--u_emb_size',type=int,default=32,help='Embedding size for user.')
		parser.add_argument('--s_emb_size',type=int,default=32,help='Embedding size for score.')
		parser.add_argument('--im_emb_size',type=int,default=16,help='Embedding size for item metadata.')
		parser.add_argument('--intent_emb_size',type=int,default=16,help='Embedding size for intent.')
		parser.add_argument('--cross_attn_qsize',type=int,default=32,help='Embedding size for cross-attention query.')
		parser.add_argument('--num_heads', type=int, default=1,
							help='Number of attention heads.')
		parser.add_argument('--dropout', type=float, default=0,
							help='Dropout probability for each deep layer')
		parser.add_argument('--num_layers', type=int, default=1,
							help='Number of self-attention layers.')
		return GeneralSeq.GeneralSeq.parse_model_args(parser)

	def __init__(self,args,corpus):
		super().__init__(args,corpus)
		self.itemfnum = utils.list_product(corpus.itemfnum)
		self.intent_num = len(corpus.zero_int)
		self.model_num = args.model_num

		# item embedding
		self.iid_embeddings = nn.Embedding(self.item_num,args.i_emb_size)
		self.im_emb_size = 0
		if self.itemfnum>0:
			self.item_embeddings = nn.Embedding(self.itemfnum,args.im_emb_size)
			self.im_emb_size = args.im_emb_size
		# user embedding
		self.uid_embeddings = nn.Embedding(self.user_num,args.u_emb_size)
		# intent embedding
		self.intent_embeddings = nn.Linear(self.intent_num,args.intent_emb_size)
		# score embedding
		self.score_embeddings = nn.Linear(args.model_num,args.s_emb_size)

		# self attention (from KDA)
		self.head_num = args.num_heads
		self.layer_num = args.num_layers
		# item self-att
		self.item_emb_size = args.i_emb_size + self.im_emb_size
		self.i_attn_head = layers.MultiHeadAttention(self.item_emb_size,self.head_num,bias=False)
		self.i_W1 = nn.Linear(self.item_emb_size,self.item_emb_size)
		self.i_W2 = nn.Linear(self.item_emb_size,self.item_emb_size)
		self.dropout_layer = nn.Dropout(args.dropout)
		self.i_layer_norm = nn.LayerNorm(self.item_emb_size)

		# score self-att
		self.score_emb_size = args.s_emb_size
		self.s_attn_head = layers.MultiHeadAttention(self.score_emb_size,self.head_num,bias=False)
		self.s_W1 = nn.Linear(self.score_emb_size,self.score_emb_size)
		self.s_W2 = nn.Linear(self.score_emb_size,self.score_emb_size)
		self.s_layer_norm = nn.LayerNorm(self.score_emb_size)

		# cross attention
		self.act_func = nn.ReLU
		self.cross_attn_qsize = args.cross_attn_qsize
		self.intent_score_embeddings = nn.Sequential(
					nn.Linear(self.intent_num, self.cross_attn_qsize),
					self.act_func(),
					nn.Linear(self.cross_attn_qsize, args.s_emb_size, bias=False)
					)
		
		self.intent_item_embeddings = nn.Sequential(
					nn.Linear(self.intent_num, self.cross_attn_qsize),
					self.act_func(),
					nn.Linear(self.cross_attn_qsize, self.item_emb_size, bias=False)
					)
		# weight
		self.weight_embeddings = nn.Linear(self.item_emb_size+args.s_emb_size
									+args.intent_emb_size+args.u_emb_size,args.model_num)

		# context embedding
		self.context_embeddings = nn.Embedding(utils.list_product(corpus.contextfnum),args.context_emb_size)
		# encoder
		self.encoder_name = args.encoder
		self.intent_pred_size = args.intent_emb_size + args.context_emb_size
		self.his_item_dim = args.intent_emb_size + args.i_emb_size
		if self.encoder_name == 'GRU4Rec':
			self.encoder = GeneralSeq.GRU4RecEncoder(self.intent_pred_size,hidden_size=128)
			self.item_encoder = GeneralSeq.GRU4RecEncoder(self.his_item_dim,hidden_size=128)
		elif self.encoder_name == 'BERT4Rec':
			self.encoder = GeneralSeq.BERT4RecEncoder(self.intent_pred_size,self.max_his,num_layers=2,num_heads=2)
			self.item_encoder = GeneralSeq.BERT4RecEncoder(self.his_item_dim,self.max_his,num_layers=2,num_heads=2)
		else:
			raise ValueError('Invalid sequence encoder.')
		self.pred_layer = nn.Linear(
					self.intent_pred_size+self.his_item_dim+
					args.context_emb_size+args.u_emb_size,
					self.intent_num)

	def forward(self,data):
		# predict intents
		intent = self.predict_intent(data)
		# predict ensemble
		weights, ens_scores = self.predict_ensemble(data,intent)

		out_dict = {"weights":weights,"ens_score":ens_scores,"intents":intent}
		return out_dict 
	
	def predict_intent(self,data):
		# load data
		history_context = data['his_context_mh']
		history_intents = data['his_intents'].float()
		lengths = data['history_len']
		current_context = data['context_mh']
		batch_size = data['batch_size']

		# get history encoding	
		his_context_emb = self.context_embeddings(history_context)
		his_intents_emb = self.intent_embeddings(history_intents)
		his_embedding = torch.cat([his_context_emb,his_intents_emb],dim=-1)
		his_vector = self.encoder(his_embedding,lengths)

		# get item history encoding
		his_item_emb = self.iid_embeddings(data['his_item_id'])
		his_item_intents = self.intent_embeddings(data['his_item_int'].float())
		his_item_embedding = torch.cat([his_item_emb,his_item_intents],dim=-1)
		his_item_vector = self.item_encoder(his_item_embedding,data['history_item_len'])

		# other current embeddings
		context_emb = self.context_embeddings(current_context)
		user_emb = self.uid_embeddings(data['u_id_c'])
		
		current_embeddings = torch.cat([context_emb,user_emb],dim=-1) # batch size * intent num * embedding dim

		# predict intent probability
		pred_intents = self.pred_layer(torch.cat([current_embeddings,his_item_vector,his_vector],dim=-1)).softmax(dim=-1)

		return pred_intents


	def predict_ensemble(self,data,intent):
		# load data
		user_id, item_list = data['u_id_c'], data['i_id_s']
		item_metadata, score_list = data['i_class_c'], data['scores'].float()
		lengths = data['session_len']
		valid_mask = (torch.arange(item_list.size(1)).to(self.device)[None,:] < lengths[:,None])
		valid_mask2 = valid_mask.unsqueeze(-1) * valid_mask.unsqueeze(-1).transpose(-1,-2)
		
		# intent embedding
		h_int = intent.unsqueeze(1)

		# item embedding
		h_iid = self.iid_embeddings(item_list) # batch * list_length * i_emb_size
		if item_metadata != None:
			h_im = self.item_embeddings(item_metadata).squeeze() # batch * list_length * im_emb_size
			h_i = torch.cat([h_iid,h_im],dim=2)
		else:
			h_i = h_iid

		# user embedding
		h_u = F.relu(self.uid_embeddings(user_id).unsqueeze(1).repeat(1,h_i.size(1),1)) # batch * u_emb_size

		# self attention
		# item self-att
		for i in range(self.layer_num):
			residual = h_i
			h_i = self.i_attn_head(h_i,h_i,h_i)
			h_i = self.i_W1(h_i)
			h_i = self.i_W2(h_i.relu())
			h_i = self.dropout_layer(h_i)
			h_i = self.i_layer_norm(h_i+residual)
		# score self-att	
		h_s = self.score_embeddings(score_list)
		for i in range(self.layer_num):
			residual = h_s
			h_s = self.s_attn_head(h_s,h_s,h_s)
			h_s = self.s_W1(h_s)
			h_s = self.s_W2(h_s.relu())
			h_s = self.dropout_layer(h_s)
			h_s = self.s_layer_norm(h_s+residual)

		# cross-attention
		item_intent = self.intent_item_embeddings(h_int) # B * 1 * d_v
		score_intent = self.intent_score_embeddings(h_int)
		item_xatt = torch.mul(h_i,item_intent)
		score_xatt = torch.mul(h_s,score_intent)

		# to weight
		h_intent = F.relu(self.intent_embeddings(h_int).repeat(1,h_i.size(1),1))
		all_xatt = torch.cat([item_xatt,score_xatt,h_u,h_intent],dim=-1)
		weights = self.weight_embeddings(all_xatt)
		ens_score = torch.mul(weights,score_list).sum(dim=2)
		
		return weights, ens_score

	class Dataset(GeneralSeq.GeneralSeq.Dataset):
		def _get_feed_dict(self, index):
			feed_dict = super()._get_feed_dict(index)
			item_position = self.data['item_position'][index]
			if item_position:
				history_items = self.corpus.user_itemhis[feed_dict['u_id_c']][:item_position]
				history_behaviors = self.corpus.user_itembehave[feed_dict['u_id_c']][:item_position]
				history_intents = [history_behaviors[i]*self.model.intent_num/self.model.model_num+self.corpus.itemmeta[history_items[i]][0] for i in range(len(history_behaviors))]
				if self.model.max_his>0:
					history_items = history_items[-self.model.max_his:]
					history_intents = history_intents[-self.model.max_his:]
				feed_dict['his_item_id'] = np.array(history_items)
				feed_dict['his_item_int'] = np.zeros([len(history_intents),self.model.intent_num])
				for i,intent in enumerate(history_intents):
					feed_dict['his_item_int'][i,int(intent)] = 1
			else:
				feed_dict['his_item_id'] = np.array([0])
				feed_dict['his_item_int'] = np.zeros([1,self.model.intent_num])
			feed_dict['history_item_len'] = len(feed_dict['his_item_id'])

			return feed_dict