'''
Add Predicted Intent to aWELv
'''
import torch.nn as nn
import torch
import torch.nn.functional as F
from models import BaseModel
from models import GeneralSeq
from modules import layers
from modules.attention import *
from utils import utils

class aWELv_Int(GeneralSeq.GeneralSeq):
	reader="SeqReader"
	extra_log_args = ['user_emb_size','intent_emb_size']

	@staticmethod
	def parse_model_args(parser):
		parser.add_argument('--context_emb_size',type=int,default=16,help='Embedding size for context.')
		parser.add_argument('--user_emb_size',type=int,default=16)
		parser.add_argument('--intent_emb_size',type=int,default=16)
		
		parser.add_argument('--encoder',type=str,default='BERT4Rec',
				help='A sequence encoder for intent prediction.')
		parser.add_argument('--i_emb_size',type=int,default=16,help='Embedding size for item id.')
		parser.add_argument('--im_emb_size',type=int,default=16,help='Embedding size for item metadata.')
		return GeneralSeq.GeneralSeq.parse_model_args(parser)
	
	def __init__(self,args, corpus,item_f_num=0):
		super().__init__(args,corpus)
		self.intent_num = len(corpus.zero_int)
		self.itemfnum = utils.list_product(corpus.itemfnum)
		self.model_num = args.model_num
		self.uid_embeddings = torch.nn.Embedding(self.user_num,args.user_emb_size)
		self.intent_embeddings = torch.nn.Linear(self.intent_num,args.intent_emb_size)
		self.hidden_size = args.user_emb_size + args.intent_emb_size
		self.model_embeddings = torch.nn.Embedding(args.model_num,self.hidden_size)
		
		# item embedding
		self.iid_embeddings = nn.Embedding(self.item_num,args.i_emb_size)
		self.im_emb_size = 0
		self.itemfnum = utils.list_product(corpus.itemfnum)
		if self.itemfnum>0:
			self.item_embeddings = nn.Embedding(self.itemfnum,args.im_emb_size)
			self.im_emb_size = args.im_emb_size
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
					args.context_emb_size+args.user_emb_size,#+args.intent_emb_size+args.intent_emb_size
					self.intent_num)

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
		context_emb = self.context_embeddings(current_context)#.unsqueeze(1).repeat(1,history_intents.size(-1),1) # batchsize * intent num * embedding dim
		user_emb = self.uid_embeddings(data['u_id_c'])#.unsqueeze(1).repeat(1,history_intents.size(-1),1) # batch size * embedding dim
		
		current_embeddings = torch.cat([context_emb,user_emb],dim=-1) # batch size * intent num * embedding dim

		# predict intent probability
		pred_intents = self.pred_layer(torch.cat([current_embeddings,his_item_vector,his_vector],dim=-1)).softmax(dim=-1)

		return pred_intents

	def forward(self,data):
		user_id, item_list = data['u_id_c'], data['i_id_s']
		item_metadata, score_list = data['i_class_c'], data['scores'].float()#, data['intents'].float()
		model_num = score_list.size(2) # batch * list_length * base_num
		intent = self.predict_intent(data)

		h_u = self.uid_embeddings(user_id) # batch * hidden_size
		h_int = self.intent_embeddings(intent) # batch * intent_emb
		h_context = torch.cat([h_u,h_int],dim=1)
		weights = []
		for m in range(model_num):
			h_m = self.model_embeddings(torch.LongTensor([m]).to(self.device)) # 1 * hidden_size
			weights.append(torch.mul(h_context,h_m.squeeze()[None,:]).sum(dim=1).unsqueeze(1)) # batch * hidden_size
		w = F.softmax(torch.cat(weights,dim=1),dim=1).unsqueeze(1).repeat(1,score_list.size(1),1)
		ens_score = torch.mul(w,score_list).sum(dim=2) # batch * list_length

		out_dict = {"weights":w,"ens_score":ens_score,"intents":intent}
		return out_dict 
	
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