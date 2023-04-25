from email.policy import default
import os
import sys
import json
import argparse
import pickle
import logging
import numpy as np
import pandas as pd
from utils import utils
from collections import Counter

class BaseReader(object):
	@staticmethod
	def parse_data_args(parser):
		parser.add_argument('--datapath', type=str, default='../data/',
							help='Input data dir.')
		parser.add_argument('--dataset', type=str, default='basedata',
							help='Choose a dataset.')
		parser.add_argument('--sep', type=str, default='\t',
							help='sep of csv file.')
		parser.add_argument('--intent_note',type=str,default='')
		parser.add_argument('--max_session_len',type=int,default=40)
		return parser

	def __init__(self,args,transfer_dict=True):
		self.sep = args.sep
		self.prefix = args.datapath
		self.dataset = args.dataset
		self.max_session_len = args.max_session_len
		
		self.cfeatures = ['c_time_i'] # define context features
		self.ifeatures = ['i_class_c',] # define item metadata features
		self.ufeatures = ['u_age_c','u_gender_c'] # define user metadata features
		self.pos_types = ['c_paynum_i','c_favnum_i','c_clicknum_i',] # number of each feedback, sorted by pre-defined ranking
		self.basic_scores = ['c_pCTR_s','c_pCVR_s','c_pFVR_s'] # basic model score features

		self._read_inter()
		self._read_meta()
		self._read_intent(intent_notation=args.intent_note)
		if transfer_dict:
			self._df2dict()

	def _read_inter(self):
		logging.info("Reading data from %s, dataset= %s"%(self.prefix,self.dataset))
		self.interactions = dict()
		self.all_df = []
		max_uid, max_iid = 0,0
		context_f = [set([0]) for f in self.cfeatures]
		for phase in ['train','dev','test']:
			logging.info("Reading data from %s set..."%(phase))
			self.interactions[phase] = pd.read_csv(os.path.join(self.prefix,self.dataset,phase+'.csv'),sep=self.sep)
			self.interactions[phase].sort_values(by=["u_id_c","c_time_i"],inplace=True)
			self.interactions[phase].reset_index(drop=True,inplace=True)
			self.all_df.append(self.interactions[phase])
			max_uid = max(max_uid,self.interactions[phase]["u_id_c"].max())
			for i,f in enumerate(self.cfeatures):
				context_f[i] = context_f[i] | set(self.interactions[phase][f].unique())
			session_lens = []
			for i_id_s in self.interactions[phase]["i_id_s"].tolist():
				i_id_s = eval(i_id_s)
				max_iid = max(max_iid,max(i_id_s))
				session_lens.append(len(i_id_s))
			self.interactions[phase]["session_len"] = session_lens
			logging.info("# session: %d"%(len(self.interactions[phase])))
			
		self.contextfnum =[max(len(c),max(c)+1) for c in context_f]
		self.max_uid, self.max_iid = max_uid, max_iid
		logging.info("#user: %d, #item %d"%(max_uid,max_iid))
		self.all_df = pd.concat(self.all_df,ignore_index=True)

	def _df2dict(self):
		logging.info("Transfer interaction dataframe to dictionary")
		for phase in ['train','dev','test']:
			if phase == 'train': # cut max length for training
				self.interactions[phase] = utils.df2dict(self.interactions[phase],max_session_len = self.max_session_len)
			else:
				self.interactions[phase] = utils.df2dict(self.interactions[phase],max_session_len = -1)
			logging.info('%s set transfer done!'%(phase))

	def _read_meta(self):
		items = json.load(open(os.path.join(self.prefix,self.dataset,"item_metadata.json"))) 
		self.itemmeta = {}
		feature_set = [set([0]) for f in self.ifeatures]
		for key in items: 
			self.itemmeta[int(key)] = np.array([items[key][f] for f in self.ifeatures]).astype(int)
			for i,f in enumerate(self.ifeatures):
				feature_set[i].add(items[key][f]-1)
		self.itemfnum = [max(len(f),max(f)+1) for f in feature_set]
		logging.info("Item metadata: %s"%(str(dict(zip(self.ifeatures,self.itemfnum)))))

		users = json.load(open(os.path.join(self.prefix,self.dataset,"user_metadata.json")))
		self.usermeta = {}
		feature_set = [set([0]) for f in self.ufeatures]
		for key in users:
			self.usermeta[int(key)] = np.array([users[key][f] for f in self.ufeatures]).astype(int)
			for i,f in enumerate(self.ufeatures):
				feature_set[i].add(users[key][f])
		self.userfnum = [max(len(f),max(f)+1) for f in feature_set]	
		logging.info("User metadata: %s"%(str(dict(zip(self.ufeatures,self.userfnum)))))

	def _read_intent(self,intent_notation=""):
		intents = json.load(open(os.path.join(self.prefix,self.dataset,"intents%s.json"%(intent_notation))))
		self.intents = {}
		for key in intents:
			key_name = eval(key)
			self.intents[int(key_name)] = np.array(intents[key])
		self.zero_int = np.zeros(len(self.intents[int(key_name)]))
		self.intentloss_w = np.ones(len(self.intents[int(key_name)]))/len(self.intents[int(key_name)])

