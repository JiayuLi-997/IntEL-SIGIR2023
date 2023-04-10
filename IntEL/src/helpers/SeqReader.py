import os
import sys
import json
import argparse
import pickle
import logging
import numpy as np
import pandas as pd
from utils import utils
import gc

from helpers.BaseReader import BaseReader

class SeqReader(BaseReader):
	def __init__(self,args):
		super().__init__(args,transfer_dict=False)
		self._append_his_info()
		self._df2dict()

	def _append_his_info(self):
		'''
		self.user_his: store user history session sequence, including session id, session features, intents, and position
		'''
		logging.info('Appending history info...')
		sort_df = self.all_df.sort_values(by=['c_time_i','u_id_c'],kind='mergesort')
		cfeature_list = [sort_df[c].tolist() for c in self.cfeatures]
		cid_list, uid_list = sort_df['c_id_c'].tolist(), sort_df['u_id_c'].tolist()
		clicknum, paynum, favnum = sort_df['c_clicknum_i'].tolist(), sort_df['c_paynum_i'].tolist(), sort_df['c_favnum_i'].tolist()
		pos_item_list = [eval(iids)[:(clicknum[i]+paynum[i]+favnum[i])] for i,iids in enumerate(sort_df['i_id_s'].tolist())]

		self.user_his = dict()
		self.user_itemhis = dict()
		self.user_itemsession = dict()
		self.user_itembehave = dict()
		position = list()
		item_position = list()
		for i in range(len(cid_list)):
			uid, cid, cfeatures = uid_list[i], cid_list[i], [flist[i] for flist in cfeature_list]
			pos_items, click, pay, fav = pos_item_list[i], clicknum[i], paynum[i], favnum[i]
			if uid not in self.user_his:
				self.user_his[uid] = list()
				self.user_itemhis[uid] = list()
				self.user_itemsession[uid] = list()
				self.user_itembehave[uid] = list()
			position.append(len(self.user_his[uid]))
			item_position.append(len(self.user_itemhis[uid]))
			self.user_his[uid].append([cid]+cfeatures)
			self.user_itemhis[uid] += pos_items
			self.user_itemsession[uid] += [[cid]+cfeatures]*len(pos_items) # add session information to each item
			self.user_itembehave[uid] += [0]*click + [1]*fav + [2]*pay
		sort_df['position'] = position
		sort_df['item_position'] = item_position

		for key in ['train','dev','test']:
			self.interactions[key] = pd.merge(left=self.interactions[key],
			right=sort_df[["u_id_c","c_time_i","c_id_c","position","item_position"]],how='left',
					on=['u_id_c','c_time_i','c_id_c'])	
		del sort_df
		gc.collect()

