'''
Reference:
	https://github.com/THUwangcy/ReChorus
'''
# -*- coding: UTF-8 -*-

import os
import logging
import torch
import json
import datetime
import numpy as np
import pandas as pd

def df2dict(df: pd.DataFrame, df_train=[],max_session_len=-1) -> dict:
	res = df.to_dict('list')
	if len(df_train):
		dict_train = df_train.to_dict('list')
	for key in res:
		if key[-2:]=="_s":
			if max_session_len==-1:
				res[key] = [eval(str(x)) for x in res[key]]
			else:
				res[key] = [eval(str(x))[:max_session_len] for x in res[key]]
		else:
			res[key] = np.array(res[key])
			if key[-2:]=="_c":
				res[key] = res[key].astype(int)
	return res

def check_dir(file_name: str):
	dir_path = os.path.dirname(file_name)
	if not os.path.exists(dir_path):
		print('make dirs:', dir_path)
		os.makedirs(dir_path)

	return

def format_arg_str(args, exclude_lst: list, max_len=20) -> str:
	linesep = os.linesep
	arg_dict = vars(args)
	keys = [k for k in arg_dict.keys() if k not in exclude_lst]
	values = [arg_dict[k] for k in keys]
	key_title, value_title = 'Arguments', 'Values'
	key_max_len = max(map(lambda x: len(str(x)), keys))
	value_max_len = min(max(map(lambda x: len(str(x)), values)), max_len)
	key_max_len, value_max_len = max([len(key_title), key_max_len]), max([len(value_title), value_max_len])
	horizon_len = key_max_len + value_max_len + 5
	res_str = linesep + '=' * horizon_len + linesep
	res_str += ' ' + key_title + ' ' * (key_max_len - len(key_title)) + ' | ' \
			   + value_title + ' ' * (value_max_len - len(value_title)) + ' ' + linesep + '=' * horizon_len + linesep
	for key in sorted(keys):
		value = arg_dict[key]
		if value is not None:
			key, value = str(key), str(value).replace('\t', '\\t')
			value = value[:max_len-3] + '...' if len(value) > max_len else value
			res_str += ' ' + key + ' ' * (key_max_len - len(key)) + ' | ' \
					   + value + ' ' * (value_max_len - len(value)) + linesep
	res_str += '=' * horizon_len
	return res_str

def get_time():
	return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def format_metric(result_dict) -> str:
	assert type(result_dict) == dict
	record_metrics = []
	format_str = []
	metrics = np.unique([k.split('@')[0] for k in result_dict.keys()])
	topks = np.unique([int(k.split('@')[1]) for k in result_dict.keys() if "@" in k])
	for topk in np.sort(topks):
		for metric in np.sort(metrics):
			name = '{}@{}'.format(metric, topk)
			if name in result_dict:
				m = result_dict[name]
			else: # point wise metrics
				if metric in result_dict:
					m = result_dict[metric]
					name=metric
				else:
					continue
			if name in record_metrics:
				continue
			if type(m) is float or type(m) is np.float or type(m) is np.float32 or type(m) is np.float64:
				format_str.append('{}:{:<.4f}'.format(name, m))
			elif type(m) is int or type(m) is np.int or type(m) is np.int32 or type(m) is np.int64:
				format_str.append('{}:{}'.format(name, m))
			record_metrics.append(name)
	return ','.join(format_str)

def batch_to_gpu(batch: dict, device) -> dict:
	for c in batch:
		if type(batch[c]) is torch.Tensor:
			batch[c] = batch[c].to(device)
	return batch

def list_product(L):
	p = 1
	for i in L:
		p *= i
	return p
