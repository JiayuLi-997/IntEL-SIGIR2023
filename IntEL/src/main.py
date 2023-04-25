# -*- coding: UTF-8 -*-
import os
import sys
import pickle
import logging
import argparse
import time
from tqdm import tqdm
import numpy as np
from sklearn.utils import shuffle
import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

from helpers import *
from models import *
from models.unsupervise import *
from models.supervise import *
from models.IntEL import *
from imp import reload
from utils import utils
from loss import *

def parse_global_args(parser):
	parser.add_argument('--gpu', type=str, default='',
						help='Set CUDA_VISIBLE_DEVICES, default for CPU only')
	parser.add_argument('--verbose', type=int, default=logging.INFO,
						help='Logging Level, 0, 10, ..., 50')
	parser.add_argument('--log_file', type=str, default='',
						help='Logging file path')
	parser.add_argument('--random_seed', type=int, default=0,
						help='Random seed of numpy and pytorch')
	parser.add_argument('--load', type=int, default=0,
						help='Whether load model and continue to train')
	parser.add_argument('--train', type=int, default=1,
						help='To train the model or not.')
	parser.add_argument('--regenerate', type=int, default=0,
						help='Whether to regenerate intermediate files')
	parser.add_argument('--save_anno',type=str,default='test',help='Annotation for saving files.')
	parser.add_argument('--test_train',type=int,default=0)
	return parser


def main():
	logging.info('-' * 45 + ' BEGIN: ' + utils.get_time() + ' ' + '-' * 45)
	exclude = ['check_epoch', 'log_file', 'model_path', 'path', 'pin_memory', 'load',
				'regenerate', 'sep', 'train', 'verbose', 'metric', 'test_epoch', 'buffer']
	logging.info(utils.format_arg_str(args, exclude_lst=exclude))
	
	# Random seed
	np.random.seed(args.random_seed)
	torch.manual_seed(args.random_seed)
	torch.cuda.manual_seed(args.random_seed)
	torch.backends.cudnn.deterministic = True

	# GPU
	os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
	args.device = torch.device('cpu')
	if args.gpu != '' and torch.cuda.is_available():
		args.device = torch.device('cuda')
	logging.info('Device: {}'.format(args.device))

	# Read data
	corpus_path = os.path.join(args.datapath, args.dataset, model_name.reader + '_%d%s.pkl'%(args.max_session_len,args.intent_note))
	if not args.regenerate and os.path.exists(corpus_path):
		logging.info('Load corpus from {}'.format(corpus_path))
		corpus = pickle.load(open(corpus_path, 'rb'))
		corpus.pos_types = ['c_paynum_i','c_favnum_i','c_clicknum_i',]
	else:
		corpus = reader_name(args)
		logging.info('Save corpus to {}'.format(corpus_path))
		pickle.dump(corpus, open(corpus_path, 'wb'))

	# Define model
	try:
		model = model_name(args,corpus).to(args.device) 
	except:
		model = model_name(args,corpus) # some models not support gpu
		logging.info('Warning: No convert function for model.')
	logging.info('#params: {}'.format(model.count_variables()))
	logging.info(model)
	criterion = loss_name(args)

	# load model
	runner = runner_name(args) 
	if args.load > 0:
		logging.info("Load model from %s..."%(model.model_path))
		model.load_model()
	
	# Define DataLoader
	data_dict = dict()
	for phase in ['train','dev','test']:
		data_dict[phase] = model_name.Dataset(model,corpus,phase)
		data_dict[phase].prepare()
		logging.info("%s prepare done!"%(phase))

	# train!	
	if args.train > 0:
		logging.info("Start Training!")
		runner.train(data_dict,criterion,save_anno = args.save_anno)
	else:
		logging.info("[Warning] No training!")

	# evaluate
	logging.info("Final evaluation!")
	if args.test_train:
		phase_list=['train','dev','test']
	else:
		phase_list=['dev','test']
	for phase in phase_list:
		sample = True if phase=='train' else False # sample part of train for evaluating
		try:
			test_loss, test_evals = runner.evaluate(data_dict[phase],runner.topk,runner.metrics,criterion, 
							sample=sample,show_num=True,phase=phase)
			logging.info("%s loss= %.4f, metrics: %s "%(phase,test_loss,utils.format_metric(test_evals)))
		except: # no criterion for Lambda Rank
			test_evals = runner.evaluate(data_dict[phase],runner.topk,runner.metrics,show_num=True,phase=phase)
			logging.info("%s metrics: %s"%(phase,utils.format_metric(test_evals)))


if __name__ == "__main__":
	init_parser = argparse.ArgumentParser(description='Model')
	init_parser.add_argument('--model_name', type=str, default='RuleBase', help='Choose an ensemble model.')
	init_parser.add_argument('--loss_name',default='BPRloss',help='Name of loss function.')
	init_parser.add_argument('--runner_name',default='BaseRunner',help='Name of runner.')
	init_args, init_extras = init_parser.parse_known_args()
	model_name = eval('{0}.{0}'.format(init_args.model_name))
	reader_name = eval('{0}.{0}'.format(model_name.reader))
	runner_name = eval('{0}.{0}'.format(init_args.runner_name))
	loss_name = eval('{0}.{0}'.format(init_args.loss_name))

	# Args
	parser = argparse.ArgumentParser(description='')
	parser = parse_global_args(parser)
	parser = reader_name.parse_data_args(parser)
	parser = model_name.parse_model_args(parser)
	parser = runner_name.parse_runner_args(parser)
	parser = loss_name.parse_loss_args(parser)
	args, extras = parser.parse_known_args()
	logging.info("Extra args: %s"%(str(extras)))
	
	# Logging configuration
	log_args = [init_args.loss_name, args.dataset, str(args.random_seed),args.save_anno]
	for arg in model_name.extra_log_args:
		log_args.append(arg + '=' + str(eval('args.' + arg)))
	log_file_name = '__'.join(log_args).replace(' ', '__')
	if args.log_file == '':
		args.log_file = '../logs/{}/{}/model.txt'.format(init_args.model_name, log_file_name)
	if args.model_path == '':
		args.model_path = '../models/{}/{}/model.pt'.format(init_args.model_name, log_file_name)
	utils.check_dir(args.log_file)
	utils.check_dir(args.model_path)
	reload(logging)
	logging.basicConfig(filename=args.log_file, level=args.verbose)
	logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
	
	logging.info("Save model to %s"%(args.model_path))
	logging.info(init_args)

	main()

