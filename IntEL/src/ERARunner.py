'''
Specific runner for ERA method.
Reference:
	Samuel Oliveira, Victor Diniz, Anisio Lacerda, and Gisele L Pappa. 2016. Evolutionary rank aggregation for recommender systems. 
	IEEE Congress on Evolutionary Computation (CEC). IEEE, 255–262.
'''
# -*- coding: UTF-8 -*-

import os
import gc
import sys
import argparse
import pickle
import torch
import torch.nn as nn
import logging
import numpy as np
from time import time
from tqdm import tqdm
from torch.utils.data import DataLoader
from typing import Dict, List
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score

import pygad
import pygad.torchga
from helpers import *
from models import *
from models.unsupervise import *
from models.supervise import *
from imp import reload
from utils import utils
from loss import *
	
def evaluate_method(prediction_scores,ranking_lists,pos_nums,topk,metrics, session_len):
	evaluations = dict()

	if len(session_len)> len(prediction_scores):
		session_len = session_len[:len(prediction_scores)]
		for key in pos_nums.keys():
			pos_nums[key] = pos_nums[key][:len(prediction_scores)]

	test_size = len(session_len)
	max_len = max(max(session_len),max(topk))
	# resort
	predictions = np.array([prediction_scores[i][:session_len[i]].tolist()+[0]*(max_len-session_len[i])
					if session_len[i] < len(prediction_scores[i]) else
					prediction_scores[i].tolist() + [0] * (max_len-len(prediction_scores[i]))
						for i in range(test_size)])
	rankings = np.array([ranking_lists[i][:session_len[i]].tolist()+[-2]*(max_len-session_len[i]) 
					if session_len[i] < len(ranking_lists[i]) else
					ranking_lists[i].tolist() + [0] * (max_len-len(ranking_lists[i]))
					for i in range(test_size)])
	rankings_idxs = np.argsort(rankings,axis=1)[:,::-1]
	rankings_first_idxs = np.arange(len(rankings)).reshape(-1,1)
	rankings = rankings[rankings_first_idxs,rankings_idxs]
	predictions = predictions[rankings_first_idxs,rankings_idxs]
	rankings[np.where(rankings<0)] = 0
	sort_idx = predictions.argsort(axis=1)
	discounts = 1/np.log2(np.arange(max_len)+2.0)

	for btype, pos_num in pos_nums.items(): # evaluation on each behavior
		behavior = btype.split("_")[1].split("num")[0]
		if 'click' in btype:
			all_pos = np.sum(np.array(list(pos_nums.values())),axis=0).reshape(-1,1)
		else:
			all_pos = pos_num.reshape(-1,1)
		positive_idxs = sort_idx < all_pos
		# calculate on lists with at least one positive item
		select_idx = [i for i in range(len(all_pos)) if all_pos[i,0]>0]
		positive_idxs = positive_idxs[select_idx,:]
		all_pos = all_pos[select_idx,:]
		logging.info("# %s session: %d"%(behavior,len(select_idx)))
		for k in topk:
			min_k = min(k,predictions.shape[1])
			for metric in metrics:
				key = '{}_{}@{}'.format(behavior,metric, k)
				if metric == 'HR':
					hit = positive_idxs[:,-min_k:].sum(axis=1)>0
					evaluations[key] = hit.mean()
				elif metric == 'NDCG':
					dcg = (positive_idxs[:,-min_k:]*discounts[:min_k][::-1]).sum(axis=1)
					ideal_idxs = np.arange(min_k).reshape(1,-1) < all_pos
					idcg = (ideal_idxs[:,:min_k] * discounts[:min_k]).sum(axis=1)
					evaluations[key] = (dcg/idcg).mean()
				else:
					raise ValueError('Undefined evaluation metric: {}.'.format(metric))
	predictions_idxs = np.argsort(predictions,axis=1)[:,::-1]
	predictions_first_idxs = np.arange(len(predictions)).reshape(-1,1)
	rankings = rankings[predictions_first_idxs,predictions_idxs]
	rankings_perfect = np.sort(rankings,axis=1)[:,::-1]
	predictions = predictions[predictions_first_idxs,predictions_idxs]
	for k in topk: # evaluate on all rankings
		dcg = (rankings[:,:k] * discounts[:k]).sum(axis=1)
		idcg = (rankings_perfect[:,:k]*discounts[:k]).sum(axis=1)
		idcg[np.where(idcg==0)] = 1
		ndcg = (dcg/idcg).mean()
		evaluations['NDCG@%d'%(k)] = ndcg

	return evaluations

def parse_runner_args(parser):
	parser.add_argument('--epoch', type=int, default=200,
						help='Number of epochs.')
	parser.add_argument('--topk', type=str, default='1,3,5',
							help='The number of items recommended to each user.')
	parser.add_argument('--metrics', type=str, default='NDCG,HR',
							help='metrics: NDCG, HR')
	parser.add_argument('--main_metric', type=str, default='NDCG@1',
							help='main metric')
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
	parser.add_argument('--num_solutions',type=int,default=100)
	parser.add_argument('--num_generations',type=int,default=200)
	parser.add_argument('--num_parents_mating',type=int,default=5)
	parser.add_argument('--crossover_prob',type=float,default=0.65)
	parser.add_argument('--mutation_prob',type=float,default=0.25)
	parser.add_argument('--reproduction_prob',type=float,default=0.1)
	parser.add_argument('--elitism',type=int,default=2)
	parser.add_argument('--batch_size', type=int, default=256,
							help='Batch size during training.')
	parser.add_argument('--pin_memory', type=int, default=0,
							help='pin_memory in DataLoader')
	return parser

def fitness_func(solution, sol_idx):
	# fitness function for Evolutionary Algorithm
	global val_ranking_list, val_pos_nums, val_session_lens, val_inputs, model
	predictions = pygad.torchga.predict(model=model,
									solution=solution,
										data=val_inputs)
	metrics = evaluate_method(predictions, val_ranking_list, val_pos_nums, [1], ['NDCG'], val_session_lens)
	logging_str = 'fitness: {}'.format(utils.format_metric(metrics))
	logging.info(logging_str)
	solution_fitness = metrics['NDCG@1']
	return solution_fitness

def callback_generation(ga_instance):
	print("Generation = {generation}".format(generation=ga_instance.generations_completed))
	print("Fitness	= {fitness}".format(fitness=ga_instance.best_solution()[1]))

def train(data_dict, args):
	global val_ranking_list, val_pos_nums, val_session_lens, val_inputs, model
	torch_ga = pygad.torchga.TorchGA(model=model, num_solutions=args.num_solutions)

	num_generations = args.num_generations
	num_parents_mating = args.num_parents_mating
	initial_population = torch_ga.population_weights

	# call for a GA
	ga_instance = pygad.GA(num_generations=num_generations,
					   num_parents_mating=num_parents_mating,
					   initial_population=initial_population,
					   fitness_func=fitness_func,
					   parent_selection_type='tournament',
					   K_tournament=7,
					   crossover_type='single_point',
					   crossover_probability=args.crossover_prob,
					   mutation_type='random',
					   mutation_probability=args.mutation_prob,
					   mutation_by_replacement=False,
					   keep_elitism=args.elitism,
					   random_seed=args.random_seed,
					   on_generation=callback_generation)
	val_pos_nums = dict()
	for dtype in data_dict['dev'].corpus.pos_types:
		val_pos_nums[dtype] = data_dict['dev'].data[dtype]
	val_session_lens = data_dict['dev'].data['session_len']

	dl = DataLoader(data_dict['dev'], batch_size=args.batch_size, shuffle=False, num_workers=4,
						collate_fn=data_dict['dev'].collate_batch, pin_memory=args.pin_memory)
	val_inputs = dict()
	val_ranking_list = list()
	key_features = ['p10','mAgr']
	max_length = 0
	for m in range(model.model_num):
		key_features.append('psc_%d'%(m))
	for batch in tqdm(dl, leave=False, ncols=100, mininterval=1):
		batch = utils.batch_to_gpu(batch, model.device)
		val_ranking_list.extend(batch['ranking'].cpu().data.numpy())
		for key in key_features:
			val_inputs[key] = val_inputs.get(key,[])
			val_inputs[key].append(batch[key])
			max_length = max(max_length,batch[key].shape[-1])
	for key in val_inputs:
		for i in range(len(val_inputs[key])):
			zero_pad = torch.zeros(val_inputs[key][i].shape[0],
						max_length-val_inputs[key][i].shape[1]).to(model.device)
			val_inputs[key][i] = torch.cat([val_inputs[key][i],zero_pad],dim=1)
		val_inputs[key] = torch.cat(val_inputs[key],dim=0)
	
	ga_instance.run()

	solution, solution_fitness, solution_idx = ga_instance.best_solution()
	print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
	print("Index of the best solution : {solution_idx}".format(solution_idx=solution_idx))

	return solution, solution_fitness, solution_idx

def evaluate(solution, dataset, topk, metrics, show_num=False, phase="test"):
	global model
	dl = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0,
						collate_fn=dataset.collate_batch, pin_memory=args.pin_memory)
	test_pos_nums = dict()
	for dtype in dataset.corpus.pos_types:
		test_pos_nums[dtype] = dataset.data[dtype]
	test_session_lens = dataset.data['session_len']
	test_inputs = dict()
	test_ranking_list = list()
	key_features = ['p10','mAgr']
	max_length = 0
	for m in range(model.model_num):
		key_features.append('psc_%d'%(m))
	for batch in tqdm(dl, leave=False, ncols=100, mininterval=1):
		if np.random.rand()>0.5:
			batch = utils.batch_to_gpu(batch, model.device)
			test_ranking_list.extend(batch['ranking'].cpu().data.numpy())
			for key in key_features:
				test_inputs[key] = test_inputs.get(key,[])
				test_inputs[key].append(batch[key])
				max_length = max(max_length,batch[key].shape[-1])
	for key in test_inputs:
		for i in range(len(test_inputs[key])):
			zero_pad = torch.zeros(test_inputs[key][i].shape[0],
						max_length-test_inputs[key][i].shape[1]).to(model.device)
			test_inputs[key][i] = torch.cat([test_inputs[key][i],zero_pad],dim=1)
		test_inputs[key] = torch.cat(test_inputs[key],dim=0)
	predictions = pygad.torchga.predict(model=model,solution=solution,data=test_inputs)
	metrics = evaluate_method(predictions, test_ranking_list, test_pos_nums, topk, metrics, test_session_lens)
	return metrics

def main():
	global val_ranking_list, val_pos_nums, val_session_lens, val_inputs, model
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
	else:
		corpus = reader_name(args)
		logging.info('Save corpus to {}'.format(corpus_path))
		pickle.dump(corpus, open(corpus_path, 'wb'))

	model = model_name(args,corpus).to(args.device)
	logging.info('#params: {}'.format(model.count_variables()))
	logging.info(model)
	
	# load model
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
		solution, solution_fitness, solution_idx = train(data_dict, args)
	else:
		logging.info("[Warning] No training!")

	# evaluate
	logging.info("Final evaluation!")
	topk = [int(x) for x in args.topk.split(",")]
	metrics = [x for x in args.metrics.split(",")]
	for phase in ['dev','test']:
		session_len = data_dict[phase].data['session_len']
		test_evals = evaluate(solution, data_dict[phase],topk,metrics, 
							show_num=True,phase=phase)
		logging.info("%s metrics: %s "%(phase,utils.format_metric(test_evals)))

if __name__ == '__main__':
	init_parser = argparse.ArgumentParser(description='Model')
	init_parser.add_argument('--model_name', type=str, default='ERA', help='Choose an ensemble model.')
	init_parser.add_argument('--loss_name',default='BPRloss',help='Name of loss function.')
	init_parser.add_argument('--runner_name',default='BaseRunner',help='Name of runner.')
	init_args, init_extras = init_parser.parse_known_args()
	model_name = eval('{0}.{0}'.format(init_args.model_name))
	reader_name = eval('{0}.{0}'.format(model_name.reader))
	
	# Args
	parser = argparse.ArgumentParser(description='')
	parser = reader_name.parse_data_args(parser)
	parser = model_name.parse_model_args(parser)
	parser = parse_runner_args(parser)
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
	
