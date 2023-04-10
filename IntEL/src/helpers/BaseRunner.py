# -*- coding: UTF-8 -*-

from email.policy import default
import os
import gc
import torch
import torch.nn as nn
import logging
import numpy as np
from time import time
from tqdm import tqdm
from torch.utils.data import DataLoader
from typing import Dict, List
from sklearn.metrics import ndcg_score, f1_score, recall_score, precision_score
from torch.utils.tensorboard import SummaryWriter

from utils import utils
from models.BaseModel import BaseModel

class BaseRunner(object):
	@staticmethod
	def parse_runner_args(parser):
		parser.add_argument('--epoch', type=int, default=200,
							help='Number of epochs.')
		parser.add_argument('--test_epoch', type=int, default=-1,
							help='Print test results every test_epoch (-1 means no print).')
		parser.add_argument('--early_stop', type=int, default=10,
							help='The number of epochs when dev results drop continuously.')
		parser.add_argument('--lr', type=float, default=1e-3,
							help='Learning rate.')
		parser.add_argument('--l2', type=float, default=0,
							help='Weight decay in optimizer.')
		parser.add_argument('--intent_l2', type=float, default=1e-6,
							help='Weight decay for intent module in optimizer.')
		parser.add_argument('--batch_size', type=int, default=256,
							help='Batch size during training.')
		parser.add_argument('--eval_batch_size', type=int, default=100,
							help='Batch size during testing.')
		parser.add_argument('--optimizer', type=str, default='Adam',
							help='optimizer: SGD, Adam, Adagrad, Adadelta')
		parser.add_argument('--num_workers', type=int, default=4,
							help='Number of processors when prepare batches in DataLoader')
		parser.add_argument('--pin_memory', type=int, default=0,
							help='pin_memory in DataLoader')
		parser.add_argument('--topk', type=str, default='1,3,5',
								help='The number of items recommended to each user.')
		parser.add_argument('--metrics', type=str, default='NDCG,HR',
								help='metrics: NDCG, HR')
		parser.add_argument('--main_metric', type=str, default='NDCG@1',
								help='main metric')
		parser.add_argument('--test_ensemble',type=int,default=1)
		parser.add_argument('--decay_lr',type=float,default=0)
		parser.add_argument('--decay_step',type=int,default=1)
		return parser

	@staticmethod
	def evaluate_method(prediction_scores,ranking_lists,pos_nums,topk,metrics, session_len, show_num=False):
		evaluations = dict()

		if len(session_len) > len(prediction_scores):
			session_len = session_len[:len(prediction_scores)]
			for key in pos_nums.keys():
				pos_nums[key] = pos_nums[key][:len(prediction_scores)]

		test_size = len(session_len)
		max_len = max(max(session_len),max(topk))
		# resort
		predictions = np.array([prediction_scores[i][:session_len[i]].tolist()+[0]*(max_len-session_len[i])
			if session_len[i]<len(prediction_scores[i]) else 
			prediction_scores[i].tolist()+[0]*(max_len-len(prediction_scores[i]))
		 for i in range(test_size)]) # 补齐list
		rankings = np.array([ranking_lists[i][:session_len[i]].tolist()+[-2]*(max_len-session_len[i]) 
			if session_len[i]<len(ranking_lists[i]) else 
			ranking_lists[i].tolist()+[-2]*(max_len-len(ranking_lists[i]))
			 for i in range(test_size)]) # 补齐list


		rankings_idxs = np.argsort(rankings,axis=1)[:,::-1]
		rankings_first_idxs = np.arange(len(rankings)).reshape(-1,1)
		rankings = rankings[rankings_first_idxs,rankings_idxs]
		predictions = predictions[rankings_first_idxs,rankings_idxs]

		rankings[np.where(rankings<0)] = 0
		
		sort_idx = predictions.argsort(axis=1)
		discounts = 1/np.log2(np.arange(max_len)+2.0)

		for btype, pos_num in pos_nums.items():
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
			if show_num:
				logging.info("# %s session: %d"%(behavior,len(select_idx)))
			for k in topk:
				min_k = min(k,predictions.shape[1])
				for metric in metrics:
					key = '{}_{}@{}'.format(behavior,metric, k)
					if metric == 'HR':
						hit = positive_idxs[:,-min_k:].sum(axis=1)>0
						evaluations[key] = hit.mean()
					elif metric == 'NDCG':
						if k == 1:
							continue # NDCG@1 is the same as HR@1
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
		for k in topk:
			dcg = (rankings[:,:k] * discounts[:k]).sum(axis=1)
			idcg = (rankings_perfect[:,:k]*discounts[:k]).sum(axis=1)
			ndcg = (dcg/idcg).mean()
			evaluations['NDCG@%d'%(k)] = ndcg

		del rankings, predictions,rankings_perfect, rankings_idxs,predictions_idxs
		gc.collect()

		return evaluations

	def evaluate_intents(self, true_intents, predict_intents,topk=[1,5,10,30]):
		evaluations = dict()
		# ranking metrics
		true_labels = np.argmax(true_intents,axis=1).reshape(-1,1)
		predict_sort = np.argsort(predict_intents,axis=1)
		predict_idxs = np.argsort(predict_intents,axis=1)[:,::-1]
		predict_first_idxs = np.arange(len(predict_intents)).reshape(-1,1)
		true_sort = np.array(true_intents)[predict_first_idxs,predict_idxs]
		true_perfect = np.sort(true_intents,axis=1)[:,::-1]
		discounts = 1/np.log2(np.arange(40)+2.0)
		for k in topk:
			dcg = (true_sort[:,:k]*discounts[:k]).sum(axis=1)
			idcg = (true_perfect[:,:k]*discounts[:k]).sum(axis=1)
			ndcg = (dcg/idcg).mean()
			evaluations['Int-NDCG@%d'%(k)] = ndcg
			hr = ((predict_sort==true_labels)[:,-k:].sum(axis=-1)>0).mean()
			evaluations['Int-HR@%d'%(k)] = hr
		return evaluations	

	def __init__(self, args):
		self.epoch = args.epoch
		self.test_epoch = args.test_epoch
		self.early_stop = args.early_stop
		self.learning_rate = args.lr
		self.batch_size = args.batch_size
		self.eval_batch_size = args.eval_batch_size
		self.l2 = args.l2
		self.intent_l2 = args.l2
		self.optimizer_name = args.optimizer
		self.num_workers = args.num_workers
		self.pin_memory = args.pin_memory
		self.topk = [int(x) for x in args.topk.split(',')]
		self.metrics = [m.strip().upper() for m in args.metrics.split(',')]
		self.main_metric = args.main_metric # early stop based on main_metric
		self.test_ensemble = args.test_ensemble
		self.decay_lr = args.decay_lr
		self.decay_step = args.decay_step
		self.stop_tol = 1e-4

		self.time = None  # will store [start_time, last_step_time]

	def _check_time(self, start=False):
		if self.time is None or start:
			self.time = [time()] * 2
			return self.time[0]
		tmp_time = self.time[1]
		self.time[1] = time()
		return self.time[1] - tmp_time

	def _build_optimizer(self, model):
		logging.info('Optimizer: ' + self.optimizer_name)
		optimizer = eval('torch.optim.{}'.format(self.optimizer_name))(
			model.customize_parameters({'intent_l2':self.intent_l2,'ens_l2':self.l2}), 
						lr=self.learning_rate, weight_decay=self.l2)
		scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.decay_step,gamma=self.decay_lr)
		return optimizer, scheduler

	def train(self, data_dict, criterion,save_anno="test"):
		model = data_dict['train'].model
		main_metric_results, dev_results = list(), list()
		self._check_time(start=True)
		writer = SummaryWriter('../logs/tensorboard_log/%s'%(save_anno))
		try:
			dev_loss, dev_result = self.evaluate(data_dict['dev'], self.topk[:1], self.metrics,criterion,topk_intent=[5])
			logging_str = 'Epoch 0	dev loss={:<.4f}, ({})'.format(dev_loss, utils.format_metric(dev_result))
			logging.info(logging_str)
			for epoch in range(self.epoch):
				# Fit
				self._check_time()
				gc.collect()
				torch.cuda.empty_cache()
				
				if epoch == 0:
					loss = self.fit(data_dict['train'], epoch=epoch + 1, criterion=criterion, use_writer=True, writer=writer)
				else:
					loss = self.fit(data_dict['train'], epoch=epoch + 1, criterion=criterion)
				writer.add_scalar('train_loss',loss,epoch)
				training_time = self._check_time()
				if np.isnan(loss):
					raise ValueError("Loss is nan!")

				# Record dev results
				if epoch == 0:
					dev_loss, dev_result = self.evaluate(data_dict['dev'], self.topk[:1], self.metrics,criterion, use_writer=True,writer=writer,topk_intent=[3,5])
				else:
					dev_loss, dev_result = self.evaluate(data_dict['dev'], self.topk[:1], self.metrics,criterion,topk_intent=[3,5])
				writer.add_scalar('dev_loss',dev_loss,epoch)
				dev_results.append(dev_result)
				main_metric_results.append(dev_result[self.main_metric])
				writer.add_scalar('dev_%s'%(self.main_metric.lower()),dev_result[self.main_metric],epoch)
				logging_str = 'Epoch {:<5} loss={:<.4f} [{:<3.1f} s]	dev loss={:<.4f}, ({})'.format(
					epoch + 1, loss, training_time,dev_loss, utils.format_metric(dev_result))

				# Test
				if self.test_epoch > 0 and epoch % self.test_epoch  == 0:
					if epoch == 0:
						test_loss, test_result = self.evaluate(data_dict['test'], self.topk[:1], self.metrics,criterion,use_writer=True,writer=writer,topk_intent=[5])
					else:
						test_loss, test_result = self.evaluate(data_dict['test'], self.topk[:1], self.metrics,criterion,topk_intent=[5],sample=300)
					logging_str += ' test loss={:<.4f}, ({})'.format(test_loss,utils.format_metric(test_result))
					writer.add_scalar('test_loss',test_loss,epoch)
					writer.add_scalar('test_%s'%(self.main_metric.lower()),test_result[self.main_metric],epoch)
				testing_time = self._check_time()
				logging_str += ' [{:<.1f} s]'.format(testing_time)

				if self.decay_lr>0:
					model.scheduler.step()
					c_lr = model.scheduler.get_last_lr()
					logging.info("LR: %.4f"%(c_lr[0]))

				# Save model and early stop
				if  len(main_metric_results)==1 or max(main_metric_results[:-1]) < main_metric_results[-1] - self.stop_tol or \
						(hasattr(model, 'stage') and model.stage == 1):
					model.save_model()
					logging_str += ' *'
				logging.info(logging_str)

				if self.early_stop > 0 and self.eval_termination(main_metric_results):
					logging.info("Early stop at %d based on dev result." % (epoch + 1))
					break
		except KeyboardInterrupt:
			logging.info("Early stop manually")
			exit_here = input("Exit completely without evaluation? (y/n) (default n):")
			if exit_here.lower().startswith('y'):
				logging.info(os.linesep + '-' * 45 + ' END: ' + utils.get_time() + ' ' + '-' * 45)
				exit(1)
		except Exception as e:
			logging.info("ERROR: "+str(e))

		# Find the best dev result across iterations
		best_epoch = main_metric_results.index(max(main_metric_results))
		logging.info(os.linesep + "Best Iter(dev)={:>5}\t dev=({}) [{:<.1f} s] ".format(
			best_epoch + 1, utils.format_metric(dev_results[best_epoch]), self.time[1] - self.time[0]))
		model.load_model()

	def fit(self, dataset, epoch=-1,criterion="",use_writer=False,writer=None):
		model = dataset.model
		if model.optimizer is None:
			model.optimizer, model.scheduler = self._build_optimizer(model)
		dataset.actions_before_epoch()  # must sample before multi thread start

		model.train()
		loss_lst = list()
		dl = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers,
						collate_fn=dataset.collate_batch, pin_memory=self.pin_memory)
		cnt = 0
		for batch in tqdm(dl, leave=False, desc='Epoch {:<3}'.format(epoch), ncols=100, mininterval=1):
			cnt += 1
			batch = utils.batch_to_gpu(batch, model.device)
			model.optimizer.zero_grad()
			out_dict = model(batch)
			loss, ensemble_loss, intent_loss = criterion(out_dict,batch)
			if use_writer:
				writer.add_scalar('batch_train_ensloss',ensemble_loss,cnt)
				writer.add_scalar('batch_train_intloss',intent_loss,cnt)
			loss.backward()
			model.optimizer.step()
			loss_lst.append(loss.detach().cpu().data.numpy())
		return np.mean(loss_lst).item()

	def evaluate(self,dataset,topk,metrics,criterion,show_num=False,phase='',sample=0,use_writer=False, writer=None,topk_intent=[1,5,10,30],cpu_evaluate=True):
		prediction_scores, loss, ranking_lists, true_intent, predict_intent = self.predict(
						dataset,criterion,phase,sample,use_writer=use_writer,writer=writer)
		if cpu_evaluate:
			pos_num = dict()
			for dtype in dataset.corpus.pos_types:
				pos_num[dtype] = dataset.data[dtype]
			evaluate_metrics = dict()
			if self.test_ensemble:
				evaluate_metrics.update(self.evaluate_method(prediction_scores, ranking_lists,pos_num, topk, metrics, dataset.data['session_len'],show_num=show_num))
			if len(true_intent): # test intent prediction results
				evaluate_metrics.update(self.evaluate_intents(true_intent,predict_intent,topk=topk_intent))
			e=time()
		else:
			evaluate_metrics = dict()

		del prediction_scores, ranking_lists, true_intent, predict_intent
		gc.collect() 
		return loss, evaluate_metrics
	
	def predict(self,dataset,criterion,phase='test',sample=0,use_writer=False,writer=None):
		dataset.model.eval()
		prediction_scores = list()
		ranking_lists = list()
		test_loss = list()
		true_intent, predict_intent = list(), list()
		session_ids = list()
		dl = DataLoader(dataset, batch_size=self.eval_batch_size, shuffle=False, num_workers=self.num_workers,
						collate_fn=dataset.collate_batch, pin_memory=self.pin_memory)
		gc.collect()
		torch.cuda.empty_cache()
		cnt = 0
		with torch.no_grad():
			for batch in tqdm(dl, leave=False, ncols=100, mininterval=1, desc='Predict'):
				cnt += 1
				out_dict = dataset.model(utils.batch_to_gpu(batch, dataset.model.device))
				loss, ensemble_loss, intent_loss = criterion(out_dict,batch)
				if use_writer:
					if batch['phase'] == 'dev':
						writer.add_scalar('batch_pred_ensloss',ensemble_loss,cnt)
						writer.add_scalar('batch_pred_intloss',intent_loss,cnt)
					else:
						writer.add_scalar('batch_pred_ensloss_test',ensemble_loss,cnt)
						writer.add_scalar('batch_pred_intloss_test',intent_loss,cnt)
				test_loss.append(loss.item())
				prediction_scores.extend(out_dict["ens_score"].cpu().data.numpy())
				ranking_lists.extend(batch['ranking'].cpu().data.numpy())
				if 'intents' in out_dict:
					true_intent.extend(batch['intents'].cpu().data.numpy())
					predict_intent.extend(out_dict['intents'].cpu().data.numpy())
					session_ids.extend(batch['c_id_c'].cpu().data.numpy())

			prediction_scores = np.array(prediction_scores)
		if len(phase):
			model_path = os.path.dirname(dataset.model.model_path)
			np.save(os.path.join(model_path,phase+"_predintent.npy"),np.array(predict_intent))
			np.save(os.path.join(model_path,phase+"_trueintent.npy"),np.array(true_intent))
			np.save(os.path.join(model_path,phase+"_sessionids.npy"),np.array(session_ids))
			np.save(os.path.join(model_path,phase+"_predscores.npy"),prediction_scores)
			np.save(os.path.join(model_path,phase+"_rankings.npy"),np.array(ranking_lists))
		del dl,session_ids
		gc.collect()
		return prediction_scores, np.mean(test_loss), ranking_lists, true_intent, predict_intent


	def eval_termination(self, criterion: List[float]) -> bool:
		if len(criterion) - criterion.index(max(criterion)) > self.early_stop:
			return True
		return False