'''
Specific runner for lambda-rank method.
Lambda-rank needs specified gradient-backward process.
'''
import os
import gc
import logging
import numpy as np
from time import time
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score
from utils import utils

class LambdaRankRunner(object):
	@staticmethod
	def parse_runner_args(parser):
		parser.add_argument('--lr', type=float, default=1e-3,
							help='Learning rate.')
		parser.add_argument('--topk', type=str, default='1,3,5',
								help='The number of items recommended to each user.')
		parser.add_argument('--metrics', type=str, default='NDCG,HR',
								help='metrics: NDCG, HR')
		parser.add_argument('--main_metric', type=str, default='NDCG@1',
								help='main metric')
		parser.add_argument('--epoch', type=int, default=200,
							help='Number of epochs.')
		parser.add_argument('--test_epoch', type=int, default=1,
							help='Print test results every test_epoch (-1 means no print).')
		parser.add_argument('--early_stop', type=int, default=10,
							help='The number of epochs when dev results drop continuously.')
		parser.add_argument('--batch_size', type=int, default=256,
							help='Batch size during training.')
		parser.add_argument('--eval_batch_size', type=int, default=256,
							help='Batch size during testing.')
		parser.add_argument('--num_workers', type=int, default=4,
							help='Number of processors when prepare batches in DataLoader')
		parser.add_argument('--pin_memory', type=int, default=0,
							help='pin_memory in DataLoader')
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

	def evaluate_intents(self, true_intents, predict_intents):
		evaluations = dict()
		# niche intent finding
		true_intents = np.array(true_intents)
		predict_intents = np.array(predict_intents)
		if predict_intents.shape[1]==11:
			true_niche = np.concatenate((true_intents[:,:6],true_intents[:,7:]),axis=1)
		else: # All intents
			true_niche = true_intents #np.concatenate((true_intents[:,:6],true_intents[:,7:]),axis=1)

		#RMSE
		rmse = np.sqrt(((true_niche - predict_intents)**2).mean(axis=0))
		evaluations['Int-rmse'] = np.mean(rmse)
		logging.info("RMSE: ["+",".join(["%.4f"%(x) for x in rmse])+"]")
		#Binary
		auc = [roc_auc_score((true_niche[:,i]>0).astype(int),predict_intents[:,i]) 
					if true_niche[:,i].sum()>0 else 0 for i in range(true_niche.shape[1])]
		evaluations['Int-auc'] = np.mean(auc)
		logging.info("AUC:  ["+",".join(["%.4f"%(x) for x in auc])+"]")
		# classification metrics
		max_true = np.argmax(true_intents,axis=1)
		max_predict = np.argmax(predict_intents,axis=1)
		evaluations['Int-MacroF1']=f1_score(max_true,max_predict,average='macro')
		evaluations['Int-MicroF1']=f1_score(max_true,max_predict,average='micro')
				
		return evaluations

	def __init__(self, args):
		self.epoch = args.epoch
		self.test_epoch = args.test_epoch
		self.early_stop = args.early_stop
		self.learning_rate = args.lr
		self.batch_size = args.batch_size
		self.eval_batch_size = args.eval_batch_size
		self.num_workers = args.num_workers
		self.pin_memory = args.pin_memory
		self.topk = [int(x) for x in args.topk.split(',')]
		self.metrics = [m.strip().upper() for m in args.metrics.split(',')]
		self.main_metric = args.main_metric # early stop based on main_metric
		# self.main_metric = '{}@{}'.format(self.metrics[0], self.topk[0]) 
		self.stop_tol = 1e-4

		self.time = None  # will store [start_time, last_step_time]

	def _check_time(self, start=False):
		if self.time is None or start:
			self.time = [time()] * 2
			return self.time[0]
		tmp_time = self.time[1]
		self.time[1] = time()
		return self.time[1] - tmp_time
	
	def train(self,data_dict, criterion, save_anno="test"):
		model = data_dict['train'].model
		self._check_time(start=True)
		main_metric_results, dev_results = list(), list()
		try:
			dev_result = self.evaluate(data_dict['dev'], self.topk[:1], self.metrics)
			logging_str = 'Epoch 0: dev {}'.format(utils.format_metric(dev_result))
			logging.info(logging_str)
			for epoch in range(self.epoch):
				# Fit
				self._check_time()
				gc.collect()
				torch.cuda.empty_cache()
				
				lambda_mean = self.fit(data_dict['train'], epoch=epoch + 1,)
				training_time = self._check_time()
				if np.isnan(lambda_mean):
					raise ValueError("Lambda is nan!")

				if epoch %3 ==0:
					train_result = self.evaluate(data_dict['train'], self.topk[:1], self.metrics,sample=True)
					logging_str = 'train: {}'.format(utils.format_metric(train_result))
					logging.info(logging_str)
				
				# Record dev results
				dev_result = self.evaluate(data_dict['dev'], self.topk[:1], self.metrics)
				dev_results.append(dev_result)
				main_metric_results.append(dev_result[self.main_metric])
				logging_str = 'Epoch {:<5} [{:<3.1f} s]	dev: {}'.format(
					epoch + 1, training_time, utils.format_metric(dev_result))
				# Test
				if self.test_epoch > 0 and epoch % self.test_epoch  == 0:
					test_result = self.evaluate(data_dict['test'], self.topk[:1], self.metrics)
					logging_str += ' test: {}'.format(utils.format_metric(test_result))
				testing_time = self._check_time()
				logging_str += ' [{:<.1f} s]'.format(testing_time)
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
		# except Exception as e:
		# 	logging.info("ERROR:"+str(e))

		# Find the best dev result across iterations
		best_epoch = main_metric_results.index(max(main_metric_results))
		logging.info(os.linesep + "Best Iter(dev)={:>5}\t dev=({}) [{:<.1f} s] ".format(
			best_epoch + 1, utils.format_metric(dev_results[best_epoch]), self.time[1] - self.time[0]))
		model.load_model()

	def fit(self, dataset, epoch):
		model = dataset.model
		model.train()
		lambda_mean = list()
		dl = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers,
						collate_fn=dataset.collate_batch, pin_memory=self.pin_memory)
		for batch in tqdm(dl, leave=False, desc='Epoch {:<3}'.format(epoch), ncols=100, mininterval=1):
			batch = utils.batch_to_gpu(batch, model.device)
			session_lens = batch['session_len']
			true_scores = batch['ranking']
			true_scores = torch.clamp(true_scores,min=0)
			# true_scores[np.where(true_scores<0)]=0

			out_dict = model(batch)
			predicted_scores = out_dict['ens_score'] # batch * length

			# 改为矩阵计算！
			# pred_score = predicted_scores.detach().cpu().numpy()
			# lambdas = np.zeros_like(pred_score)
			# zip_parameters = zip(true_scores.cpu().numpy(), pred_score, session_lens.cpu().numpy())
			# for bid,(ts, ps, sl) in enumerate(zip_parameters):
			# 	sub_lambda, sub_w = self.compute_lambda(ts, ps, sl, true_scores, predicted_scores.detach(), session_lens, bid) # 得到该query下document的lambda
			# 	lambdas[bid,:sl] = sub_lambda[:sl]
			# 	lambda_mean.append(sub_lambda[:sl].mean())
			lambdas_torch = self.compute_lambda_new(true_scores,predicted_scores.detach(),session_lens)
			if torch.isnan(lambdas_torch).sum():
				print("NAN!")
				inputs = ''
				while inputs != 'continue':
					try:
						print(eval(inputs))
					except Exception as e:
						print(e)
					inputs = input()

			lambda_mean.append(lambdas_torch.mean().cpu().detach().numpy())

			model.zero_grad()
			# lambdas_torch = torch.Tensor(lambdas).to(model.device)
			predicted_scores.backward(lambdas_torch, retain_graph=True)
			with torch.no_grad():
				for param in model.parameters():
					param.data.add_(param.grad.data * self.learning_rate)
					# inputs = ''
					# while inputs != 'continue':
					# 	try:
					# 		print(eval(inputs))
					# 	except Exception as e:
					# 		print(e)
					# 	inputs = input()

		return np.mean(lambda_mean)

	def evaluate(self,dataset,topk,metrics,show_num=False,phase='',sample=0, use_writer=False, writer=None, topk_intent=[1,5,10,30]):
		prediction_scores, ranking_lists, true_intent, predict_intent = self.predict(
						dataset,phase,sample,use_writer=use_writer,writer=writer)
		pos_num = dict()
		for dtype in dataset.corpus.pos_types:
			pos_num[dtype] = dataset.data[dtype]
		evaluate_metrics = dict()
		# if 0: # test ensemble results
		evaluate_metrics.update(self.evaluate_method(prediction_scores, ranking_lists,pos_num, topk, metrics, dataset.data['session_len'],show_num=show_num))
		if len(true_intent): # test intent prediction results
			evaluate_metrics.update(self.evaluate_intents(true_intent,predict_intent))
		del prediction_scores, ranking_lists, true_intent, predict_intent
		gc.collect() 
		return evaluate_metrics

	def predict(self,dataset,phase='test',sample=0,use_writer=False,writer=None):
		dataset.model.eval()
		prediction_scores = list()
		ranking_lists = list()
		true_intent, predict_intent = list(), list()
		session_ids = list()
		dl = DataLoader(dataset, batch_size=self.eval_batch_size, shuffle=False, num_workers=self.num_workers,
						collate_fn=dataset.collate_batch, pin_memory=self.pin_memory)
		with torch.no_grad():
			cnt = 0
			for batch in tqdm(dl, leave=False, ncols=100, mininterval=1, desc='Predict'):
				cnt += 1
				out_dict = dataset.model(utils.batch_to_gpu(batch, dataset.model.device))
				prediction_scores.extend(out_dict["ens_score"].cpu().data.numpy())
				ranking_lists.extend(batch['ranking'].cpu().data.numpy())
				if 'intents' in out_dict:
					true_intent.extend(batch['intents'].cpu().data.numpy())
					predict_intent.extend(out_dict['intents'].cpu().data.numpy())
					session_ids.extend(batch['c_id_c'].cpu().data.numpy())
				if sample and cnt>400:
					break
			prediction_scores = np.array(prediction_scores)
		if len(phase):
			model_path = os.path.dirname(dataset.model.model_path)
			np.save(os.path.join(model_path,phase+"_predintent.npy"),np.array(predict_intent))
			np.save(os.path.join(model_path,phase+"_trueintent.npy"),np.array(true_intent))
			np.save(os.path.join(model_path,phase+"_sessionids.npy"),np.array(session_ids))
		return prediction_scores, ranking_lists, true_intent, predict_intent
	
	def eval_termination(self, criterion):
		if len(criterion) - criterion.index(max(criterion)) > self.early_stop:
			return True
		return False

		
	def get_pairs(self,scores):
		"""
		:param scores: given score list of documents for a particular query
		:return: the documents pairs whose firth doc has a higher value than second one.
		"""
		pairs = []
		for i in range(len(scores)):
			for j in range(len(scores)):
				if scores[i] > scores[j]:
					pairs.append((i, j))
		return pairs

	def compute_lambda_new(self, true_scores, temp_scores, session_len):
		batch_size, max_lens = true_scores.shape
		device = true_scores.device
		positions = torch.arange(max_lens).float()
		valid = (torch.arange(max_lens).to(device)[None,:] < session_len[:,None])
		valid_mask = valid.unsqueeze(2) * valid.unsqueeze(2).transpose(1,2)
		
		discounts = (1/torch.log2(torch.arange(max_lens)+2.0)).to(device)
		ranking_perfect, _ = torch.sort(true_scores, descending=True, dim=-1)
		IDCG = ((torch.pow(2,ranking_perfect)-1) * discounts * valid).sum(dim=-1)

		order_pairs= ((true_scores.unsqueeze(2) - true_scores.unsqueeze(2).transpose(1,2))>0) * valid_mask
		order_pairs_neg =  ((true_scores.unsqueeze(2) - true_scores.unsqueeze(2).transpose(1,2))<0) * valid_mask

		dcg_n = torch.pow(2, true_scores)-1 # batch * max_len
		dcg_d = (1/ torch.log2(positions+2)).to(device) # max_len
		pair_dcg = dcg_n[:,:,None] * dcg_d.unsqueeze(0).unsqueeze(2).transpose(1,2) # batch * max_len * max_len
		single_dcg_t= dcg_n * dcg_d[None,:] # batch * max_len
		
		Delta = (pair_dcg + pair_dcg.transpose(1,2) - single_dcg_t.unsqueeze(2) - single_dcg_t.unsqueeze(2).transpose(1,2)).abs()/IDCG[:,None,None]
		
		score_diff = temp_scores.unsqueeze(2) - temp_scores.unsqueeze(2).transpose(1,2)
		Rho = 1/ (1+torch.exp(score_diff))

		Lambda_i = (Delta * Rho * order_pairs* valid_mask).sum(axis=-1)
		Lambda_j = (Delta.transpose(1,2) * Rho.transpose(1,2) * order_pairs_neg * valid_mask).sum(axis=-1)

		Lambda = Lambda_i - Lambda_j

		return Lambda # Avoid large gradient, leading to inf/nan in output

	def compute_lambda(self,true_scores, temp_scores, session_len, true_scores_torch, temp_scores_torch, session_len_torch,
				bid):
		"""
		:param true_scores: the score list of the documents for the query
		:param temp_scores: the predict score list of the these documents
		:return:
			lambdas: changed lambda value for these documents
			w: w value
		"""
		try:
			order_pairs = self.get_pairs(true_scores[:session_len])
			doc_num = len(true_scores)
			lambdas = np.zeros(doc_num)
			w = np.zeros(doc_num)
			IDCG = idcg(true_scores[:session_len]) # ideal
			single_dcgs = {}
			delta_all = {}
			for i, j in order_pairs: # pair中每一对
				if (i, i) not in single_dcgs:
					single_dcgs[(i, i)] = single_dcg(true_scores, i, i) # 第i个document排在第i个位置时的dcg
				if (j, j) not in single_dcgs:
					single_dcgs[(j, j)] = single_dcg(true_scores, j, j) # 第j个document排在第j个位置时的dcg
				single_dcgs[(i, j)] = single_dcg(true_scores, i, j) # 第i个document与第j个document互换后的dcg
				single_dcgs[(j, i)] = single_dcg(true_scores, j, i)

			for i, j in order_pairs:
				delta = abs(single_dcgs[(i,j)] + single_dcgs[(j,i)] - single_dcgs[(i,i)] -single_dcgs[(j,j)])/IDCG # 互换后的delta
				rho = 1 / (1 + np.exp(temp_scores[i] - temp_scores[j]))
				delta_all[(i,j)] = (delta, rho)
				lambdas[i] += rho * delta # 第i个document的labmda值（量化了一个待排序的文档在下一次迭代时应该调整的方向和强度）
				lambdas[j] -= rho * delta

				rho_complement = 1.0 - rho
				w[i] += rho * rho_complement * delta  # 第i个document的偏导数
				w[j] -= rho * rho_complement * delta
			
		except Exception as e:
			logging.info(e)
			inputs = ''
			while inputs != 'continue':
				try:
					print(eval(inputs))
				except Exception as e:
					print(e)
				inputs = input()

		return lambdas, w 

def dcg(scores):
	"""
	compute the DCG value based on the given score
	:param scores: a score list of documents
	:return v: DCG value
	"""
	v = 0
	for i in range(len(scores)):
		v += (np.power(2, scores[i]) - 1) / np.log2(i+2)  # i+2 is because i starts from 0
	return v


def idcg(scores):
	"""
	compute the IDCG value (best dcg value) based on the given score
	:param scores: a score list of documents
	:return:  IDCG value
	"""
	best_scores = sorted(scores)[::-1]
	return dcg(best_scores)

def single_dcg(scores, i, j):
	"""
	compute the single dcg that i-th element located j-th position
	:param scores:
	:param i:
	:param j:
	:return:
	"""
	return (np.power(2, scores[i]) - 1) / np.log2(j+2)

def ndcg(scores):
	"""
	compute the NDCG value based on the given score
	:param scores: a score list of documents
	:return:  NDCG value
	"""
	return dcg(scores)/idcg(scores)

def ndcg_k(scores, k):
	scores_k = scores[:k]
	dcg_k = dcg(scores_k)
	idcg_k = dcg(sorted(scores)[::-1][:k])
	if idcg_k == 0:
		return np.nan
	return dcg_k/idcg_k


