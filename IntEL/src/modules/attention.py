# coding=utf-8

import torch
import numpy as np

"""
本文件实现了一些注意力机制，方便对变长的向量序列进行聚合等操作。
from HappyRec
"""

def single_query_att_func(q, k, v, valid=None, scale=None):
	"""
	基本的注意力函数。要求query和key已经准备好，q和k直接相乘。
	:param query: ? * 1 * a，Query张量
	:param key: ? * l * a，Keys张量
	:param value: ? * l * v，Values张量，被聚合的对象
	:param valid: ? * l，哪些value是合法的，1表示合法，0表示非法
	:return: ? * v，聚合后的张量。? * l，注意力权重。
	"""
	att_v = (q * k).sum(dim=-1)  # ? * l
	if scale is not None:
		att_v = att_v * scale  # ? * l
	att_v = att_v - att_v.max(dim=-1, keepdim=True)[0]  # ? * l
	if valid is not None:
		att_v = att_v.masked_fill(valid.le(0), -np.inf)  # ? * l
	att_w = att_v.softmax(dim=-1)  # ? * l
	att_w = att_w.masked_fill(torch.isnan(att_w), 0)  # ? * l
	result = (att_w.unsqueeze(dim=-1) * v).sum(dim=-2)  # ? * v
	return result, att_w

class SingleQueryAtt(torch.nn.Module):
	"""
	最基本的注意力聚合。 a = h\sigma(wv+b); softmax(a);
	"""

	def __init__(self, input_size, att_size, act_func=torch.nn.ReLU):
		super().__init__()
		self.input_size = input_size
		self.att_size = att_size
		self.act_func = act_func
		self.init_modules()

	def init_modules(self):
		self.attention_layers = torch.nn.Sequential(
			torch.nn.Linear(self.input_size, self.att_size),
			self.act_func(),
			torch.nn.Linear(self.att_size, 1, bias=False)
		)

	def forward(self, v, valid=None, scale=None):
		"""
		:param value: ? * l * v，Values张量，被聚合的对象
		:param valid: ? * l，哪些value是合法的，1表示合法，0表示非法
		:return: ? * v，聚合后的张量。? * l，注意力权重。
		"""
		return single_query_att_func(q=self.attention_layers(v), k=1, v=v, valid=valid, scale=scale)


class MultiQueryAtt(torch.nn.Module):
	def forward(self, q, k, v, valid=None, scale=None):
		"""
		根据q和k的两两匹配程度，把k对应的v加权平均。可以有多个Query，每个query分别匹配key聚合出一个向量。
		:param q: Queries张量，? * L_q * a
		:param k: Keys张量，? * L_k * a
		:param v: Values张量，? * L_k * V
		:param scale: 缩放因子，浮点标量
		:param valid: ? * L_q * L_k，哪些value是合法的，1表示合法，0表示非法
		:return ? * L_q * V，每个Query聚合出的向量。? * L_q * L_k，注意力权重。
		"""
		att_v = torch.matmul(q, k.transpose(-1, -2))  # ? * L_q * L_k
		if scale is not None:
			att_v = att_v * scale  # ? * L_q * L_k
		att_v = att_v - att_v.max(dim=-1, keepdim=True)[0]  # ? * L_q * L_k
		if valid is not None:
			att_v = att_v.masked_fill(valid.le(0), -np.inf)  # ? * L_q * L_k
		att_w = att_v.softmax(dim=-1)  # ? * L_q * L_k
		att_w = att_w.masked_fill(torch.isnan(att_w), 0)  # ? * L_q * L_k
		result = torch.matmul(att_w, v)  # ? * L_q * V
		return result, att_w


class SelfAtt(torch.nn.Module):
	"""
	自注意力。 Q = WV; K = WV; V = WV; A = softmax((Q * K^T)/\sqrt(d)); V = AV
	"""

	def __init__(self, input_size, query_size=-1, key_size=-1, value_size=-1):
		"""
		初始化函数。
		:param input_size: 输入Values向量长度
		:param query_size: Query向量长度。>=0表示需要做一层变换，=0表示变换后维度大小和input_size相同。
		:param key_size: Key向量长度。>=0表示需要做一层变换，=0表示变换后维度大小和input_size相同。
		:param value_size: Value向量长度。>=0表示需要做一层变换，=0表示变换后维度大小和input_size相同。
		"""
		super().__init__()
		self.input_size = input_size
		self.query_size = query_size if query_size != 0 else self.input_size
		self.key_size = key_size if key_size != 0 else self.input_size
		self.value_size = value_size if value_size != 0 else self.input_size
		assert self.query_size == self.key_size \
				or (self.query_size < 0 and self.key_size == self.input_size) \
				or (self.key_size < 0 and self.query_size == self.input_size)  # Query和Key的向量维度需要匹配
		self.att_size = input_size
		if self.query_size > 0:
			self.att_size = self.query_size
		if self.key_size > 0:
			self.att_size = self.key_size

		self.init_modules()

	def init_modules(self):
		if self.query_size >= 0:
			self.query_layer = torch.nn.Linear(self.input_size, self.att_size, bias=False)
		else:
			self.query_layer = None
		if self.key_size >= 0:
			self.key_layer = torch.nn.Linear(self.input_size, self.att_size, bias=False)
		else:
			self.key_layer = None
		if self.value_size >= 0:
			self.value_layer = torch.nn.Linear(self.input_size, self.value_size, bias=False)
		else:
			self.value_layer = None

	def forward(self, x, valid=None, scale=None, act_v=None):
		"""
		:param x: ? * L * a，自注意力输入向量。
		:param valid: ? * L，哪些value是合法的，1表示合法，0表示非法。
		:param scale: 缩放因子，浮点标量
		:param act_v: 对V做变换成QKV时所使用的激活函数，None表示不用。
		:return: ? * L * v，一层自注意力后的向量。? * L(Q) * L(V)，自注意力权重。
		"""

		def transfer_if_valid_layer(layer):
			result = x
			if layer is not None:
				result = layer(x)
				if act_v is not None:
					result = act_v(result)
			return result
		att_query = transfer_if_valid_layer(self.query_layer)  # ? * L * a
		att_key = transfer_if_valid_layer(self.key_layer)  # ? * L * a
		att_value = transfer_if_valid_layer(self.value_layer)  # ? * L * v
		return MultiQueryAtt()(q=att_query, k=att_key, v=att_value, scale=scale, valid=valid)

class CrossAtt(torch.nn.Module):
	"""
	Cross attention
	"""
	def __init__(self, input_qsize, input_vsize, query_size=-1, key_size=-1, value_size=-1):
		"""
		初始化函数。
		:param input_qsize: 输入query向量长度 
		:param input_vsize: 输入Values向量长度
		:param query_size: Query向量长度。>=0表示需要做一层变换，=0表示变换后维度大小和input_size相同。
		:param key_size: Key向量长度。>=0表示需要做一层变换，=0表示变换后维度大小和input_size相同。
		:param value_size: Value向量长度。>=0表示需要做一层变换，=0表示变换后维度大小和input_size相同。
		"""
		super().__init__()
		self.input_qsize = input_qsize
		self.input_vsize = input_vsize
		self.query_size = query_size if query_size != 0 else self.input_qsize
		self.key_size = key_size if key_size != 0 else self.input_qsize
		self.value_size = value_size if value_size != 0 else self.input_vsize
		assert self.query_size == self.key_size \
				or (self.query_size < 0 and self.key_size == self.input_size) \
				or (self.key_size < 0 and self.query_size == self.input_size)  # Query和Key的向量维度需要匹配
		self.att_size = input_qsize
		if self.query_size > 0:
			self.att_size = self.query_size
		if self.key_size > 0:
			self.att_size = self.key_size

		self.init_modules()

	def init_modules(self):
		if self.query_size >= 0:
			self.query_layer = torch.nn.Linear(self.input_qsize, self.att_size, bias=False)
		else:
			self.query_layer = None
		if self.key_size >= 0:
			self.key_layer = torch.nn.Linear(self.input_vsize, self.att_size, bias=False)
		else:
			self.key_layer = None
		if self.value_size >= 0:
			self.value_layer = torch.nn.Linear(self.input_vsize, self.value_size, bias=False)
		else:
			self.value_layer = None

	def forward(self, query, x, valid=None, scale=None, act_v=None):
		"""
		:param query: ? * L1 * a, Cross attention中的查询向量
		:param x: ? * L2 * a，Cross attention输入向量。
		:param valid: ? * L1 * L2，Cross attention后哪些value是合法的，1表示合法，0表示非法。
		:param scale: 缩放因子，浮点标量
		:param act_v: 做变换成QKV时所使用的激活函数，None表示不用。
		:return: ? * L1 * v，一层cross注意力后的向量。? * L(Q) * L(V)，cross注意力权重。
		"""

		def transfer_if_valid_layer(layer,input):
			result = input
			if layer is not None:
				result = layer(input)
				if act_v is not None:
					result = act_v(result)
			return result
		att_query = transfer_if_valid_layer(self.query_layer,query)  # ? * L * a
		att_key = transfer_if_valid_layer(self.key_layer,x)  # ? * L * a
		att_value = transfer_if_valid_layer(self.value_layer,x)  # ? * L * v
		return MultiQueryAtt()(q=att_query, k=att_key, v=att_value, scale=scale, valid=valid)


class MultiHeadSelfAtt(SelfAtt):
	"""
	多头自注意力机制。每个头是一个自注意力，每个头的结果聚合后作为最终结果。
	"""

	def __init__(self, input_size, query_size=-1, key_size=-1, value_size=-1, head_num=1):
		self.head_num = head_num
		SelfAtt.__init__(self, input_size=input_size, query_size=query_size, key_size=key_size,
						 value_size=value_size)

	def init_modules(self):
		"""
		注意通常论文中或一些实现中是把att_size拆成head_num个，比如向量长度64有8个头是说每个头的向量长度是8，但是可能出现不整除的问题。
		这里我们使用att_size*head_num来描述，因此如果希望Query总向量长度64有8个头，应该初始化参数是query_size=8，head_num=8。
		"""
		if self.query_size > 0:
			self.query_layer = torch.nn.Linear(self.input_size, self.att_size * self.head_num, bias=False)
		else:
			self.query_layer = None
		if self.key_size > 0:
			self.key_layer = torch.nn.Linear(self.input_size, self.att_size * self.head_num, bias=False)
		else:
			self.key_layer = None
		if self.value_size > 0:
			self.value_layer = torch.nn.Linear(self.input_size, self.value_size * self.head_num, bias=False)
		else:
			self.value_layer = None

	def forward(self, x, valid=None, scale=None, act_v=None):
		"""
		注意返回值没有自行聚合head那一维，这一维是拼接是平均或求和由外部自定义操作，这里不做限制。
		:param x: ? * L * a，自注意力输入向量。
		:param valid: ? * L，哪些value是合法的，1表示合法，0表示非法。
		:param scale: 缩放因子，浮点标量
		:param act_v: 对V做变换成QKV时所使用的激活函数，None表示不用。
		:return: ? * h * L * v，一层自注意力后的向量。? * h * L(Q) * L(V)，自注意力权重。
		"""
		head_x = torch.cat([x] * self.head_num, dim=-1)  # ? * L * (V*h)
		if valid is not None:
			valid = torch.cat([valid.unsqueeze(dim=-3)] * self.head_num, dim=-3)  # ? * h * L * L

		def transfer_if_valid_layer(layer, head_size):
			result = head_x
			if layer is not None:
				result = layer(x)
				if act_v is not None:
					result = act_v(result)
			result = torch.stack(result.split(head_size, dim=-1), dim=-3)
			return result

		att_query = transfer_if_valid_layer(self.query_layer, self.att_size)  # ? * h * L * a
		att_key = transfer_if_valid_layer(self.key_layer, self.att_size)  # ? * h * L * a
		att_value = transfer_if_valid_layer(
			self.value_layer, self.value_size if self.value_size > 0 else self.input_size)  # ? * h * L * v
		return MultiQueryAtt()(q=att_query, k=att_key, v=att_value, scale=scale, valid=valid)  # ? * h * L * v