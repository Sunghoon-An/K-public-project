from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras.layers import Dense, LayerNormalization, BatchNormalization, GaussianDropout, Dropout, Add, ELU, Lambda, LeakyReLU, ReLU, Input
from tensorflow.keras.callbacks import ModelCheckpoint,TensorBoard,EarlyStopping
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras import Model
import networkx as nx
import numpy as np
import argparse
import os
import time

class RWN(object):
	def __init__(self, input_shape, num_stages, hidden_size, num_class, graph_model, graph_param):
		"""
		Randomly Wired Neural Nets
		Args
			num_stages : number of stages
			hidden_size : the size of node's output in stage
			graph_model : random graph model,  choose one from ['ws', 'ba', 'er']
			graph_param : parameters when create random graph. In paper, best parameter is [32, 4, 0.75]
			num_class : the number of class to classify
		Return
			RWN model
		"""
		self.input_shape = input_shape
		self.num_stages = num_stages
		self.hidden_size = hidden_size
		self.num_class = num_class
		self.graph_model = graph_model
		self.graph_param = graph_param

	def build_model(self):
		x = Input(shape = (self.input_shape,))
		outputs = self.regime(x, self.num_stages, self.hidden_size, self.num_class, self.graph_model, self.graph_param)
		model = Model(inputs = [x], outputs = [outputs])
		model.build(input_shape=[self.input_shape])

		return model

	def dense_block(self, x, hidden_size):
		"""
		operation in Node
		Args
			x : input data
			hidden_size : Node's output size
		Return
			Node operation
		"""
		x = ReLU()(x)
		x = Dense(hidden_size, kernel_initializer="he_uniform")(x)
		x = BatchNormalization()(x)
		x = Dropout(0.4)(x)
		
		return x

	def build_stage(self, x, hidden_size, graph_data):
		"""
		Create Neural Net using random graph model. this is a stage
		Args
			x : input data
			hidden_size : Node's output size
			graph_data : Returned values after create random graph
		Return
			A neural net, stage
		"""
		
		graph, graph_order, start_node, end_node = graph_data

		interms = {}
		for node in graph_order:
			if node in start_node:
				interm = self.dense_block(x, hidden_size)
				interms[node] = interm
			else:
				in_node = list(nx.ancestors(graph, node))
				if len(in_node) > 1:
					weight = tf.Variable(
						initial_value = tf.keras.initializers.GlorotNormal()(shape=[len(in_node)])
						, name = 'sum_weight'
						, dtype = tf.float32
						, constraint = lambda x : tf.clip_by_value(x, 0, np.infty))
					weight = tf.nn.sigmoid(weight)
					interm = 0
					for idx in range(len(in_node)):
						interm += weight[idx] * interms[in_node[idx]]
					interm = self.dense_block(interm, hidden_size)
					interms[node] = interm
				elif len(in_node) == 1:
					interm = self.dense_block(interms[in_node[0]], hidden_size)
					interms[node] = interm
		output = 0
		for idx in range(len(end_node)):
			output += interms[end_node[idx]]

		return output

	def regime(self, x, num_stages, hidden_size, num_class, graph_model, graph_param):
		"""
		Create total Network. 

		"""
		#### stage 1 ######
		x = Dense(256, kernel_initializer = "he_uniform")(x)
		x = BatchNormalization()(x)
		x = ReLU()(x)
		x = Dropout(0.4)(x)

		#### stage 2 ######
		x = self.dense_block(x, hidden_size)

		#### stage 3 ~ 4 ####
		for stage in range(3, num_stages+1):
			graph_data = self.graph_generator(graph_model
				, graph_param, "graph_model", 'dense'+str(stage)+'_'+graph_model)
			x = self.build_stage(x, hidden_size, graph_data)

		#### stage 5 ######
		x = self.dense_block(x, hidden_size)
		x = Dense(64, kernel_initializer = "he_uniform")(x)
		x = BatchNormalization()(x)
		x = ReLU()(x)
		x = Dense(num_class, activation="softmax", kernel_initializer="glorot_uniform")(x)

		return x

	def graph_generator(self, graph_model, graph_param, save_path, file_name):
		"""
		Create random graph

		"""

		graph_param[0] = int(graph_param[0])
		if graph_model == 'ws':
			graph_param[1] = int(graph_param[1])
			graph = nx.random_graphs.connected_watts_strogatz_graph(*graph_param)
		elif graph_model == 'er':
			graph = nx.random_graphs.erdos_renyi_graph(*graph_param)
		elif graph_model == 'ba':
			graph_param[1] = int(graph_param[1])
			graph = nx.random_graphs.barabasi_albert_graph(*graph_param)

		if os.path.isfile(save_path + '/' + file_name + '.yaml') is True:
			print('graph loaded')
			dgraph = nx.read_yaml(save_path + '/' + file_name + '.yaml')

		else:
			dgraph = nx.DiGraph()
			dgraph.add_nodes_from(graph.nodes)
			dgraph.add_edges_from(graph.edges)

		dgraph = nx.DiGraph()
		dgraph.add_nodes_from(graph.nodes)
		dgraph.add_edges_from(graph.edges)

		in_node = []
		out_node = []
		for indeg, outdeg in zip(dgraph.in_degree, dgraph.out_degree):
			if indeg[1] == 0:
				in_node.append(indeg[0])
			elif outdeg[1] == 0:
				out_node.append(outdeg[0])
		sorted = list(nx.topological_sort(dgraph))

		if os.path.isdir(save_path) is False:
			os.makedirs(save_path)

		if os.path.isfile(save_path + '/' + file_name + '.yaml') is False:
			print('graph_saved')
			nx.write_yaml(dgraph, save_path + '/' + file_name + '.yaml')

		return dgraph, sorted, in_node, out_node
