import tensorflow.compat.v1 as tensorflow
import numpy
import random
import pickle
from tqdm import tqdm
from math import inf
from math import ceil
import heapq
import itertools
from pprint import pprint

class MHOPLOP():

	def __init__(self, dataset, embedding, relation_network_units = [1000], relation_network_activations = ['relu'], max_hops = 10, batch_size = 8, patience = 100, learning_rate = 1e-3):

		self.max_hops = max_hops
		self.dataset = dataset
		self.embedding = embedding
		self.batch_size = batch_size
		self.patience = patience
		self.learning_rate = learning_rate

		self.positive_store_relation_distribution = []
		self.positive_store_entity_distribution = []

		self.negative_store_relation_distribution = []
		self.negative_store_entity_distribution = []

		self.positive_path = []
		self.negative_path = []

		self.entity_embeddings = tensorflow.constant(embedding['entity_embeddings'], name = 'entity_embeddings')
		self.relation_embeddings = tensorflow.constant(embedding['relation_embeddings'], name = 'relation_embeddings')

		self.relation_network_layers = []

		for units, activation in zip(relation_network_units, relation_network_activations):
			self.relation_network_layers.append(tensorflow.keras.layers.Dense(units = units, activation = activation))
		self.relation_network_layers.append(tensorflow.keras.layers.Dense(units = embedding['embedding_size'], activation = 'linear'))

		self.positive_query_entity_id = tensorflow.placeholder(tensorflow.int32)
		self.negative_query_entity_id = tensorflow.placeholder(tensorflow.int32)
		self.positive_target_entity_id = tensorflow.placeholder(tensorflow.int32)
		self.negative_target_entity_id = tensorflow.placeholder(tensorflow.int32)

		self.task_id = tensorflow.placeholder(tensorflow.int32)
		
		positive_target_entity_embedding = tensorflow.nn.embedding_lookup(self.entity_embeddings, self.positive_target_entity_id)
		negative_target_entity_embedding = tensorflow.nn.embedding_lookup(self.entity_embeddings, self.negative_target_entity_id)

		positive_query_entity_embedding = tensorflow.nn.embedding_lookup(self.entity_embeddings, self.positive_query_entity_id)
		negative_query_entity_embedding = tensorflow.nn.embedding_lookup(self.entity_embeddings, self.negative_query_entity_id)

		positive_current_entity_embedding = tensorflow.nn.embedding_lookup(self.entity_embeddings, self.positive_query_entity_id)
		negative_current_entity_embedding = tensorflow.nn.embedding_lookup(self.entity_embeddings, self.negative_query_entity_id)

		def do_one_edge_traversal(current_entity_embedding, target_entity_embedding):

			input_relation_network = tensorflow.concat([current_entity_embedding, target_entity_embedding], axis = 1)
			input_relation_network = tensorflow.reshape(input_relation_network, (-1, 2 * embedding['embedding_size']))

			net = tensorflow.keras.Input(tensor = input_relation_network)
			for layer in self.relation_network_layers:
				net = layer(net)

			relation_network_output = net

			next_entity_embedding = tensorflow.add(current_entity_embedding, relation_network_output)
			
			return relation_network_output, next_entity_embedding

		self.relation_constraint_losses = []

		for hop in range(self.max_hops):
			
			positive_relation_network_output, positive_next_entity_embedding = do_one_edge_traversal(positive_current_entity_embedding, positive_target_entity_embedding)
			negative_relation_network_output, negative_next_entity_embedding = do_one_edge_traversal(negative_current_entity_embedding, negative_target_entity_embedding)

			self.positive_path.append(positive_relation_network_output)
			self.negative_path.append(negative_relation_network_output)

			def get_relation_distribution(current_relation_embedding):
				
				relation_embeddings = tensorflow.concat([self.relation_embeddings, numpy.zeros((1, self.embedding['embedding_size']))], axis = 0)
				
				return -tensorflow.linalg.norm(relation_embeddings - current_relation_embedding, axis = 1)

			self.positive_store_relation_distribution.append(tensorflow.map_fn(get_relation_distribution, positive_relation_network_output))
			self.negative_store_relation_distribution.append(tensorflow.map_fn(get_relation_distribution, negative_relation_network_output))

			positive_current_entity_embedding = positive_next_entity_embedding	
			negative_current_entity_embedding = negative_next_entity_embedding
			
			def get_entity_distribution(current_entity_embedding):
		
				return -tensorflow.linalg.norm(self.entity_embeddings - current_entity_embedding, axis = 1)

			self.positive_store_entity_distribution.append(tensorflow.map_fn(get_entity_distribution, positive_current_entity_embedding))
			self.negative_store_entity_distribution.append(tensorflow.map_fn(get_entity_distribution, negative_current_entity_embedding))

		positive_distance_to_target_per_example = tensorflow.reduce_sum((positive_target_entity_embedding - positive_current_entity_embedding)**2, axis = 1)
		negative_distance_to_target_per_example = tensorflow.reduce_sum((negative_target_entity_embedding - negative_current_entity_embedding)**2, axis = 1)

		self.positive_distance_to_target = tensorflow.reduce_sum(positive_distance_to_target_per_example, axis = 0)
		self.negative_distance_to_target = tensorflow.reduce_sum(negative_distance_to_target_per_example, axis = 0)
		self.distance_to_target_loss = self.positive_distance_to_target + self.negative_distance_to_target

		self.RNN = tensorflow.keras.layers.LSTM(units = embedding['embedding_size'])
		self.classifier = tensorflow.keras.layers.Dense(units = self.dataset.n_tasks, activation = 'sigmoid')

		positive_path = tensorflow.reshape(tensorflow.convert_to_tensor(self.positive_path), (-1, self.max_hops, embedding['embedding_size']))
		negative_path = tensorflow.reshape(tensorflow.convert_to_tensor(self.negative_path), (-1, self.max_hops, embedding['embedding_size']))

		path = tensorflow.concat([positive_path, negative_path], axis = 0)

		input_link_prediction_network = tensorflow.keras.Input(tensor = path)
		net = self.RNN(input_link_prediction_network)
		output = self.classifier(net)
		
		indices = tensorflow.concat([tensorflow.reshape(tensorflow.range(tensorflow.shape(self.positive_query_entity_id)[0] + tensorflow.shape(self.negative_query_entity_id)[0]), (-1, 1)), tensorflow.reshape(self.task_id, (-1, 1))], axis = 1)
		predict = tensorflow.gather_nd(output, indices)

		labels = tensorflow.concat([tensorflow.ones(tensorflow.shape(self.positive_query_entity_id)[0], dtype = tensorflow.float32), tensorflow.zeros(tensorflow.shape(self.negative_query_entity_id)[0], dtype = tensorflow.float32)], axis = 0)

		bce = tensorflow.keras.losses.BinaryCrossentropy()
		self.classification_loss = bce(labels, predict)

		self.predict = predict
		
		self.loss = self.distance_to_target_loss + self.classification_loss

		self.optimizer = tensorflow.train.AdamOptimizer(learning_rate = learning_rate)

		self.learn = self.optimizer.minimize(self.loss)

	def fit(self, save_weights = None):

		waited = 0
		best_map_score = self.evaluate(type_of_split = 'test')['map_score']
		print("Test MAP score:", best_map_score)

		self.store_weights()

		epoch = 0

		while waited < self.patience:

			print()
			print("EPOCH", epoch + 1)
			print("-"*79)
			
			print("TRAINING")

			train_generator = self.create_data_generator(type_of_split = 'train').__iter__()

			training_step = 0
			training_loss = 0
			acc = 0

			for _ in tqdm(range(ceil(sum(len(self.dataset.dict_train[task_id]) for task_id in self.dataset.dict_train) / self.batch_size))):

				loss = self.train_model(*train_generator.__next__())
				training_loss += loss
				training_step += 1

			print("Train loss:", training_loss)

			training_map_score = self.evaluate(type_of_split = 'test')['map_score']
			print("Test MAP score:", training_map_score)

			if training_map_score > best_map_score:
				best_map_score = training_map_score

				self.store_weights()

				waited = 0

			else:
				waited += 1

			epoch += 1

			print("-"*79)
			print()

		if save_weights:

			pickle.dump(self.best_weights, open(save_weights, "wb"))

		return self.best_weights

	def load_weights(self, weights):

		for layer, w in zip(self.relation_network_layers, weights['NN']):
			layer.set_weights(w)

		self.RNN.set_weights(weights['RNN'][0])
		self.classifier.set_weights(weights['RNN'][1])

	def store_weights(self):
		
		self.best_weights = {}
		self.best_weights['RNN'] = [self.RNN.get_weights(), self.classifier.get_weights()]
		self.best_weights['NN'] = [layer.get_weights() for layer in self.relation_network_layers]

	def train_model(self, query_entities, positive_target_entities, negative_target_entities, task_ids):

		session = tensorflow.get_default_session()

		feed_dict = {
						self.positive_query_entity_id: query_entities,
						self.negative_query_entity_id: query_entities,
						self.positive_target_entity_id: positive_target_entities,
						self.negative_target_entity_id: negative_target_entities,
						self.task_id: numpy.tile(task_ids, 2)
					}

		_, loss = session.run([self.learn, self.loss], feed_dict = feed_dict)
		
		return loss

	def create_data_generator(self, type_of_split = 'train'):

		data = self.dataset.dict_test if type_of_split == 'test' else self.dataset.dict_train

		query_entities = numpy.zeros((sum([len(data[task_id]) for task_id in data]),), dtype = numpy.int32)
		positive_target_entities = numpy.zeros((sum([len(data[task_id]) for task_id in data]),), dtype = numpy.int32)
		negative_target_entities = numpy.zeros((sum([len(data[task_id]) for task_id in data]),), dtype = numpy.int32)
		task_ids = numpy.zeros((sum([len(data[task_id]) for task_id in data]),), dtype = numpy.int32)

		i = 0
		for task_id in data:
			for query_entity in data[task_id]:
				query_entities[i] = query_entity
				task_ids[i] = task_id

				positive_target_entities[i] = random.choice(list(data[task_id][query_entity]['positive']))

				if len(data[task_id][query_entity]['negative']):
					negative_target_entities[i] = random.choice(list(data[task_id][query_entity]['negative']))
				else:
					negative_target_entities[i] = numpy.random.randint(low = 0, high = self.dataset.entity_count)

				i += 1

		idx = numpy.arange(i)
		numpy.random.shuffle(idx)
			
		for i in range(0, len(idx), self.batch_size):

			yield query_entities[idx[i:i+self.batch_size]], positive_target_entities[idx[i:i+self.batch_size]], negative_target_entities[idx[i:i+self.batch_size]], task_ids[idx[i:i+self.batch_size]]

	def evaluate(self, type_of_split = 'train'):

		session = tensorflow.get_default_session()

		data = self.dataset.dict_test if type_of_split == 'test' else self.dataset.dict_train

		self.results = {}
		
		for task_id in tqdm(self.dataset.id2task):

			self.results[task_id] = {}

			self.results[task_id]['for_each_query_entity'] = {}
			
			for query_entity in tqdm(data[task_id]):

				positive_target_entities = list(data[task_id][query_entity]['positive'])
				negative_target_entities = list(data[task_id][query_entity]['negative'])

				positive_heads = numpy.array([query_entity] * len(positive_target_entities))
				negative_heads = numpy.array([query_entity] * len(negative_target_entities))
				positive_tails = numpy.array(positive_target_entities)
				negative_tails = numpy.array(negative_target_entities)

				print(positive_heads, negative_heads, negative_tails)

				feed_dict = {
								self.positive_query_entity_id: positive_heads,
								self.negative_query_entity_id: negative_heads,
								self.positive_target_entity_id: positive_tails,
								self.negative_target_entity_id: negative_tails,
								self.task_id: numpy.array([task_id] * (len(positive_heads) + len(negative_heads)))
							}

				lst = session.run([self.predict, *self.positive_store_relation_distribution, *self.positive_store_entity_distribution, *self.negative_store_relation_distribution, *self.negative_store_entity_distribution], feed_dict = feed_dict)
				
				predict = lst[0]
				positive_relation_distributions = numpy.array(lst[1:1+self.max_hops])
				positive_entity_distributions = numpy.array(lst[1+self.max_hops:1+(2*self.max_hops)])

				negative_relation_distributions = numpy.array(lst[1+(2*self.max_hops):1+(3*self.max_hops)])
				negative_entity_distributions = numpy.array(lst[1+(3*self.max_hops):1+(4*self.max_hops)])
				
				# if self.evaluation_type == 'prob':

					# results = None
				y_score = numpy.reshape(predict, (-1,)).tolist()

				'''
				elif self.evaluation_type == 'beam':

					heads = numpy.concatenate([positive_heads, negative_heads], axis = 0)
					tails = numpy.concatenate([positive_tails, negative_tails], axis = 0)
					relation_distributions = numpy.concatenate([positive_relation_distributions, negative_relation_distributions], axis = 1)
					entity_distributions = numpy.concatenate([positive_entity_distributions, negative_entity_distributions], axis = 1)
					
					results = self.beam_search(heads, tails, relation_distributions, entity_distributions, return_n_paths = 1)
					y_score = numpy.array([result[0][1] for result in results])

					y_score = [s if s != 0 else -inf for s in y_score]
				'''

				y_true = ([1.0] * len(positive_target_entities)) + ([0.0] * len(negative_target_entities))

				count = list(zip(y_score, y_true))
				count.sort(key=lambda x: x[0], reverse=True)

				ranks = []
				correct = 0
				for idx_, item in enumerate(count):
					if item[1] == 1:
						correct += 1
						ranks.append(correct / (1.0 + idx_))
				if len(ranks) == 0:
					ranks.append(0)
				ap_score = numpy.mean(ranks)
				
				self.results[task_id]['for_each_query_entity'][query_entity] = {
					'y_score': y_score,
					'y_true': y_true,
					# 'results': results,
					'target_entities': positive_target_entities + negative_target_entities,
					'ap_score': ap_score
				}

			ap_scores = numpy.array([self.results[task_id]['for_each_query_entity'][query_entity]['ap_score'] for query_entity in self.results[task_id]['for_each_query_entity']])

			self.results[task_id]['map_score'] = numpy.mean(ap_scores)

		self.results['map_score'] = numpy.mean([self.results[task_id]['map_score'] for task_id in self.results])

		return self.results
