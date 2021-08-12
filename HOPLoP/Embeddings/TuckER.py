import tensorflow.compat.v1 as tensorflow
import pickle
import numpy
from tqdm import tqdm
from math import ceil
from math import inf

class TuckER():

	def __init__(self, dataset, embedding_size = 100, margin = 1.0, learning_rate = 1e-3, batch_size = 1024, sampling_type = 'bernoulli', patience = 50):

		self.train_loss_history = []
		self.validation_loss_history = []
		self.dataset = dataset
		self.entity_count = dataset.entity_count
		self.relation_count = dataset.relation_count
		self.embedding_size = embedding_size
		self.margin = margin
		self.learning_rate = learning_rate
		self.batch_size = batch_size
		self.patience = patience
		self.sampling_type = sampling_type
		self._labels = {}

		self.entity_embeddings = tensorflow.get_variable(name = "entity_embeddings", shape = [self.entity_count, self.embedding_size])
		self.relation_embeddings = tensorflow.get_variable(name = "relation_embeddings", shape = [self.relation_count, self.embedding_size])
		self.weight = tensorflow.get_variable(name = "weight_matrix", shape = [self.embedding_size, self.embedding_size, self.embedding_size])

		self.inp_dropout = tensorflow.keras.layers.Dropout(rate = 0.3)
		self.hidden_dropout1 = tensorflow.keras.layers.Dropout(rate = 0.4)
		self.hidden_dropout2 = tensorflow.keras.layers.Dropout(rate = 0.5)

		self.head_id = tensorflow.placeholder(tensorflow.int32)
		self.tail_id = tensorflow.placeholder(tensorflow.int32)
		self.relation_id = tensorflow.placeholder(tensorflow.int32)

		self.label = tensorflow.placeholder(tensorflow.float32)

		head_embedding_vector = tensorflow.nn.embedding_lookup(self.entity_embeddings, self.head_id)
		head_embedding_vector = tensorflow.nn.l2_normalize(head_embedding_vector, axis = 1)
		head_embedding_vector = self.inp_dropout(head_embedding_vector)
		head_embedding_vector = tensorflow.reshape(head_embedding_vector, [-1, 1, self.embedding_size])

		relation_embedding_vector = tensorflow.nn.embedding_lookup(self.relation_embeddings, self.relation_id)
		W_mat = tensorflow.matmul(relation_embedding_vector, tensorflow.reshape(self.weight, [self.embedding_size, -1]))
		W_mat = tensorflow.reshape(W_mat, [-1, self.embedding_size, self.embedding_size])
		W_mat = self.hidden_dropout1(W_mat)

		x = tensorflow.matmul(head_embedding_vector, W_mat)
		x = tensorflow.reshape(x, [-1, self.embedding_size])
		x = tensorflow.nn.l2_normalize(x, axis = 1)
		x = self.hidden_dropout2(x)
		x = tensorflow.matmul(x, self.entity_embeddings, transpose_b = True)
		out = tensorflow.nn.sigmoid(x)

		indices = tensorflow.concat([tensorflow.reshape(tensorflow.range(tensorflow.size(self.head_id)), (-1, 1)), tensorflow.reshape(self.tail_id, (-1, 1))], axis = 1)

		predict = out

		self.predict = -tensorflow.reshape(tensorflow.gather_nd(out, indices), (-1,))
	
		bce = tensorflow.keras.losses.BinaryCrossentropy()

		self.loss = bce(self.label, predict) 

		self.optimizer = tensorflow.train.AdamOptimizer(learning_rate = self.learning_rate)
		self.learn = self.optimizer.minimize(self.loss)

	def fit(self, save_embedding):

		self.create_embeddings(data = self.dataset.graph)

		dct = self.create_dictionary()

		if save_embedding:

			pickle.dump(dct, open(save_embedding, "wb"))

		return dct

	def create_embeddings(self, data):

		self.data = data

		if self.sampling_type == 'bernoulli':

			self._bernoulli_probablities = {}
			for i in range(self.relation_count):
				data = self.data[self.data[:, 1] == i]
				head_entities = set(data[:, 0])
				tph = sum(len(data[data[:, 0] == head_entity]) for head_entity in head_entities) / len(head_entities)
				tail_entities = set(data[:, 2])
				hpt = sum(len(data[data[:, 2] == tail_entity]) for tail_entity in tail_entities) / len(tail_entities)				
				self._bernoulli_probablities[i] = tph / (hpt + tph)

		waited = 0
		best_loss = inf
		epoch = 0

		while (waited < self.patience) and (epoch < 100):

			print()
			print("EPOCH", epoch + 1)
			print("-"*79)
			
			print("TRAINING")
			
			train_generator = self.create_data_generator().__iter__()
			
			training_step = 0
			training_loss = 0
			
			for _ in tqdm(range(ceil(len(self.data) / self.batch_size))):

				training_loss += self.train_model(*train_generator.__next__())
				training_step += 1

			print("Train loss:", training_loss)

			if training_loss < best_loss:
				best_loss = training_loss
				waited = 0
				
				self.store_embeddings()
			else:
				waited += 1

			epoch += 1

			print("-"*79)
			print()

	def create_data_generator(self, positive = True, negative = True):

		data = self.data

		if self.sampling_type == 'uniform':
			
			for i in range(0, len(data), self.batch_size):
				
				positive_heads = data[i:i+self.batch_size, 0]
				positive_tails = data[i:i+self.batch_size, 2]
				positive_relations = data[i:i+self.batch_size, 1]

				if negative == False:
					yield positive_heads, positive_tails, positive_relations

				if negative == True:
					negative_heads = positive_heads.copy()
					negative_relations = positive_relations.copy()
					negative_tails = positive_tails.copy()

					for j in range(len(positive_heads)):
						if numpy.random.rand() < 0.5:
							negative_heads[j] = numpy.random.randint(0, self.entity_count)
						else:
							negative_tails[j] = numpy.random.randint(0, self.entity_count)

					if positive == False:
						yield negative_heads, negative_tails, negative_relations

					if positive == True:
						yield positive_heads, positive_tails, positive_relations, negative_heads, negative_tails, negative_relations

		elif self.sampling_type == 'bernoulli':
			
			for i in range(0, len(data), self.batch_size):

				positive_heads = data[i:i+self.batch_size, 0]
				positive_tails = data[i:i+self.batch_size, 2]
				positive_relations = data[i:i+self.batch_size, 1]

				if negative == False:
					yield positive_heads, positive_tails, positive_relations

				if negative == True:
					negative_heads = positive_heads.copy()
					negative_relations = positive_relations.copy()
					negative_tails = positive_tails.copy()

					for j in range(len(positive_heads)):

						if numpy.random.rand() < self._bernoulli_probablities[negative_relations[j]]:
							negative_heads[j] = numpy.random.randint(0, self.entity_count)
						else:
							negative_tails[j] = numpy.random.randint(0, self.entity_count)

					if positive == False:
						yield negative_heads, negative_tails, negative_relations
					
					if  positive == True:
						yield positive_heads, positive_tails, positive_relations, negative_heads, negative_tails, negative_relations	

	def train_model(self, positive_heads, positive_tails, positive_relations, negative_heads, negative_tails, negative_relations):

		session = tensorflow.get_default_session()

		# heads = numpy.concatenate([positive_heads, negative_heads], axis = 0)
		# tails = numpy.concatenate([positive_tails, negative_tails], axis = 0)
		# relations = numpy.concatenate([positive_relations, negative_relations], axis = 0)

		heads = positive_heads
		tails = positive_tails
		relations = positive_relations
		
		labels = []
		for i in range(len(heads)):
			if ((heads[i], relations[i]) not in self._labels):
				label = numpy.zeros((1, self.dataset.entity_count), dtype = numpy.float32)
				for e in self.dataset.graph[numpy.logical_and(self.dataset.graph[:, 0] == heads[i], self.dataset.graph[:, 1] == relations[i])][:, -1]:
					label[0, e] = 1.0
				self._labels[(heads[i], relations[i])] = label
			labels.append(self._labels[(heads[i], relations[i])])

		labels = numpy.concatenate(labels, axis = 0)

		feed_dict = {
						self.head_id: heads,
						self.tail_id: tails,
						self.relation_id: relations,
						self.label: labels
					}

		_, loss = session.run([self.learn, self.loss], feed_dict = feed_dict)

		self.train_loss_history.append(loss)

		return loss

	def test_model(self, positive_heads, positive_tails, positive_relations):

		session = tensorflow.get_default_session()

		feed_dict = {
						self.head_id: positive_heads,
						self.tail_id: positive_tails,
						self.relation_id: positive_relations
					}

		loss = session.run(self.predict, feed_dict = feed_dict)
		loss = numpy.reshape(loss, (-1,))

		return loss

	def store_embeddings(self):

		self.best_entity_embeddings = self.entity_embeddings.eval()
		self.best_relation_embeddings = self.relation_embeddings.eval()
		self.best_weight_matrix = self.weight.eval()

	def load_embeddings(self, embedding):

		session = tensorflow.get_default_session()

		tensorflow.assign(self.entity_embeddings, embedding['entity_embeddings']).eval()
		tensorflow.assign(self.relation_embeddings, embedding['relation_embeddings']).eval()
		tensorflow.assign(self.weight, embedding['weight_matrix']).eval()

	def create_dictionary(self):

		dct = {
			'train_loss_history': self.train_loss_history,
			'embedding_size': self.embedding_size,
			'margin': self.margin,
			'learning_rate': self.learning_rate,
			'batch_size': self.batch_size,
			'sampling_type': self.sampling_type,
			'patience': self.patience,
			'data': self.data,
			'entity_embeddings': self.best_entity_embeddings,
			'relation_embeddings': self.best_relation_embeddings,
			'weight_matrix': self.best_weight_matrix
		}

		return dct

	def evaluate(self, type_of_split = 'test'):

		session = tensorflow.get_default_session()

		data = self.dataset.dict_test if type_of_split == 'test' else self.dataset.dict_train

		if self.dataset.dataset == "NELL995":
			task = self.dataset.task.replace("_", ":")
		elif self.dataset.dataset == "FB15K237":
			task = "/" + self.dataset.task.replace("@", "/")
		elif self.dataset.dataset == "WN18RR":
			task = self.dataset.task
		elif self.dataset.dataset == "YAGO310":
			task = self.dataset.task

		self.results = {}
		self.results['for_each_query_entity'] = {}
		
		for query_entity in tqdm(data):

			positive_target_entities = list(data[query_entity]['positive'])

			heads = numpy.array([query_entity] * len(positive_target_entities))
			relations = numpy.array([self.dataset.relation2id[task]] * len(positive_target_entities))
			tails = numpy.array(positive_target_entities)
			
			positive_predict = numpy.reshape(-1 * self.test_model(heads, tails, relations), (-1,)).tolist()
			positive_label = [1] * len(positive_target_entities)

			negative_target_entities = list(data[query_entity]['negative'])

			heads = numpy.array([query_entity] * len(negative_target_entities))
			relations = numpy.array([self.dataset.relation2id[task]] * len(negative_target_entities))
			tails = numpy.array(negative_target_entities)
			
			negative_predict = numpy.reshape(-1 * self.test_model(heads, tails, relations), (-1,)).tolist()
			negative_label = [0] * len(negative_target_entities)

			y_score = positive_predict + negative_predict
			y_true = positive_label + negative_label

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
			
			self.results['for_each_query_entity'][query_entity] = {
				'y_score': y_score,
				'y_true': y_true,
				'target_entities': positive_target_entities + negative_target_entities,
				'ap_score': ap_score
			}

		ap_scores = numpy.array([self.results['for_each_query_entity'][query_entity]['ap_score'] for query_entity in self.results['for_each_query_entity']])

		self.results['map_score'] = numpy.mean(ap_scores)

		return self.results
