import tensorflow.compat.v1 as tensorflow
import pickle
import numpy
from tqdm import tqdm
from math import ceil
from math import inf

class TransE():

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

		self.entity_embeddings = tensorflow.get_variable(name = "entity_embeddings", shape = [self.entity_count, self.embedding_size])
		self.relation_embeddings = tensorflow.get_variable(name = "relation_embeddings", shape = [self.relation_count, self.embedding_size])

		self.positive_head_id = tensorflow.placeholder(tensorflow.int32)
		self.positive_tail_id = tensorflow.placeholder(tensorflow.int32)
		self.positive_relation_id = tensorflow.placeholder(tensorflow.int32)

		self.negative_head_id = tensorflow.placeholder(tensorflow.int32)
		self.negative_tail_id = tensorflow.placeholder(tensorflow.int32)
		self.negative_relation_id = tensorflow.placeholder(tensorflow.int32)

		positive_head_embedding = tensorflow.nn.embedding_lookup(self.entity_embeddings, self.positive_head_id)
		positive_tail_embedding = tensorflow.nn.embedding_lookup(self.entity_embeddings, self.positive_tail_id)
		positive_relation_embedding = tensorflow.nn.embedding_lookup(self.relation_embeddings, self.positive_relation_id)

		negative_head_embedding = tensorflow.nn.embedding_lookup(self.entity_embeddings, self.negative_head_id)
		negative_tail_embedding = tensorflow.nn.embedding_lookup(self.entity_embeddings, self.negative_tail_id)
		negative_relation_embedding = tensorflow.nn.embedding_lookup(self.relation_embeddings, self.negative_relation_id)

		positive_predict_tail_embedding = tensorflow.add(positive_head_embedding, positive_relation_embedding)
		negative_predict_tail_embedding = tensorflow.add(negative_head_embedding, negative_relation_embedding)

		positive = tensorflow.reduce_sum((positive_predict_tail_embedding - positive_tail_embedding) ** 2, 1, keep_dims = True)
		negative = tensorflow.reduce_sum((negative_predict_tail_embedding - negative_tail_embedding) ** 2, 1, keep_dims = True)
		
		self.predict = positive
	
		self.loss = tensorflow.reduce_sum(tensorflow.maximum(positive - negative + self.margin, 0)) 

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
			for i in tqdm(range(self.relation_count)):
				data = self.data[self.data[:, 1] == i]
				head_entities = set(data[:, 0])
				tph = sum(len(data[data[:, 0] == head_entity]) for head_entity in head_entities) / len(head_entities)
				tail_entities = set(data[:, 2])
				hpt = sum(len(data[data[:, 2] == tail_entity]) for tail_entity in tail_entities) / len(tail_entities)				
				self._bernoulli_probablities[i] = tph / (hpt + tph)

		waited = 0
		best_loss = inf
		epoch = 0

		while (waited < self.patience):

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

		feed_dict = {
						self.positive_head_id: positive_heads,
						self.positive_tail_id: positive_tails,
						self.positive_relation_id: positive_relations,
						self.negative_head_id: negative_heads,
						self.negative_tail_id: negative_tails,
						self.negative_relation_id: negative_relations
					}

		_, loss = session.run([self.learn, self.loss], feed_dict = feed_dict)

		self.train_loss_history.append(loss)

		return loss

	def test_model(self, positive_heads, positive_tails, positive_relations):

		session = tensorflow.get_default_session()

		feed_dict = {
						self.positive_head_id: positive_heads,
						self.positive_tail_id: positive_tails,
						self.positive_relation_id: positive_relations
					}

		loss = session.run(self.predict, feed_dict = feed_dict)

		return loss

	def store_embeddings(self):

		self.best_entity_embeddings = self.entity_embeddings.eval()
		self.best_relation_embeddings = self.relation_embeddings.eval()

	def load_embeddings(self, embedding):

		session = tensorflow.get_default_session()

		tensorflow.assign(self.entity_embeddings, embedding['entity_embeddings']).eval()
		tensorflow.assign(self.relation_embeddings, embedding['relation_embeddings']).eval()

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
