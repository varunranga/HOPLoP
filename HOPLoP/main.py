import tensorflow.compat.v1 as tensorflow

tensorflow.disable_eager_execution()

config = tensorflow.ConfigProto()
config.gpu_options.allow_growth = True

import pickle

import numpy
import random

from pprint import pprint

from Dataset import Dataset
from HOPLOP import HOPLOP

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset', type = str, default = "NELL995")
parser.add_argument('-t', '--task', type = str, default = "concept_athleteplaysforteam")
parser.add_argument('-e', '--load-embedding', type = str, default = "NELL995_Embeddings.bin")
parser.add_argument('-x', '--save-result', type = str, default = None)
parser.add_argument('-a', '--evaluation-type', type = str, default = 'prob')
parser.add_argument('-p', '--patience', type = int, default = 100)
parser.add_argument('-r', '--learning-rate', type = float, default = 1e-3)
parser.add_argument('-s', '--save', type = str, default = None)
parser.add_argument('-l', '--load', type = str, default = None)
parser.add_argument('-o', '--hops', type = int, default = 10)
parser.add_argument('-c', '--batch-size', type = int, default = 8)
parser.add_argument('-n', '--network', nargs='*', default = ["1000", "relu"])
parser.add_argument('-sd', '--seed', type = int, default = None)
args = parser.parse_args()

if args.seed == None:
	args.seed = numpy.random.randint(low = 0, high = 2**32)

tensorflow.reset_default_graph()

random.seed(args.seed)
numpy.random.seed(args.seed)
tensorflow.random.set_random_seed(args.seed)

print("Seed =", args.seed)

dataset = Dataset(dataset = args.dataset, task = args.task)

embedding = pickle.load(open(args.load_embedding, "rb"))

if args.load:

	weights = pickle.load(open(args.load, "rb"))

else:

	with tensorflow.Session(config = config) as session:

		relation_network_units = [x for x in args.network if x.isnumeric()]
		relation_network_activations = [x for x in args.network if not x.isnumeric()]

		model = HOPLOP(dataset = dataset, embedding = embedding, relation_network_units = relation_network_units, relation_network_activations = relation_network_activations, max_hops = args.hops, batch_size = args.batch_size, patience = args.patience, learning_rate = args.learning_rate, evaluation_type = args.evaluation_type)

		session.run(tensorflow.global_variables_initializer())

		weights = model.fit(save_weights = args.save)

tensorflow.reset_default_graph()

with tensorflow.Session(config = config) as session:

	relation_network_units = [x for x in args.network if x.isnumeric()]
	relation_network_activations = [x for x in args.network if not x.isnumeric()]

	model = HOPLOP(dataset = dataset, embedding = embedding, relation_network_units = relation_network_units, relation_network_activations = relation_network_activations, max_hops = args.hops, batch_size = args.batch_size, patience = args.patience, learning_rate = args.learning_rate, evaluation_type = args.evaluation_type)

	session.run(tensorflow.global_variables_initializer())

	model.load_weights(weights)

	result = model.evaluate(type_of_split = 'test')

	print(args.dataset, args.task, "| MAP Score:", result['map_score'])

	if args.save_result:
		pickle.dump(result, open(args.save_result, "wb"))
