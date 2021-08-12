import tensorflow.compat.v1 as tensorflow

tensorflow.disable_eager_execution()

config = tensorflow.ConfigProto()
config.gpu_options.allow_growth = True

import random
import numpy

import pickle

from Dataset import Dataset
from Embeddings.TransE import TransE
from Embeddings.ComplEx import ComplEx
from Embeddings.TuckER import TuckER

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset', type = str, default = "NELL995")
parser.add_argument('-t', '--task', type = str, default = "concept_agentbelongstoorganization")
parser.add_argument('-x', '--embedding-method', type = str, default = "TransE")
parser.add_argument('-e', '--embedding-size', type = int, default = 100)
parser.add_argument('-m', '--margin', type = float, default = 1.0)
parser.add_argument('-r', '--learning-rate', type = float, default = 1e-3)
parser.add_argument('-b', '--batch-size', type = int, default = 1024)
parser.add_argument('-g', '--sampling-type', type = str, default = 'bernoulli')
parser.add_argument('-p', '--patience', type = int, default = 100)
parser.add_argument('-l', '--load', type = str, default = None)
parser.add_argument('-s', '--save', type = str, default = None)
parser.add_argument('-sd', '--seed', type = int, default = None)
args = parser.parse_args()

if args.seed == None:
	args.seed = numpy.random.randint(low = 0, high = 2**32)

print("Seed =", args.seed)

pretrained_embeddings = pickle.load(open(args.load, "rb"))

dataset = Dataset(dataset = args.dataset, task = args.task)

tensorflow.reset_default_graph()

random.seed(args.seed)
numpy.random.seed(args.seed)
tensorflow.random.set_random_seed(args.seed)

with tensorflow.Session() as session:

	embedding = eval(args.embedding_method)(dataset = dataset, embedding_size = args.embedding_size, margin = args.margin, learning_rate = args.learning_rate, batch_size = args.batch_size, sampling_type = args.sampling_type, patience = args.patience)
	
	session.run(tensorflow.global_variables_initializer())

	embedding.load_embeddings(embedding = pretrained_embeddings)

	result = embedding.evaluate()

	print(args.dataset, args.task, "| MAP Score:", result['map_score'])

	if args.save:
		pickle.dump(result, open(args.save, "wb"))
