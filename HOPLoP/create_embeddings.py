import tensorflow.compat.v1 as tensorflow

tensorflow.disable_eager_execution()

config = tensorflow.ConfigProto()
config.gpu_options.allow_growth = True

import random
import numpy

from Dataset import Dataset
from Embeddings.TransE import TransE
from Embeddings.ComplEx import ComplEx
from Embeddings.TuckER import TuckER

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset', type = str, default = "NELL995")
parser.add_argument('-x', '--embedding-method', type = str, default = "TransE")
parser.add_argument('-e', '--embedding-size', type = int, default = 100)
parser.add_argument('-m', '--margin', type = float, default = 1.0)
parser.add_argument('-r', '--learning-rate', type = float, default = 1e-3)
parser.add_argument('-b', '--batch-size', type = int, default = 1024)
parser.add_argument('-g', '--sampling-type', type = str, default = 'bernoulli')
parser.add_argument('-p', '--patience', type = int, default = 100)
parser.add_argument('-s', '--save', type = str, default = None)
parser.add_argument('-sd', '--seed', type = int, default = None)
args = parser.parse_args()

if args.seed == None:
	args.seed = numpy.random.randint(low = 0, high = 2**32)

print("Seed =", args.seed)

if args.dataset == "NELL995":
	task = "concept_agentbelongstoorganization"
elif args.dataset == "FB15K237":
	task = "sports@sports_team@sport"
elif args.dataset == "WN18RR":
	task = "_hypernym"
elif args.dataset == "YAGO310":
	task = "isAffiliatedTo"

dataset = Dataset(dataset = args.dataset, task = task)

tensorflow.reset_default_graph()

random.seed(args.seed)
numpy.random.seed(args.seed)
tensorflow.random.set_random_seed(args.seed)

with tensorflow.Session() as session:

	embedding = eval(args.embedding_method)(dataset = dataset, embedding_size = args.embedding_size, margin = args.margin, learning_rate = args.learning_rate, batch_size = args.batch_size, sampling_type = args.sampling_type, patience = args.patience)
	
	session.run(tensorflow.global_variables_initializer())

	embedding.fit(save_embedding = args.save)
