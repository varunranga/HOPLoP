import numpy

class Dataset():

	def __init__(self, dataset = "NELL995", task = "concept_athleteplaysforteam"):

		self._all_tasks = {
			"NELL995": {
				"concept_agentbelongstoorganization",
				"concept_athleteplaysinleague",
				"concept_organizationhiredperson",
				"concept_teamplaysinleague",
				"concept_athletehomestadium",
				"concept_athleteplayssport",
				"concept_personborninlocation",
				"concept_teamplayssport",
				"concept_athleteplaysforteam",
				"concept_organizationheadquarteredincity",
				"concept_personleadsorganization",
				"concept_worksfor"
			},
			"FB15K237": {
				"sports@sports_team@sport",
				"location@capital_of_administrative_division@capital_of.@location@administrative_division_capital_relationship@administrative_division",
				"people@person@place_of_birth",
				"people@person@nationality",
				"film@director@film",
				"film@film@written_by",
				"film@film@language",
				"film@film@music",
				"film@film@country",
				"organization@organization_founder@organizations_founded",
				"people@ethnicity@languages_spoken",
				"tv@tv_program@languages",
				"time@event@locations",
				"tv@tv_program@country_of_origin",
				"music@artist@origin",
				"tv@tv_program@genre",
				"organization@organization@headquarters.@location@mailing_address@citytown",
				"base@schemastaging@organization_extra@phone_number.@base@schemastaging@phone_sandbox@service_location",
				"organization@organization_member@member_of.@organization@organization_membership@organization",
				"people@profession@specialization_of"
			},
			"WN18RR": {
				"_hypernym", 
				"_derivationally_related_form", 
				"_instance_hypernym", 
				"_also_see", 
				"_member_meronym", 
				"_synset_domain_topic_of", 
				"_has_part", 
				"_member_of_domain_usage", 
				"_member_of_domain_region", 
				"_verb_group"
			},
			"YAGO310": {
				"isAffiliatedTo",
				"playsFor",
				"isLocatedIn",
				"hasGender",
				"wasBornIn",
				"actedIn",
				"isConnectedTo",
				"hasWonPrize",
				"influences",
				"diedIn",
				"hasMusicalRole",
				"graduatedFrom",
				"created",
				"wroteMusicFor",
				"directed",
				"participatedIn",
				"hasChild",
				"happenedIn",
				"isMarriedTo",
				"isCitizenOf",
				"worksAt",
				"edited",
				"livesIn"
			}
		}

		if dataset not in self._all_tasks:
			print("Invalid dataset!", dataset, "is not in", self._all_tasks.keys())
			exit()

		if task not in self._all_tasks[dataset]:
			print("Invalid task!", task, "is not in", self._all_tasks[dataset])
			exit()

		self.dataset = dataset
		self.task = task
		
		if dataset == "NELL995":
			graph_file = "./Datasets/" + self.dataset + "/kb_env_rl.txt"
		elif dataset == "FB15K237":
			graph_file = "./Datasets/" + self.dataset + "/kb_env.txt"
		elif dataset == "WN18RR":
			graph_file = "./Datasets/" + self.dataset + "/graph.txt"
		elif dataset == "YAGO310":
			graph_file = "./Datasets/" + self.dataset + "/graph.txt"

		train_file = "./Datasets/" + self.dataset + "/tasks/" + self.task + "/train.pairs"
		test_file = "./Datasets/" + self.dataset + "/tasks/" + self.task + "/sort_test.pairs"

		self.raw_graph = list(map(lambda line: line.strip().split('\t'), open(graph_file, "r").readlines()))

		self.raw_entities = sorted(list(set(list(map(lambda x: x[0], self.raw_graph)) + list(map(lambda x: x[1], self.raw_graph)))))
		self.raw_relations = sorted(list(set(list(map(lambda x: x[2], self.raw_graph)))))
		self.id2entity = dict(enumerate(self.raw_entities))
		self.id2relation = dict(enumerate(self.raw_relations))
		self.entity2id = {self.id2entity[id_]:id_ for id_ in self.id2entity}
		self.relation2id = {self.id2relation[id_]:id_ for id_ in self.id2relation}
		self.graph = numpy.array(list(map(lambda x: [self.entity2id[x[0]], self.relation2id[x[2]], self.entity2id[x[1]]], self.raw_graph)))

		self.set_graph = set(map(tuple, self.graph.tolist()))

		if dataset in {"NELL995", "FB15K237"}:
			self.raw_train = list(map(lambda x: [x[0][6:], x[1][6:-3], x[1][-1]], list(map(lambda line: line.strip().split(','), open(train_file, "r").readlines()))))
		elif dataset in {"WN18RR", "YAGO310"}:
			self.raw_train = list(map(lambda x: [x[0], x[1], x[2]], list(map(lambda line: line.strip().split(','), open(train_file, "r").readlines()))))

		self.array_train = []
		self.dict_train = {}

		for query_entity, target_entity, classification in self.raw_train:

			if dataset == "FB15K237":
				query_entity = list(query_entity)
				query_entity[1] = "/"
				query_entity = "".join(["/"] + query_entity)
				target_entity = list(target_entity)
				target_entity[1] = "/"
				target_entity = "".join(["/"] + target_entity)

			try:
				self.array_train.append([self.entity2id[query_entity], self.entity2id[target_entity], (0 if classification == '-' else 1)])

				if self.entity2id[query_entity] not in self.dict_train:
					self.dict_train[self.entity2id[query_entity]] = {'positive': set(), 'negative': set()}
				
				if classification == '+':
					self.dict_train[self.entity2id[query_entity]]['positive'].add(self.entity2id[target_entity])
				elif classification == '-':
					self.dict_train[self.entity2id[query_entity]]['negative'].add(self.entity2id[target_entity])
			except:
				pass

		self.array_train = numpy.array(self.array_train)

		if dataset in {"NELL995", "FB15K237"}:
			self.raw_test = list(map(lambda x: [x[0][6:], x[1][6:-3], x[1][-1]], list(map(lambda line: line.strip().split(','), open(test_file, "r").readlines()))))
		elif dataset in {"WN18RR", "YAGO310"}:
			self.raw_test = list(map(lambda x: [x[0], x[1], x[2]], list(map(lambda line: line.strip().split(','), open(test_file, "r").readlines()))))

		self.array_test = []
		self.dict_test = {}

		for query_entity, target_entity, classification in self.raw_test:
			
			if dataset == "FB15K237":
				query_entity = list(query_entity)
				query_entity[1] = "/"
				query_entity = "".join(["/"] + query_entity)
				target_entity = list(target_entity)
				target_entity[1] = "/"
				target_entity = "".join(["/"] + target_entity)

			try:
				self.array_test.append([self.entity2id[query_entity], self.entity2id[target_entity], (0 if classification == '-' else 1)])

				if self.entity2id[query_entity] not in self.dict_test:
					self.dict_test[self.entity2id[query_entity]] = {'positive': set(), 'negative': set()}
				if classification == '+':
					self.dict_test[self.entity2id[query_entity]]['positive'].add(self.entity2id[target_entity])
				elif classification == '-':
					self.dict_test[self.entity2id[query_entity]]['negative'].add(self.entity2id[target_entity])
			except:
				pass

		useless_keys = []
		for key in self.dict_test:
			if (len(self.dict_test[key]['negative']) == 0):
				useless_keys.append(key)

		for key in useless_keys:
			del self.dict_test[key]
					
		self.array_test = numpy.array(self.array_test)

		self.entity_count = len(self.entity2id)
		self.relation_count = len(self.relation2id)
		self.fact_count = len(self.graph)
