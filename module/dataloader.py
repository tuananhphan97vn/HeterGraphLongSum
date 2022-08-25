

import re
import os
from nltk.corpus import stopwords

import glob
import copy
import random
import time
import json
import pickle
import nltk
import collections
from collections import Counter
from itertools import combinations
import numpy as np
from random import shuffle

import torch
import torch.utils.data
import torch.nn.functional as F

from tools.logger import *

import dgl
from dgl.data.utils import save_graphs, load_graphs

FILTERWORD = stopwords.words('english')
punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%', '\'\'', '\'', '`', '``',
				'-', '--', '|', '\/']
FILTERWORD.extend(punctuations)


######################################### Example #########################################

class Example(object):
	"""Class representing a train/val/test example for single-document extractive summarization."""

	def __init__(self, article_sents, abstract_sents, vocab, sent_max_len, label):
		""" Initializes the Example, performing tokenization and truncation to produce the encoder, decoder and target sequences, which are stored in self.

		:param article_sents: list(strings) for single document or list(list(string)) for multi-document; one per article sentence. each token is separated by a single space.
		:param abstract_sents: list(strings); one per abstract sentence. In each sentence, each token is separated by a single space.
		:param vocab: Vocabulary object
		:param sent_max_len: int, max length of each sentence
		:param label: list, the No of selected sentence, e.g. [1,3,5]
		"""

		self.sent_max_len = sent_max_len
		self.enc_sent_len = []
		self.enc_sent_input = []
		self.enc_sent_input_pad = []

		# Store the original strings
		self.original_article_sents = article_sents
		self.original_abstract = "\n".join(abstract_sents)

		# Process the article
		if isinstance(article_sents, list) and isinstance(article_sents[0], list):  # multi document
			self.original_article_sents = []
			for doc in article_sents:
				self.original_article_sents.extend(doc)
		for sent in self.original_article_sents:
			article_words = sent.split()
			self.enc_sent_len.append(len(article_words))  # store the length before padding
			self.enc_sent_input.append([vocab.word2id(w.lower()) for w in article_words])  # list of word ids; OOVs are represented by the id for UNK token
		self._pad_encoder_input(vocab.word2id('[PAD]'))

		# Store the label
		self.label = label
		label_shape = (len(self.original_article_sents), len(label))  # [N, len(label)]
		# label_shape = (len(self.original_article_sents), len(self.original_article_sents))
		self.label_matrix = np.zeros(label_shape, dtype=int)
		if label != []:
			self.label_matrix[np.array(label), np.arange(len(label))] = 1  # label_matrix[i][j]=1 indicate the i-th sent will be selected in j-th step

	def _pad_encoder_input(self, pad_id):
		"""
		:param pad_id: int; token pad id
		:return: 
		"""
		max_len = self.sent_max_len
		for i in range(len(self.enc_sent_input)):
			article_words = self.enc_sent_input[i].copy()
			if len(article_words) > max_len:
				article_words = article_words[:max_len]
			if len(article_words) < max_len:
				article_words.extend([pad_id] * (max_len - len(article_words)))
			self.enc_sent_input_pad.append(article_words)


class Example2(Example):
	"""Class representing a train/val/test example for multi-document extractive summarization."""

	def __init__(self, article_sents, abstract_sents, vocab, sent_max_len, label):
		""" Initializes the Example, performing tokenization and truncation to produce the encoder, decoder and target sequences, which are stored in self.

		:param article_sents: list(list(string)) for multi-document; one per article sentence. each token is separated by a single space.
		:param abstract_sents: list(strings); one per abstract sentence. In each sentence, each token is separated by a single space.
		:param vocab: Vocabulary object
		:param sent_max_len: int, max length of each sentence
		:param label: list, the No of selected sentence, e.g. [1,3,5]
		"""

		super().__init__(article_sents, abstract_sents, vocab, sent_max_len, label)
		cur = 0
		self.original_articles = []
		self.article_len = []
		self.enc_doc_input = []
		for doc in article_sents:
			if len(doc) == 0:
				continue
			docLen = len(doc)
			self.original_articles.append(" ".join(doc))
			self.article_len.append(docLen)
			self.enc_doc_input.append(catDoc(self.enc_sent_input[cur:cur + docLen]))
			cur += docLen


######################################### ExampleSet #########################################

class ExampleSet(torch.utils.data.Dataset):
	""" Constructor: Dataset of example(object) for single document summarization"""

	def __init__(self, data_path, vocab, doc_max_timesteps, sent_max_len, filter_word_path, w2s_path, passage_length):
		""" Initializes the ExampleSet with the path of data
		
		:param data_path: string; the path of data
		:param vocab: object;
		:param doc_max_timesteps: int; the maximum sentence number of a document, each example should pad sentences to this length
		:param sent_max_len: int; the maximum token number of a sentence, each sentence should pad tokens to this length
		:param filter_word_path: str; file path, the file must contain one word for each line and the tfidf value must go from low to high (the format can refer to script/lowTFIDFWords.py) 
		:param w2s_path: str; file path, each line in the file contain a json format data (which can refer to the format can refer to script/calw2sTFIDF.py)
		"""
		self.segment_length = passage_length
		self.vocab = vocab
		self.sent_max_len = sent_max_len
		self.doc_max_timesteps = doc_max_timesteps

		logger.info("[INFO] Start reading %s", self.__class__.__name__)
		start = time.time()
		self.example_list = readJson(data_path)
		logger.info("[INFO] Finish reading %s. Total time is %f, Total size is %d", self.__class__.__name__,
					time.time() - start, len(self.example_list))
		self.size = len(self.example_list)

		logger.info("[INFO] Loading filter word File %s", filter_word_path)
		tfidf_w = readText(filter_word_path)
		self.filterwords = FILTERWORD
		self.filterids = [vocab.word2id(w.lower()) for w in FILTERWORD]
		self.filterids.append(vocab.word2id("[PAD]"))   # keep "[UNK]" but remove "[PAD]"
		lowtfidf_num = 0
		pattern = r"^[0-9]+$"
		for w in tfidf_w:
			if vocab.word2id(w) != vocab.word2id('[UNK]'):
				self.filterwords.append(w)
				self.filterids.append(vocab.word2id(w))
				# if re.search(pattern, w) == None:  # if w is a number, it will not increase the lowtfidf_num
					# lowtfidf_num += 1
				lowtfidf_num += 1
			if lowtfidf_num > 5000:
				break

		logger.info("[INFO] Loading word2sent TFIDF file from %s!" % w2s_path)
		self.w2s_tfidf = readJson(w2s_path)

	def get_example(self, index):
		e = self.example_list[index]
		e["summary"] = e.setdefault("summary", [])
		example = Example(e["text"], e["summary"], self.vocab, self.sent_max_len, e["label"])
		return example

	def pad_label_m(self, label_matrix):
		label_m = label_matrix[:self.doc_max_timesteps, :self.doc_max_timesteps]
		N, m = label_m.shape
		if m < self.doc_max_timesteps:
			pad_m = np.zeros((N, self.doc_max_timesteps - m))
			return np.hstack([label_m, pad_m])
		return label_m

	def AddWordNode(self, G, inputid):
		wid2nid = {}
		nid2wid = {}
		nid = 0
		for sentid in inputid:
			for wid in sentid:
				if wid not in self.filterids and wid not in wid2nid.keys():
					wid2nid[wid] = nid
					nid2wid[nid] = wid
					nid += 1

		w_nodes = len(nid2wid)

		G.add_nodes(w_nodes)
		G.set_n_initializer(dgl.init.zero_initializer)
		G.ndata["unit"] = torch.zeros(w_nodes)
		G.ndata["id"] = torch.LongTensor(list(nid2wid.values()))
		G.ndata["dtype"] = torch.zeros(w_nodes)

		return wid2nid, nid2wid

	def CreateGraphWS(self, input_pad, label, w2s_w):
		""" Create a graph for each document
		
		:param input_pad: list(list); [sentnum, wordnum]
		:param label: list(list); [sentnum, sentnum]
		:param w2s_w: dict(dict) {str: {str: float}}; for each sentence and each word, the tfidf between them
		:return: G: dgl.DGLGraph
			node:
				word: unit=0, dtype=0, id=(int)wordid in vocab
				sentence: unit=1, dtype=1, words=tensor, position=int, label=tensor
			edge:
				word2sent, sent2word:  tffrac=int, dtype=0
		"""
		G = dgl.DGLGraph()
		wid2nid, nid2wid = self.AddWordNode(G, input_pad)
		w_nodes = len(nid2wid)

		N = len(input_pad)
		G.add_nodes(N)
		G.ndata["unit"][w_nodes:] = torch.ones(N)
		G.ndata["dtype"][w_nodes:] = torch.ones(N)
		sentid2nid = [i + w_nodes for i in range(N)]

		G.set_e_initializer(dgl.init.zero_initializer)
		for i in range(N):
			c = Counter(input_pad[i])
			sent_nid = sentid2nid[i]
			sent_tfw = w2s_w[str(i)]
			for wid in c.keys():
				if wid in wid2nid.keys() and self.vocab.id2word(wid) in sent_tfw.keys():
					tfidf = sent_tfw[self.vocab.id2word(wid)]
					tfidf_box = np.round(tfidf * 9)  # box = 10
					G.add_edges(wid2nid[wid], sent_nid,
								data={"tffrac": torch.LongTensor([tfidf_box]), "dtype": torch.Tensor([0])})
					G.add_edges(sent_nid, wid2nid[wid],
								data={"tffrac": torch.LongTensor([tfidf_box]), "dtype": torch.Tensor([0])})
			
			# The two lines can be commented out if you use the code for your own training, since HSG does not use sent2sent edges. 
			# However, if you want to use the released checkpoint directly, please leave them here.
			# Otherwise it may cause some parameter corresponding errors due to the version differences.
			G.add_edges(sent_nid, sentid2nid, data={"dtype": torch.ones(N)})
			G.add_edges(sentid2nid, sent_nid, data={"dtype": torch.ones(N)})
		G.nodes[sentid2nid].data["words"] = torch.LongTensor(input_pad)  # [N, seq_len]
		G.nodes[sentid2nid].data["position"] = torch.arange(1, N + 1).view(-1, 1).long()  # [N, 1]
		G.nodes[sentid2nid].data["label"] = torch.LongTensor(label)  # [N, doc_max]

		return G
	
	def CreateGraphWP(self, input_pad, label, w2s_w):
		""" Create a graph for each document
		:param input_pad: list(list); [sentnum, wordnum]
		:param label: list(list); [sentnum, sentnum]
		:param w2s_w: dict(dict) {str: {str: float}}; for each sentence and each word, the tfidf between them
		:return: G: dgl.DGLGraph
			node:
				word: unit=0, dtype=0, id=(int)wordid in vocab
				passage: unit=1, dtype=1, words=tensor, position=int, label=tensor
			edge:
				word2pass, pass2word:  tffrac=int, dtype=0
		"""
		G = dgl.DGLGraph()

		#add word node to empty graph
		wid2nid, nid2wid = self.AddWordNode(G, input_pad)
		w_nodes = len(nid2wid)  #number nodes

		#split sentences to passage
		list_passage = []
		chunk_size = self.segment_length
		for i in range(0, len(input_pad), chunk_size):
			passage = [] 
			for sentence in input_pad[i:i+chunk_size]:
				passage.extend(sentence)
			#passage is list that consist multi sentences
			list_passage.append(passage)

		num_passage = len(list_passage)
		G.add_nodes(num_passage)

		G.ndata["unit"][w_nodes:] = torch.ones(num_passage)
		G.ndata["dtype"][w_nodes:] = torch.ones(num_passage)

		#define sentence node id use word node id
		passageid2nid = [i + w_nodes for i in range(num_passage)]

		G.set_e_initializer(dgl.init.zero_initializer)

		for i in range(num_passage):
			c = Counter(list_passage[i])
			passage_id = passageid2nid[i] #convert passage id to node id of pass in graph_wp
			for wid in c.keys():
				if wid in wid2nid.keys():
					#only consider edge from word to passage
					G.add_edges(wid2nid[wid], passage_id, data={"dtype": torch.Tensor([0])})
		return G

	def CreateGraphSP(self, input_pad, label, w2s_w):

		G = dgl.DGLGraph()

		#split sentences to passage
		list_passage = []
		count_sent_passage = []  #count number sentence per passage, that attribute useful for compute passage feature
		chunk_size = self.segment_length
		for i in range(0, len(input_pad), chunk_size):
			passage = [] 
			for j in range(i , min(i+chunk_size , len(input_pad)) , 1):
				passage.append(j)
			count_sent_passage.append(len(passage))
			#passage is list that consist multi sentences
			list_passage.append(passage)
       
		num_passage = len(list_passage)

		#add passage node 
		G.add_nodes(num_passage)
		G.ndata["unit"]= torch.zeros(num_passage)
		G.ndata["dtype"] = torch.zeros(num_passage)
		G.ndata['num_sent'] = torch.LongTensor(count_sent_passage)

		passageid2nid = [i for i in range(num_passage)]

		#add sentence node 
		num_sentence = len(input_pad)
		G.add_nodes(num_sentence)
		G.ndata["unit"][num_passage:]= torch.ones(num_sentence)
		G.ndata["dtype"][num_passage:] = torch.ones(num_sentence)
		sentid2nid = [i + num_passage for i in range(num_sentence)]
  
		G.set_e_initializer(dgl.init.zero_initializer)

		for i in range(num_passage):
			list_sentence = list_passage[i]
			node_passage_id = passageid2nid[i] #convert passage id to node id of pass in graph_wp
			node_sent_pass_id = [ sentid2nid[t] for t in list_sentence]  #get list of sent node id 
			for sent_node in node_sent_pass_id:
				#only consider node from passage --> sentence
				G.add_edges(node_passage_id, sent_node, data={"dtype": torch.Tensor([0])})
		return G

	def __getitem__(self, index):
		"""
		:param index: int; the index of the example
		:return 
			G: graph for the example
			index: int; the index of the example in the dataset
		"""
		item = self.get_example(index)
		input_pad = item.enc_sent_input_pad[:self.doc_max_timesteps]
		label = self.pad_label_m(item.label_matrix)
		w2s_w = self.w2s_tfidf[index]
		G_ws = self.CreateGraphWS(input_pad, label, w2s_w)
		G_wp = self.CreateGraphWP(input_pad , label , w2s_w)
		G_sp = self.CreateGraphSP(input_pad , label , w2s_w)
		return G_ws , G_wp , G_sp , index

	def __len__(self):
		return self.size

class LoadHiExampleSet(torch.utils.data.Dataset):
	def __init__(self, data_root):
		super().__init__()
		self.data_root = data_root
		self.gfiles = [f for f in os.listdir(self.data_root) if f.endswith("graph.bin")]
		logger.info("[INFO] Start loading %s", self.data_root)

	def __getitem__(self, index):
		graph_file = os.path.join(self.data_root, "%d.graph.bin" % index)
		g, label_dict = load_graphs(graph_file)
		# print(graph_file)
		return g[0], index

	def __len__(self):
		return len(self.gfiles)

######################################### Tools #########################################

import dgl


def catDoc(textlist):
	res = []
	for tlist in textlist:
		res.extend(tlist)
	return res


def readJson(fname):
	data = []
	with open(fname, encoding="utf-8") as f:
		for line in f:
			data.append(json.loads(line))
	return data


def readText(fname):
	data = []
	with open(fname, encoding="utf-8") as f:
		for line in f:
			data.append(line.strip())
	return data


def graph_collate_fn(samples):
	'''
	:param batch: (G, input_pad)
	:return: 
	'''
	graph_ws , graph_wp , graph_sp, index = map(list, zip(*samples))
	graph_len = [len(g.filter_nodes(lambda nodes: nodes.data["dtype"] == 1)) for g in graph_ws]  # sent node of graph
	sorted_len, sorted_index = torch.sort(torch.LongTensor(graph_len), dim=0, descending=True)
	batched_graph_ws = dgl.batch([graph_ws[idx] for idx in sorted_index])
	batched_graph_wp = dgl.batch([graph_wp[idx] for idx in sorted_index])
	batched_graph_sp = dgl.batch([graph_sp[idx] for idx in sorted_index])

	return batched_graph_ws , batched_graph_wp, batched_graph_sp, [index[idx] for idx in sorted_index]
