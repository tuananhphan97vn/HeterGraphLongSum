
import numpy as np
import torch.nn.functional as F

import torch
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn

import dgl

# from module.GAT import GAT, GAT_ffn
from module.Encoder import sentEncoder
from module.GAT import WSWGAT, WPWGAT , SPSGAT
from module.PositionEmbedding import get_sinusoid_encoding_table


class HSumGraph(nn.Module):
	""" without sent2sent and add residual connection """

	def __init__(self, hps, embed):
		"""

		:param hps: 
		:param embed: word embedding
		"""
		super().__init__()

		self._hps = hps
		self._n_iter = hps.n_iter
		self._embed = embed
		self.embed_size = hps.word_emb_dim

		# sent node feature
		self._init_sn_param()
		self._TFembed = nn.Embedding(10, hps.feat_embed_size)   # box=10
		self.n_feature_proj = nn.Linear(
			hps.n_feature_size * 2, hps.hidden_size, bias=False)

		# word -> sent
		embed_size = hps.word_emb_dim
		self.word2sent = WSWGAT(in_dim=embed_size,
								out_dim=hps.hidden_size,
								num_heads=hps.n_head,
								attn_drop_out=hps.atten_dropout_prob,
								ffn_inner_hidden_size=hps.ffn_inner_hidden_size,
								ffn_drop_out=hps.ffn_dropout_prob,
								feat_embed_size=hps.feat_embed_size,
								layerType="W2S"
								)
		# sent -> word
		self.sent2word = WSWGAT(in_dim=hps.hidden_size,
								out_dim=embed_size,
								num_heads=4,
								attn_drop_out=hps.atten_dropout_prob,
								ffn_inner_hidden_size=hps.ffn_inner_hidden_size,
								ffn_drop_out=hps.ffn_dropout_prob,
								feat_embed_size=hps.feat_embed_size,
								layerType="S2W"
								)

		# word -> passage
		self.word2passage = WPWGAT(in_dim=embed_size,
								out_dim=hps.hidden_size,
								num_heads=1,
								attn_drop_out=hps.atten_dropout_prob,
								ffn_inner_hidden_size=hps.ffn_inner_hidden_size,
								ffn_drop_out=hps.ffn_dropout_prob,
								feat_embed_size=hps.feat_embed_size,
								layerType="W2P"
								)

		# passage --> sent 
		self.passage2sent = SPSGAT(in_dim=hps.hidden_size,
								out_dim=hps.hidden_size,
								num_heads=1,
								attn_drop_out=hps.atten_dropout_prob,
								ffn_inner_hidden_size=hps.ffn_inner_hidden_size,
								ffn_drop_out=hps.ffn_dropout_prob,
								feat_embed_size=hps.feat_embed_size,
								layerType="P2S"
								)

		# node classification
		self.n_feature = hps.hidden_size
		self.wh = nn.Linear(self.n_feature, 2)
		self.use_doc = hps.use_doc
		self.l1 = nn.Linear(2 * hps.hidden_size , hps.hidden_size)
		self.doc_layer = nn.Linear(hps.hidden_size , hps.hidden_size)
		self.doc_att_linear = nn.Linear(hps.hidden_size , 1)


	def forward(self, graph_ws, graph_wp, graph_sp):
		"""
		#each document is transformed to one DGLGraph
		:param graph: [batch_size] * DGLGraph  
			node:
				word: unit=0, dtype=0, id=(int)wordid in vocab
				sentence: unit=1, dtype=1, words=tensor, position=int, label=tensor
			edge:
				word2sent, sent2word:  tffrac=int, type=0

		#result feed through binary classify layer 
		:return: result: [sentnum, 2]
		"""
		# word node init and get edge embedding
		word_feature = self.set_wnfeature(graph_ws)    # [wnode, embed_size]
		# sentence node init
		sent_feature = self.n_feature_proj(self.set_snfeature(graph_ws))    # [snode, n_feature_size]

		# the start state
		word_state = word_feature
		sent_state = self.word2sent(graph_ws, word_feature, sent_feature)

		pnode_id = graph_sp.filter_nodes(lambda nodes: nodes.data["unit"] == 0)
		p_sent = graph_sp.nodes[pnode_id].data["num_sent"].tolist()
		passages = torch.split(sent_state , p_sent) #list of tensor 
		sent_passage_tensor = self.create_sent_passage_tensor(passages) #shape (num passage , max sentence = seq length , sent dim )
		passage_state = self.compute_passage_feature(sent_passage_tensor) #shape (num passage , sent dim = passage dim )

		for i in range(self._n_iter):
			# update word 
			word_state = self.sent2word(graph_ws, word_state, sent_state)
			# update sentence
			sent_state_from_word= self.word2sent(graph_ws, word_state, sent_state)
			sent_state_from_passage = self.passage2sent(graph_sp , passage_state , sent_state)
			sent_state = sent_state_from_word + sent_state_from_passage
			#update passage
			passage_state = self.word2passage(graph_wp , word_state , passage_state)

		if self.use_doc == True:	
			list_n_sent_node , list_n_passage_node  = []  , [] 
			G_ws_unbatch , G_sp_unbatch = dgl.unbatch(graph_ws) , dgl.unbatch(graph_sp)

			for g_ws in G_ws_unbatch:
				edges = g_ws.edges()
				sentence_node = g_ws.filter_nodes(lambda nodes: nodes.data["unit"]==1)
				list_n_sent_node.append(len(sentence_node) )
			list_sent_represent_matrix= torch.split(sent_state , list_n_sent_node , dim = 0)

			for g_sp in G_sp_unbatch:
				edges = g_sp.edges()
				passage_node = g_sp.filter_nodes(lambda nodes: nodes.data["unit"]==0)
				list_n_passage_node.append(len(passage_node) )
			list_passage_represent_matrix = torch.split(passage_state , list_n_passage_node , dim = 0 ) #list of elements, each element match to list passage representation of a doc 

			sd_state = []
			for i in range(len(list_sent_represent_matrix)):
				sents = list_sent_represent_matrix[i] # shape (num sent , hidden size )
				passages = list_passage_represent_matrix[i]  # (num topic per doc , hidden size )
				doc = self.compute_doc(passages) # (hidden size)
				doc_repeat = doc.repeat(sents.shape[0] , 1)  #shape (num sent , hidden size )
				sents_doc = torch.cat( (sents, doc_repeat) , dim = 1) #shape (num sent , 2 * hidden size )
				sd_state.append(sents_doc)

			sd_state = torch.cat(sd_state , dim = 0 ) # (num sent , 2 * hidden size )
			result = self.wh(F.relu(self.l1(sd_state)))#shape (snode , 2)
		
		elif self.use_doc == False:
			result = self.wh(sent_state)  #shape (snode , sent dim )
	
		return result 

	def compute_doc(self, passages):
		z_pass = self.doc_layer(passages) #shape (num topic , hidden size )
		#compute attention 
		w = F.leaky_relu(self.doc_att_linear(z_pass)) #shape (num topic , 1)
		att = F.softmax( w , dim = 0) #shape (num topic , 1) satisfy condition: sum = 1
		s = att * z_pass #shape (num topic , 1) * (num topic , hidden size ) --> (num topic , hidden size) 
		out = torch.sum(s , dim  = 0 ) #shape (hidden size )
		return out       

	def compute_passage_feature(self, sent_passage_tensor):
		#sent passage tensor : (num passage , max sentence , sent dim )
		output , (_ , _) = self.lstm_passage(sent_passage_tensor) #output shape (num passage , max sentence , 2 * sent hidden dim )
		passage_feature = output[: , - 1 , :] #shape (num passage, 2* sent dim)
		return self.lstm_pass_proj(passage_feature) #shape (num passage , sent dim)

	def create_sent_passage_tensor(self , list_passage):
		#list passage in shape list of tensor, each tensor has shape (num sent , sent dim )
		#we need padding if num_sent < max sentenc to corresponse with 
		num_sent = [passage.shape[0] for passage in list_passage]
		max_sent = max(num_sent) #find max sent 
		result = [] 
		for i in range(len(list_passage)):
			passage = list_passage[i]
			if passage.shape[0] == max_sent:
				result.append(passage)
			else:
				#num sent < max_sent, so frist need to pad 
				num_sent = passage.shape[0]
				pad_sent = max_sent - num_sent  
				result.append(torch.cat( (passage , torch.zeros(pad_sent , passage.shape[1]).to(torch.device('cuda:0'))) , dim = 0 )) 
		result = torch.stack(result , dim = 0 ) #shape (num passage = batch size , max sent per passage = sequence length , sent dim  = hidden sate )
		return result

	def _init_sn_param(self):
		self.sent_pos_embed = nn.Embedding.from_pretrained(
			get_sinusoid_encoding_table(
				self._hps.doc_max_timesteps + 1, self.embed_size, padding_idx=0),
			freeze=True)
		self.cnn_proj = nn.Linear(self.embed_size, self._hps.n_feature_size)
		self.lstm_hidden_state = self._hps.lstm_hidden_state
		self.lstm = nn.LSTM(self.embed_size, self.lstm_hidden_state, num_layers=self._hps.lstm_layers, dropout=0.1,
							batch_first=True, bidirectional=self._hps.bidirectional)
		if self._hps.bidirectional:
			self.lstm_proj = nn.Linear(
				self.lstm_hidden_state * 2, self._hps.n_feature_size)
		else:
			self.lstm_proj = nn.Linear(
				self.lstm_hidden_state, self._hps.n_feature_size)

		self.ngram_enc = sentEncoder(self._hps, self._embed)

		#lstm for passage 
		self.lstm_passage = nn.LSTM(self._hps.hidden_size, self._hps.hidden_size, num_layers=self._hps.lstm_layers, dropout=0.1,batch_first=True, bidirectional=self._hps.bidirectional)

		if self._hps.bidirectional:
			self.lstm_pass_proj = nn.Linear(
				self._hps.hidden_size * 2, self._hps.hidden_size)
		else:
			self.lstm_proj = nn.Linear(
				self._hps.hidden_size , self._hps.hidden_size)

	def _sent_cnn_feature(self, graph, snode_id):
		ngram_feature = self.ngram_enc.forward(
			graph.nodes[snode_id].data["words"])  # [snode, embed_size]
		graph.nodes[snode_id].data["sent_embedding"] = ngram_feature
		# [n_nodes]
		snode_pos = graph.nodes[snode_id].data["position"].view(-1)
		position_embedding = self.sent_pos_embed(snode_pos)
		cnn_feature = self.cnn_proj(ngram_feature + position_embedding)
		return cnn_feature

	def _sent_lstm_feature(self, features, glen):
		pad_seq = rnn.pad_sequence(features, batch_first=True)
		lstm_input = rnn.pack_padded_sequence(pad_seq, glen, batch_first=True)
		lstm_output, _ = self.lstm(lstm_input)
		unpacked, unpacked_len = rnn.pad_packed_sequence(
			lstm_output, batch_first=True)
		lstm_embedding = [unpacked[i][:unpacked_len[i]]
						  for i in range(len(unpacked))]
		# [n_nodes, n_feature_size]
		lstm_feature = self.lstm_proj(torch.cat(lstm_embedding, dim=0))
		return lstm_feature

	def set_wnfeature(self, graph):
		# set word node feature
		wnode_id = graph.filter_nodes(lambda nodes: nodes.data["unit"] == 0)
		# for word to supernode(sent&doc)
		wsedge_id = graph.filter_edges(lambda edges: edges.data["dtype"] == 0)
		wid = graph.nodes[wnode_id].data["id"]  # [n_wnodes]
		w_embed = self._embed(wid)  # [n_wnodes, D]
		# assign graph node value
		graph.nodes[wnode_id].data["embed"] = w_embed
		etf = graph.edges[wsedge_id].data["tffrac"]
		graph.edges[wsedge_id].data["tfidfembed"] = self._TFembed(etf)
		return w_embed

	def set_snfeature(self, graph):
		# set sentence node feature
		snode_id = graph.filter_nodes(lambda nodes: nodes.data["dtype"] == 1)

		# set data vs feature name is sent embedding,  value cnn encode
		# shape (n node , embedding - size )
		cnn_feature = self._sent_cnn_feature(graph, snode_id)
		# get node sentence
		features, glen = get_snode_feat(graph, feat="sent_embedding")
		# pass output from cnn layer to bilstm layer
		lstm_feature = self._sent_lstm_feature(features, glen)

		# [n_nodes, n_feature_size * 2]
		node_feature = torch.cat([cnn_feature, lstm_feature], dim=1)

		# each sentence is presented by 256-dimension vector
		return node_feature


class HSumDocGraph(HSumGraph):
	"""
		without sent2sent and add residual connection
		add Document Nodes
	"""

	def __init__(self, hps, embed):
		super().__init__(hps, embed)
		self.dn_feature_proj = nn.Linear(
			hps.hidden_size, hps.hidden_size, bias=False)
		self.wh = nn.Linear(self.n_feature * 2, 2)

	def forward(self, graph):
		"""
		:param graph: [batch_size] * DGLGraph
			node:
				word: unit=0, dtype=0, id=(int)wordid in vocab
				sentence: unit=1, dtype=1, words=tensor, position=int, label=tensor
				document: unit=1, dtype=2
			edge:
				word2sent, sent2word: tffrac=int, type=0
				word2doc, doc2word: tffrac=int, type=0
				sent2doc: type=2  #? edge from sent to doc 
		:return: result: [sentnum, 2]
		"""

		snode_id = graph.filter_nodes(lambda nodes: nodes.data["dtype"] == 1)
		dnode_id = graph.filter_nodes(lambda nodes: nodes.data["dtype"] == 2)
		supernode_id = graph.filter_nodes(
			lambda nodes: nodes.data["unit"] == 1)

		# word node init
		word_feature = self.set_wnfeature(graph)    # [wnode, embed_size]
		sent_feature = self.n_feature_proj(
			self.set_snfeature(graph))    # [snode, n_feature_size]

		# sent and doc node init
		graph.nodes[snode_id].data["init_feature"] = sent_feature
		doc_feature, snid2dnid = self.set_dnfeature(graph)
		doc_feature = self.dn_feature_proj(doc_feature)
		graph.nodes[dnode_id].data["init_feature"] = doc_feature

		# the start state
		word_state = word_feature
		sent_state = graph.nodes[supernode_id].data["init_feature"]
		sent_state = self.word2sent(graph, word_state, sent_state)

		for i in range(self._n_iter):
			# sent -> word
			word_state = self.sent2word(graph, word_state, sent_state)
			# word -> sent
			sent_state = self.word2sent(graph, word_state, sent_state)

		graph.nodes[supernode_id].data["hidden_state"] = sent_state

		# extract sentence nodes
		s_state_list = []
		for snid in snode_id:
			d_state = graph.nodes[snid2dnid[int(snid)]].data["hidden_state"]
			s_state = graph.nodes[snid].data["hidden_state"]
			s_state = torch.cat([s_state, d_state], dim=-1)
			s_state_list.append(s_state)

		s_state = torch.cat(s_state_list, dim=0)
		result = self.wh(s_state)
		return result

	def set_dnfeature(self, graph):
		""" init doc node by mean pooling on the its sent node (connected by the edges with type=1) """
		dnode_id = graph.filter_nodes(lambda nodes: nodes.data["dtype"] == 2)
		node_feature_list = []
		snid2dnid = {}
		for dnode in dnode_id:
			snodes = [nid for nid in graph.predecessors(
				dnode) if graph.nodes[nid].data["dtype"] == 1]
			doc_feature = graph.nodes[snodes].data["init_feature"].mean(dim=0)
			assert not torch.any(torch.isnan(doc_feature)
								 ), "doc_feature_element"
			node_feature_list.append(doc_feature)
			for s in snodes:
				snid2dnid[int(s)] = dnode
		node_feature = torch.stack(node_feature_list)
		return node_feature, snid2dnid


def get_snode_feat(G, feat):
	glist = dgl.unbatch(G)
	feature = []
	glen = []
	for g in glist:
		# for each document <=> each graph
		snode_id = g.filter_nodes(lambda nodes: nodes.data["dtype"] == 1)
		feature.append(g.nodes[snode_id].data[feat])
		glen.append(len(snode_id))
	return feature, glen
