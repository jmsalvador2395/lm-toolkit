import torch
from torch import nn
import torch.nn.functional as f
from torch.nn import init
import numpy as np
from tqdm import tqdm

#from utils import data, misc
import copy

class GloVe(nn.Module):
	
	def __init__(self, model_path, device, params):
		super(GloVe, self).__init__()
		self.vocab_w2i_dict = {'<UNK>' : 0}
		self.vocab_i2w_dict = {0 : '<UNK>'}
		embeddings = []
		self.dev=device
		if not misc.valid_path(model_path): 
			raise FileNotFoundError("could not find any embeddings to load")

		misc.green('reading in vocab and embeddings')
		with open(model_path, 'r') as f:
			for i, line in tqdm(enumerate(f)):
				split = line.split()
				self.vocab_i2w_dict[i+1] = split[0]
				self.vocab_w2i_dict[split[0]] = i+1
				embeddings.append(np.array(
					[float(x) for x in split[1:]]
				))
		misc.green('reading in word probabilities')
		if data.counts_exists(params['corpus_dir'], min_freq=params['min_freq']):
			freqs = data.load_counts(params['corpus_dir'], min_freq=params['min_freq'])
			total_tokens = np.sum(np.array(list(freqs.values())))
			p = {key : val/total_tokens for key, val in freqs.items()}

		# set word probabilities
		vocab = copy.deepcopy(self.vocab_w2i_dict)
		self.w2p = p
		self.i2p = {vocab.get(w, 0) : prob for w, prob in p.items()}

		# set weight
		self.alpha = params['alpha_weight']



		embeddings = torch.tensor(np.array(embeddings))
		embeddings = torch.vstack((
			torch.zeros(embeddings.shape[-1]),
			embeddings
		))
		self.embeddings = torch.nn.Embedding(
			embeddings.shape[0],
			embeddings.shape[1],
			_weight=embeddings,
		)

		
	def forward(self, indices):
		return self.embeddings(indices)

	def vocab_w2i(self, words):
		indices = []
		for w in words:
			indices.append(self.vocab_w2i_dict.get(w, 0))
		return torch.tensor(indices, dtype=torch.int64, device=self.dev)

	def word_sim(self, words1, words2, dim=1):
		indices1 = self.vocab_w2i(words1)
		indices2 = self.vocab_w2i(words2)

		emb1 = self.embeddings(indices1)
		emb2 = self.embeddings(indices2) 

		return f.cosine_similarity(emb1, emb2)

	def word_weights(self, indices):
		word_probs = torch.zeros(len(indices), device=self.dev)
		for i, idx in enumerate(indices):
			word_probs[i] = self.i2p.get(int(idx), 0)
		return self.alpha/(self.alpha+word_probs)

	def sentence_sim(self, sentence1, sentence2, batch=False):
		
		if batch:
			batch_size = len(sentence1)
			# compute indices
			indices1 = []
			indices2 = []
			for s1, s2 in zip(sentence1, sentence2):
				indices1.append(self.vocab_w2i(s1))
				indices2.append(self.vocab_w2i(s2))

			# get all averaged embeddings
			emb1 = torch.vstack([
				torch.mean(
					self.word_weights(s)[:, None]*self.embeddings(s), 
					dim=0
				) 
				for s in indices1
			])
			emb2 = torch.vstack([
				torch.mean(
					self.word_weights(s)[:, None]*self.embeddings(s), 
					dim=0
				) 
				for s in indices2
			])

			emb_all = torch.vstack((emb1, emb2))

			U, S, Vh = torch.svd(emb_all)

			u = U[0]

			bias = u@u[:, None]@u[None]

			emb_all -= bias

			emb1 = emb_all[:batch_size]
			emb2 = emb_all[batch_size:]

			return f.cosine_similarity(emb1, emb2)
		# compute sentence similarity for just 1 pair of sentences
		else:
			indices1 = torch.tensor(
				self.vocab_w2i(sentence1),
				dtype=torch.int64,
				device=self.dev
			)
			indices2 = torch.tensor(
				self.vocab_w2i(sentence2),
				dtype=torch.int64,
				device=self.dev
			)

			# get embeddings
			emb1 = self.embeddings(indices1)
			emb2 = self.embeddings(indices2)

			# compute averaged embedding
			emb1 = torch.mean(emb1, dim=0)
			emb2 = torch.mean(emb2, dim=0)

			return f.cosine_similarity(emb1, emb2, dim=0)

