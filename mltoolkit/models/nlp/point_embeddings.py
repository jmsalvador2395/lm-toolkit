import torch
from torch import nn
import torch.nn.functional as f
from torch.nn import init
import numpy as np

from utils import data

class PointEmbeddings(nn.Module):

	def __init__(self, vocab, p, params):
		"""
		sets the vocabulary, parameter dvice, and dimensionality of the model
		"""
		super(PointEmbeddings, self).__init__()


		self.svd = True
		
		# set vocab
		self.vocab = vocab

		# set word probabilities
		self.w2p = p
		self.i2p = {vocab([w])[0] : prob for w, prob in p.items()}

		# set weight
		self.alpha = params['alpha_weight']

		# set model device
		self.dev = params['device']

		self.embedding_dim = params['embedding_dim']
		
		# embeddings layer
		self.embeddings = nn.Embedding(
			len(vocab),
			self.embedding_dim,
		)

		# projection layer
		self.projection = nn.Embedding(
			len(self.vocab),
			self.embedding_dim,
		)

		init.uniform_(self.embeddings.weight)
		init.uniform_(self.projection.weight)

	def word2emb(self, words, matrix='emb'):
		"""
		takes a list of words and returns its corresponding embeddings based on the vocab
		"""
		
		indices = torch.tensor(self.vocab.forward(words), device=self.dev)
		if matrix == 'emb':
			return self.embeddings(indices)
		elif matrix == 'proj':
			return self.projection(indices)
			

	def ind2emb(self, indices, matrix='emb'):
		"""
		takes in a torch tensor of indices and returns the corresponding embeddings
		"""
		indices = torch.tensor(indices, device=self.dev)
		if matrix == 'emb':
			return self.embeddings(indices)
		elif matrix == 'proj':
			return self.projection(indices)

	def word_weights(self, indices):
		word_probs = torch.zeros(len(indices), device=self.dev)
		for i, idx in enumerate(indices):
			word_probs[i] = self.i2p.get(int(idx), 0)
		return self.alpha/(self.alpha+word_probs)

	def word_sim(self, words1, words2, dim=1):
		emb1 = self.ind2emb(self.vocab(words1), matrix='emb')
		emb2 = self.ind2emb(self.vocab(words2), matrix='emb')

		return f.cosine_similarity(emb1, emb2)

	def sentence_sim(self, sentence1, sentence2, batch=False):
		
		if batch:
			batch_size = len(sentence1)
			# compute indices
			indices1 = []
			indices2 = []
			for s1, s2 in zip(sentence1, sentence2):
				indices1.append(torch.tensor(
					self.vocab(s1),
					dtype=torch.int64,
					device=self.dev
				))
				indices2.append(torch.tensor(
					self.vocab(s2),
					dtype=torch.int64,
					device=self.dev
				))

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

			# compute the sentence embeddings using the method of "a simple but tough to beat baseline..."
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
				self.vocab(sentence1),
				dtype=torch.int64,
				device=self.dev
			)
			indices2 = torch.tensor(
				self.vocab(sentence2),
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


class CBOW(PointEmbeddings):
	
	def __init__(self, vocab, p, params):
		"""
		This is an implementation of the CBOW model
		"""
		super(CBOW, self).__init__(vocab, p, params)

	
	def forward(self, center_words, outside_words, outside_mask):

		# get embeddings
		center_vecs = self.projection(center_words)
		outside_vecs = self.embeddings(outside_words)
		outside_vecs.masked_fill_(~outside_mask[..., None], 0)

		# do this for multiple samples
		if len(outside_words.size()) > 1:
			outside_vecs = torch.mean(outside_vecs, dim=1)
		# do this for just one sample
		else:
			outside_vecs = torch.mean(outside_vecs, dim=0)

		if len(center_vecs.shape) > 2:
			outside_vecs = outside_vecs[:, None, :]

		# compute projection
		scores = torch.sum(outside_vecs*center_vecs, dim=-1)

		return scores


class Skipgram(PointEmbeddings):
	
	def __init__(self, vocab, p, embedding_dim):
		super(Skipgram, self).__init__(vocab, p, embedding_dim)

	def forward(self, center_words, outside_words):

		# get embeddings
		center_vecs = self.embeddings(center_words)
		outside_vecs = self.projection(outside_words)

		"""
		# other way 
		center_vecs = self.projection(center_words)
		outside_vecs = self.embeddings(outside_words)
		"""

		if len(center_vecs.shape) < 3:
			center_vecs = center_vecs[:, None, :]
		if len(outside_vecs.shape) < 3:
			outside_vecs = outside_vecs[:, None, :]

		# compute projection
		scores = torch.sum(outside_vecs*center_vecs, dim=-1)
		return scores


class GloVe(PointEmbeddings):
	
	def __init__(self, vocab, p, embedding_dim):
		super(GloVe, self).__init__(vocab, p, embedding_dim)

		self.b = nn.Parameter(torch.ones(len(self.vocab)))
		self.tilde_b = nn.Parameter(torch.ones(len(self.vocab)))
	
	def forward(self, w, tilde_w):

		w_vecs = self.embeddings(w)
		b_vecs = self.b[w]

		tilde_w_vecs = self.projection(tilde_w)
		tilde_b_vecs = self.tilde_b[tilde_w]

		scores = torch.sum(w_vecs*tilde_w_vecs, dim=-1) + b_vecs + tilde_b_vecs

		return scores
		

	def word2emb(self, words):
		"""
		takes a list of words and returns its corresponding embeddings based on the vocab
		"""
		indices = torch.tensor(self.vocab.forward(words), device=self.dev)
		return self.embeddings(indices) + self.projection(indices)

	def ind2emb(self, indices):
		"""
		takes in a torch tensor of indices and returns the corresponding embeddings
		"""
		return self.embeddings(indices) + self.projection(indices)

	def word_sim(self, words1, words2, dim=1):
		indices1 = torch.tensor(self.vocab(words1), device=self.dev)
		indices2 = torch.tensor(self.vocab(words2), device=self.dev)

		emb1 = \
			self.embeddings(indices1) + \
			self.projection(indices1)

		emb2 = \
			self.embeddings(indices2) + \
			self.projection(indices2)

		return f.cosine_similarity(emb1, emb2)

	def sentence_sim(self, sentence1, sentence2, batch=False):
		
		if batch:
			batch_size = len(sentence1)

			# compute indices
			indices1 = []
			indices2 = []
			for s1, s2 in zip(sentence1, sentence2):
				indices1.append(torch.tensor(
					self.vocab(s1),
					dtype=torch.int64,
					device=self.dev
				))
				indices2.append(torch.tensor(
					self.vocab(s2),
					dtype=torch.int64,
					device=self.dev
				))

			# get all averaged embeddings
			emb1 = torch.vstack([
				torch.mean(
					self.word_weights(s)[:, None]*(self.embeddings(s) + self.projection(s)), 
					dim=0
				) 
				for s in indices1
			])
			emb2 = torch.vstack([
				torch.mean(
					self.word_weights(s)[:, None]*(self.embeddings(s) + self.projection(s)), 
					dim=0
				) 
				for s in indices2
			])

			# compute the sentence embeddings using the method of "a simple but tough to beat baseline..."
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
				self.vocab(sentence1),
				dtype=torch.int64,
				device=self.dev
			)
			indices2 = torch.tensor(
				self.vocab(sentence2),
				dtype=torch.int64,
				device=self.dev
			)

			# get embeddings
			emb1 = self.word_weights(indices1)[:, None]*(self.embeddings(indices1) + self.projection(indices1))
			emb2 = self.word_weights(indices2)[:, None]*(self.embeddings(indices2) + self.projection(indices2))

			# compute averaged embedding
			emb1 = torch.mean(emb1, dim=0)
			emb2 = torch.mean(emb2, dim=0)

			return f.cosine_similarity(emb1, emb2, dim=0)


