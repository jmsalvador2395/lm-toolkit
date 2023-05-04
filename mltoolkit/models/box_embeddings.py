import torch
from torch import nn
import torch.nn.functional as f
from torch.nn import init
import numpy as np

from utils import data

class BoxEmbeddings(nn.Module):
	
	def __init__(self, vocab, params, p):

		super(BoxEmbeddings, self).__init__()

		# gumbel parameters
		self.beta = 1.9678289474987882
		self.euler_gamma = 0.57721566490153286060
		self.volume_temp = 0.33243242379830407
		
		# set vocab
		self.vocab = vocab

		# set word probabilities
		self.w2p = p
		self.i2p = {vocab([w])[0] : prob for w, prob in p.items()}

		# set weight (this is from poitn embeddings but keeping it just in case
		self.alpha = params['alpha_weight']

		# set model device
		self.dev = params['device']

		self.embedding_dim = params['embedding_dim']
		
		# create box embeddings embeddings
		# embeddings[:, 0, :] is the lower coordinate, embeddings[:, 1, :] is the upper coordinate
		self.embeddings = nn.Parameter(
			torch.zeros(
				len(self.vocab),
				2,
				self.embedding_dim
			)
		)

		self.context = nn.Parameter(
			torch.zeros(
				len(self.vocab),
				2,
				self.embedding_dim
			)
		)
			
		with torch.no_grad():
			init.uniform_(self.embeddings, 1e-7, .9-1e-7)
			self.embeddings[:, 1, :] = self.embeddings[..., 0, :] + .1

			init.uniform_(self.context, 1e-7, .9-1e-7)
			self.context[:, 1, :] = self.context[..., 0, :] + .1
	
	def forward(self, indices):
		return self.embeddings[indices]
	
	def z(indices=None):
		if indices is None:
			return self.embeddings[:, 0, :]
		else:
			return self.embeddings[indices, 0, :]

	def Z(indices=None):
		if indices is None:
			return self.embeddings[:, 1, :]
		else:
			return self.embeddings[indices, 1, :]

	def center(indices=None):
		"""
		returns the center coordinat for the selected embeddings 
		"""
		if indices is None:
			return torch.mean(self.embeddings, dim=1)
		else:
			selections = self.embeddings[indices]
			return torch.mean(selections, dim=1)

	def max_gumbel(self, a, b=None, stack=False):
		
		if b is not None and not stack:
			intersection = self.beta*torch.logaddexp(
				a/self.beta,
				b/self.beta
			)
			#intersection = torch.maximum(intersection, torch.min(a, b))
			return intersection

		# a and b might be both provided so we decide what to do using the stack variable
		if stack:
			embeddings = torch.hstack((a, b))
		else:
			embeddings = a

		intersection = self.beta*torch.logsumexp(
			embeddings/self.beta,
			dim=1,
			keepdim=True
		)
		#intersection = torch.maximum(intersection, embeddings.min(dim=1, keepdim=True).values)
		return intersection
			
	
	def min_gumbel(self, a, b=None, stack=False):
		if b is not None and not stack:
			intersection = -self.beta*torch.logaddexp(
				-a/self.beta,
				-b/self.beta
			)
			#intersection = torch.minimum(intersection, torch.min(a, b))
			return intersection

		# a and b might be both provided so we decide what to do using the stack variable
		if stack:
			embeddings = torch.hstack((a, b))
		else:
			embeddings = a

		intersection = -self.beta*torch.logsumexp(
			-embeddings/self.beta,
			dim=1,
			keepdim=True
		)
		#intersection = torch.minimum(intersection, embeddings.min(dim=1, keepdim=True).values)
		return intersection

	def volume(self, z, Z):
		vol = torch.sum(
			torch.log(
				f.softplus(
					Z - z - 2 * self.euler_gamma * self.beta, 
					beta=self.volume_temp
				) + 1e-23
			),
			dim=-1,
		)
		return vol

	def sim(self, a, b, method='gumbel_vol', stack=False):
		low = range(0, 1); high = range(1, 2)
		if method == 'gumbel_vol':
			# compute intersections
			z = self.max_gumbel(
				a[..., low, :],
				b[..., low, :],
				stack=stack
			)
			Z = self.min_gumbel(
				a[..., high, :],
				b[..., high, :], 
				stack=stack
			)
			scores = torch.flatten(self.volume(z, Z))
			return scores
		elif method == 'fuzzy_jaccard':
			pass
		elif method == 'fuzzy_otsuka':
			pass
		elif method == 'fuzzy_dice':
			pass
		else:
			raise Exception('invalid method specified in function "sim(self, a, b, method)"')

	def word_sim(self, words1, words2, method='gumbel_vol'):
		indices1 = torch.tensor(self.vocab(words1), device=self.dev)
		indices2 = torch.tensor(self.vocab(words2), device=self.dev)

		mask = torch.ones(indices1.shape, dtype=torch.bool, device=self.dev)
		mask[indices1 == 0] = False
		mask[indices2 == 0] = False

		emb1 = self.embeddings[indices1][:, None, ...]
		emb2 = self.embeddings[indices2][:, None, ...]

		
		return \
			self.sim(emb1, emb2, method, stack=True), \
			mask
	
	def gumbel_sentence_sim(self, sentence1, sentence2, batch=False):

		low = range(0, 1); high = range(1, 2)
		if batch:
			# compute indices
			s1_intersections = []
			s2_intersections = []
			for s1, s2 in zip(sentence1, sentence2):
				indices1 = torch.tensor(
					self.vocab(s1),
					dtype=torch.int64,
					device=self.dev
				)
				indices2 = torch.tensor(
					self.vocab(s2),
					dtype=torch.int64,
					device=self.dev
				)
				
				word_embeddings1 = self.embeddings[indices1][None]
				word_embeddings2 = self.embeddings[indices2][None]

				s1_intersections.append(
					torch.cat(
						(self.max_gumbel(word_embeddings1[..., low, :]),
						self.min_gumbel(word_embeddings1[..., high, :])),
						dim=-2
					)
				)
				s2_intersections.append(
					torch.cat(
						(self.max_gumbel(word_embeddings2[..., low, :]),
						self.min_gumbel(word_embeddings2[..., high, :])),
						dim=-2
					)
				)

			s1_intersections = torch.vstack(s1_intersections)
			s2_intersections = torch.vstack(s2_intersections)

			return self.sim(
				s1_intersections,
				s2_intersections,
				method='gumbel_vol',
				stack=True
			)
		# compute sentence similarity for just 1 pair of sentences
		else:
			raise NotImplementedError('Have not implemented this yet')
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
			emb1 = self.embeddings(indices1) + self.projection(indices1)
			emb2 = self.embeddings(indices2) + self.projection(indices2)

			# compute averaged embedding
			emb1 = torch.mean(emb1, dim=0)
			emb2 = torch.mean(emb2, dim=0)

			return f.cosine_similarity(emb1, emb2, dim=0)
		
	
	def sentence_sim(self, sentence1, sentence2, batch=False, method='gumbel_vol'):
		if method == 'gumbel_vol':
			return self.gumbel_sentence_sim(sentence1, sentence2, batch=batch)
		else:
			raise Exception('invalid method for computing sentence similarity')

		


class Word2Box(BoxEmbeddings):
	
	def __init__(self, vocab, params, p):
		super(Word2Box, self).__init__(vocab, params, p)

	def forward(self, center_ids, context_ids, context_mask):

		# for indexing
		low = range(0, 1); high = range(1, 2)

		# get embeddings
		center = self.embeddings[center_ids]
		context = self.context[context_ids]

		center_mask = (center_ids == 0)
		center[..., 0, :].masked_fill_(center_mask[..., None], float('-inf'))
		center[..., 1, :].masked_fill_(center_mask[..., None], float('inf'))

		context_mask = (context_ids == 0)
		context[..., 0, :].masked_fill_(context_mask[..., None], float('-inf'))
		context[..., 1, :].masked_fill_(context_mask[..., None], float('inf'))

		# shape-dependent operations
		if len(center_ids.shape) < len(context_ids.shape):
			# add dimension so that logaddexp works
			center = center[:, None, :, :]
			stack=True # determines the operation for min_gumbel and max_gumbel
		else:
			# compute intersection of the context before intersecting with multiple center words
			context = torch.cat(
				(self.max_gumbel(context[..., low, :]),
				self.min_gumbel(context[..., high, :])),
				dim=-2
			)
			stack=False # determines the operation for min_gumbel and max_gumbel

		# compute intersections
		z = self.max_gumbel(
			center[..., low, :],
			context[..., low, :],
			stack=stack
		)
		Z = self.min_gumbel(
			center[..., high, :],
			context[..., high, :], 
			stack=stack
		)

		# compute volume
		score = self.volume(z, Z)

		#idx = torch.where(torch.flatten(score != torch.inf))[0]
		#idx = torch.where(score > 0)[0]

		return score.reshape(center_ids.shape)
		

class GloBE(BoxEmbeddings):
	def __init__(self, vocab, params, p):
		super(GloBE, self).__init__(vocab, params, p)

	def forward(self, i, j):
		# for indexing
		low = range(0, 1); high = range(1, 2)

		# get embeddings
		wi = self.embeddings[i][:, None, ...]
		wj = self.context[j][:, None, ...]

		# compute intersections
		z = self.max_gumbel(
			wi[..., low, :],
			wj[..., low, :],
			stack=True
		)
		Z = self.min_gumbel(
			wi[..., high, :],
			wj[..., high, :], 
			stack=True
		)

		# compute volume
		score = self.volume(z, Z)

		return score.reshape(i.shape)

	
