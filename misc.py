import _pickle as pickle
import numpy as np
import os

data_dir = './data/'
embeddings_dir = os.path.join(data_dir, 'glove.6B.50d.txt')

def prepare_embeds(dir=embeddings_dir):
	try:

		with open(os.path.join(data_dir, "embeds.pkl"), 'rb') as f:
			embeds = pickle.load(f)
		print("embeddings exist.")
		vocab = embeds["vocab"]
		vectors = embeds["vectors"]
		print("returning vectos and vocab.")
		return vocab, vectors
	except:
		print("creating embeddings.")
		with open(dir) as f:
			x = f.readlines()
		vocab = []
		vectors = []
		for i in x:
			line = i.strip('\n').split(' ')
			if line[0]=="unk":
				vocab.insert(0, "unk")
				vectors.insert(0, map(np.float32, line[1:]))
				continue
			vocab.append(line[0])
			vectors.append(map(np.float32, line[1:]))
		with open(os.path.join(data_dir, 'embeds.pkl'), 'wb') as f:
			pickle.dump({'vocab':vocab, 'vectors':vectors}, f)
		print("done creating embeddings.")
		return vocab, vectors

def get_postion_enc(embedding_dim, max_seq_len):
	pe_matrix = np.array(
		[[pos / np.power(10000, 2. * i / embedding_dim) for i in range(embedding_dim)] for pos in range(max_seq_len)])
	pe_matrix[:, 0::2] = np.sin(pe_matrix[:, 0::2])
	pe_matrix[:, 1::2] = np.cos(pe_matrix[:, 1::2])

	return pe_matrix