import tensorflow as tf
import misc

class model:
	def __init__(self, params):
		self.params = params
		self.context = tf.placeholder(tf.string)
		self.question = tf.placeholder(tf.string)
		vocab, vectors = misc.prepare_embeds()
		self.embeddings = tf.constant(vectors)
		self.embedding_size = 50
		self.vocab2id = tf.contrib.lookup.index_table_from_tensor(vocab,default_value=0)
		self.global_step = tf.Variable(0)

	def one_encode_block(self, input, scope):
		with tf.name_scope(scope):
			input_shape = input.get_shape()[-2].value
			pe_matrix = misc.get_postion_enc(self.embedding_size, input_shape)
			pe_input = tf.multiply(pe_matrix, input)
			layernorm = tf.contrib.layers.layer_norm(pe_input)


	def forward(self):
		pass
