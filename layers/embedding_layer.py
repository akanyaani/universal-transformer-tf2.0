import math

import tensorflow as tf


class EmbeddingLayer(tf.keras.layers.Layer):

	def __init__(self, vocab_size, embedding_size, initializer=None, stddev=0.01, mean=0.0):
		super(EmbeddingLayer, self).__init__()
		self.vocab_size = vocab_size
		self.embedding_size = embedding_size
		self.stddev = stddev
		self.mean = mean
		self.initializer = initializer
		if self.initializer is None:
			self.initializer = tf.random_normal_initializer(mean=self.mean,
															stddev=self.stddev)

	def build(self, input_shape):
		with tf.name_scope("embedding_weights"):
			self.embedding_weights = self.add_weight(
				"weights",
				shape=[self.vocab_size, self.embedding_size],
				dtype="float32",
				initializer=self.initializer
			)
		super(EmbeddingLayer, self).build(input_shape)

	def call(self, inputs, mode="embedding", scale=False):
		if mode == "embedding":
			return self.embedding(inputs, scale=scale)
		elif mode == "projection":
			return self.projection(inputs)
		else:
			raise ValueError("mode {} is not valid.".format(mode))

	def embedding(self, inputs, scale=False):
		with tf.name_scope("embedding"):
			# Create binary mask of size [batch_size, length]
			mask = tf.cast(tf.not_equal(inputs, 0), tf.float32)
			inputs = tf.cast(inputs, tf.int32)
			embeddings = tf.nn.embedding_lookup(self.embedding_weights, inputs)
			embeddings *= tf.expand_dims(mask, -1)
			# Scale embedding by the sqrt of the hidden size
			if scale:
				embeddings *= self.embedding_size ** 0.5

			return embeddings

	def projection(self, inputs):
		with tf.name_scope("output_layer"):
			batch_size = tf.shape(inputs)[0]
			seq_len = tf.shape(inputs)[1]

			h_flat = tf.reshape(inputs, [-1, self.embedding_size])
			logits = tf.matmul(h_flat, self.embedding_weights, transpose_b=True)

			return tf.reshape(logits, [batch_size, seq_len, self.vocab_size])


def get_position_embedding(seq_len, hidden_size, min_timescale=1.0, max_timescale=1.0e4):
	position = tf.cast(tf.range(seq_len), tf.float32)
	num_timescales = hidden_size // 2
	log_timescale_increment = (
			math.log(float(max_timescale) / float(min_timescale)) /
			(tf.cast(num_timescales, tf.float32) - 1))
	inv_timescales = min_timescale * tf.exp(
		tf.cast(tf.range(num_timescales), tf.float32) * -log_timescale_increment)
	scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(inv_timescales, 0)
	signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
	return signal
