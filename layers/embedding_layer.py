import numpy as np
import tensorflow as tf


class EmbeddingLayer(tf.keras.layers.Layer):

	def __init__(self, vocab_size, embedding_size, initializer=None, stddev=0.01, mean=0.0):
		super(EmbeddingLayer, self).__init__()
		self.vocab_size = vocab_size
		self.embedding_size = embedding_size
		self.stddev = stddev
		self.mean = mean
		self.embedding_weights = None
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


class PositionEmbeddingLayer(tf.keras.layers.Layer):

	def __init__(self, max_seq_len,
	             pos_embedding_size,
	             trainable=True,
	             time_embd=False,
	             stddev=0.02,
	             mean=0.0):
		super(PositionEmbeddingLayer, self).__init__()
		self.max_seq_len = max_seq_len
		self.hidden_size = pos_embedding_size
		self.trainable = trainable
		self.time_embd = time_embd
		self.stddev = stddev
		self.mean = mean

		if self.trainable:
			self.position_embedding = EmbeddingLayer(self.max_seq_len + 1, self.hidden_size,
			                                         stddev=self.stddev, mean=self.mean)

	def call(self, inputs, time_step=None):

		with tf.name_scope("pos_embedding"):
			if self.trainable:
				batch_size = tf.shape(inputs)[0]
				seq_len = tf.shape(inputs)[1]

				if self.time_embd:
					print("Executing time")
					time_pos = tf.tile(tf.constant([[time_step + 1]], tf.int32), [batch_size, seq_len])
					# print("Time pos shape", time_pos.numpy().shape)
					return self.position_embedding(time_pos)

				positions = tf.reshape(tf.tile(tf.range(1, seq_len + 1), [batch_size]),
				                       [batch_size, seq_len])

				"""
				sample seq_len = 8
				sample batch_size = 2
				positions = <tf.Tensor: shape=(2, 8), dtype=int32, numpy=
				array([[0, 1, 2, 3, 4, 5, 6, 7],
					[0, 1, 2, 3, 4, 5, 6, 7]], dtype=int32)>
				"""

				# print("Position shape :-", tf.shape(positions))
				positions = tf.cast(positions, tf.int32)
				
				mask_int = tf.reduce_sum(inputs, axis=-1)

				# print(tf.shape(mask_int))
				# print(mask_int)

				position_mask = tf.cast(tf.not_equal(mask_int, 0), tf.int32)

				# print(position_mask)
				"""
				zeros are padded token id
				inputs = [[2, 3, 6, 0, 0], [2, 3, 0, 0, 0]]
				position_mask = <tf.Tensor: shape=(2, 5), dtype=int32, numpy=
				array([[1, 1, 1, 0, 0],
				[1, 1, 0, 0, 0]], dtype=int32)>
				"""
				positions *= position_mask

				return self.position_embedding(positions)
			else:
				print("Position encoding getting executed.")

				return positional_encoding(self.max_seq_len, self.hidden_size)


# https://www.tensorflow.org/tutorials/text/transformer
def get_angles(pos, i, d_model):
	angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
	return pos * angle_rates


def positional_encoding(position, d_model):
	angle_rads = get_angles(np.arange(position)[:, np.newaxis],
	                        np.arange(d_model)[np.newaxis, :],
	                        d_model)
	angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
	angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
	pos_encoding = tf.expand_dims(tf.cast(angle_rads, dtype=tf.float32), 0)
	return pos_encoding
