import tensorflow as tf

from layers.feed_forward import Conv1d


class MultiHeadAttention(tf.keras.layers.Layer):
	def __init__(self, d_model, num_heads, att_dropout=0.1, residual_dropout=0.1, scale=True):
		super(MultiHeadAttention, self).__init__()
		self.num_heads = num_heads
		self.d_model = d_model
		self.att_dropout = att_dropout
		self.residual_dropout = residual_dropout
		self.scale = scale

		assert d_model % self.num_heads == 0

		self.depth = d_model // self.num_heads

		self.wq = tf.keras.layers.Dense(d_model)
		self.wk = tf.keras.layers.Dense(d_model)
		self.wv = tf.keras.layers.Dense(d_model)

		self.c_proj = Conv1d(self.d_model, self.d_model)

	def multihead_attention(self, q, k, v, training, mask=None):
		matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)
		if self.scale:
			dk = tf.cast(tf.shape(k)[-1], tf.float32)
			matmul_qk = matmul_qk / tf.math.sqrt(dk)

		if mask is not None:
			matmul_qk += (mask * -1e9)

		attention_weights = tf.nn.softmax(matmul_qk, axis=-1)  # (..., seq_len_q, seq_len_k)

		if training:
			attention_weights = tf.nn.dropout(attention_weights, rate=self.att_dropout, name="attn_dropout")
		output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

		return output, attention_weights

	def split_heads(self, x):
		batch_size = tf.shape(x)[0]
		x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
		return tf.transpose(x, perm=[0, 2, 1, 3])

	def merge_heads(self, x):
		batch_size = tf.shape(x)[0]
		x = tf.transpose(x, perm=[0, 2, 1, 3])
		# (batch_size, seq_len_q, num_heads, depth)

		merged = tf.reshape(x, (batch_size, -1, self.d_model))
		# (batch_size, seq_len_q, d_model)
		return merged

	def call(self, x, mask=None, training=True):
		query, key, value = tf.split(x, axis=-1)
		query = self.split_heads(self.wq(query))
		key = self.split_heads(self.wk(key))
		value = self.split_heads(self.wv(value))

		scaled_attention, attention_weights = self.multihead_attention(query, key, value, training, mask)

		concat_attention = self.merge_heads(scaled_attention)

		output = self.c_proj(concat_attention)  # (batch_size, seq_len_q, d_model)
		if training:
			output = tf.nn.dropout(output, rate=self.residual_dropout, name="resid_dropout")

		return output
