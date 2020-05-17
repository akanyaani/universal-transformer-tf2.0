from layers.feed_forward import *


class EncoderLayer(tf.keras.layers):
	def __init__(self):
		super(EncoderLayer, self).__init__()
		self.d_model = d_model
		self.num_heads = num_heads
		self.dff = dff
		self.dr_rate = dr_rate

		self.mha = MultiHeadAttention(self.d_model, self.num_heads)
		self.feed_forward = FeedForward(self.d_model, self.dff, self.dr_rate)
		self.layer_norm1 = LayerNormalization(self.d_model)
		self.layer_norm2 = LayerNormalization(self.d_model)

	def call(self):
		out, present = self.mha(self.layer_norm1(x), mask=mask, past_layer=past,
								training=training)  # (batch_size, input_seq_len, d_model)
		with tf.name_scope("residual_conn"):
			x = x + out
		out = self.feed_forward(self.layer_norm2(x), training=training)  # (batch_size, input_seq_len, d_model)
		with tf.name_scope("residual_conn"):
			x = x + out
		return x, present
		return None
