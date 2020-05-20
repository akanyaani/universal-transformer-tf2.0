from layers.attention_layer import *
from layers.feed_forward import *
from layers.layer_norm import *


class EncoderLayer(tf.keras.layers.Layer):
	def __init__(self,
	             d_model,
	             num_heads,
	             dff,
	             dr_rate=0.1):
		super(EncoderLayer, self).__init__()
		self.d_model = d_model
		self.num_heads = num_heads
		self.dff = dff
		self.dr_rate = dr_rate

		self.mha = MultiHeadAttention(self.d_model, self.num_heads)
		self.feed_forward = TransitionLayer(self.d_model,
		                                    self.dff,
		                                    self.dr_rate,
		                                    layer_type='ffn')

		self.layer_norm1 = LayerNormalization()
		self.layer_norm2 = LayerNormalization()

	def call(self, x, training, mask):
		out, present = self.mha(self.layer_norm1(x),
		                        mask=mask,
		                        training=training)  # (batch_size, input_seq_len, d_model)
		with tf.name_scope("residual_conn"):
			x = x + out
		out = self.feed_forward(self.layer_norm2(x), training=training)  # (batch_size, input_seq_len, d_model)
		with tf.name_scope("residual_conn"):
			x = x + out
		return x, present
