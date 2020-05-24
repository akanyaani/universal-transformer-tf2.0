from layers.attention_layer import *
from layers.feed_forward import *
from layers.layer_norm import *


class DecoderLayer(tf.keras.layers.Layer):
	def __init__(self, d_model, num_heads, dff,
	             dr_rate=0.1):
		super(DecoderLayer, self).__init__()
		self.num_heads = num_heads
		self.dff = dff
		self.dr_rate = dr_rate

		self.mha1 = MultiHeadAttention(d_model, num_heads)
		self.mha2 = MultiHeadAttention(d_model, num_heads)

		self.feed_forward = TransitionLayer(d_model,
		                                    dff,
		                                    dr_rate,
		                                    layer_type='ffn')

		self.layer_norm1 = LayerNormalization()
		self.layer_norm2 = LayerNormalization()
		self.layer_norm3 = LayerNormalization()

		self.dropout = tf.keras.layers.Dropout(self.dr_rate)

	def call(self, x, enc_output, training, mask, padding_mask=None):
		out = self.mha1(x, x, x, mask=mask,
		                training=training)
		with tf.name_scope("residual_conn"):
			out1 = self.layer_norm1(out + x)

		out2 = self.mha2(enc_output, enc_output, out1, mask=padding_mask,
		                 training=training)

		with tf.name_scope("residual_conn"):
			out2 = self.layer_norm1(out2 + x)

		ffn_out = self.feed_forward(out2, training=training)
		ffn_out = self.dropout(ffn_out, training=training)

		with tf.name_scope("residual_conn"):
			out3 = self.layer_norm3(ffn_out + out2)

		# print("Decoder output shape is :- ", out3.numpy().shape)
		return out3
