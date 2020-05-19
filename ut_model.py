import os

from tensorflow.python.framework import tensor_shape

from layers.decoder_layer import DecoderLayer
from layers.embedding_layer import *
from layers.encoder_layer import *
from utils.tf_utils import *

_ROOT = os.path.abspath(os.path.dirname(__file__))
LOG_DIR = _ROOT + "/log"

train_step_signature = [
	tf.TensorSpec(shape=(None, None), dtype=tf.int32, name="Inputs"),
	tf.TensorSpec(shape=(None, None), dtype=tf.int32, name="Targets"),
]


class UTModel(tf.keras.Model):
	def __init__(self, num_layers,
	             d_model,
	             num_heads,
	             dff,
	             max_seq_len,
	             inp_vocab_size=32000,
	             out_vocab_size=32000,
	             optimizer="adam",
	             learning_rate=1e-3,
	             rev_embd_proj=True,
	             pos_n_time_train=False):
		super(UTModel, self).__init__()

		self.optimizer = None
		self.ckpt_manager = None
		self.train_writer = None
		self.test_writer = None

		self.rev_embd_proj = rev_embd_proj
		self.num_layers = num_layers
		self.num_heads = num_heads
		self.dff = dff
		self.max_seq_len = max_seq_len
		self.inp_vocab_size = inp_vocab_size
		self.out_vocab_size = out_vocab_size

		self.d_model = d_model
		self.learning_rate = learning_rate
		self.optimizer_t = optimizer

		self.encoder = Encoder(self.num_layers,
		                       self.d_model,
		                       self.num_heads,
		                       self.dff,
		                       self.act,
		                       self.inp_vocab_size,
		                       self.max_seq_len,
		                       pos_n_time_train=pos_n_time_train)

		self.decoder = Decoder(self.num_layers,
		                       self.d_model,
		                       self.num_heads,
		                       self.dff,
		                       self.act,
		                       self.out_vocab_size,
		                       self.max_seq_len,
		                       pos_n_time_train=pos_n_time_train)

		self.projection_layer = OutputLayer(self.out_vocab_size,
		                                    proj_weights=None)

	def call(self, x, training=True):
		inp, tar = x

		enc_out = self.encoder(inp, training=training)
		dec_out = self.decoder(tar, enc_out, training=training)
		logits = self.projection_layer(dec_out)
		return logits

	@staticmethod
	def get_padded_accuracy(labels, logits):
		with tf.name_scope("padded_accuracy"):
			weights = tf.cast(tf.not_equal(labels, 0), tf.float32)

			outputs = tf.cast(tf.argmax(logits, axis=-1), tf.int32)
			padded_labels = tf.cast(labels, tf.int32)

			nonpad_seq = tf.math.count_nonzero(weights, dtype=tf.dtypes.float32, )
			acc = tf.cast(tf.equal(outputs, padded_labels), tf.float32)

			accuracy = tf.reduce_sum(tf.cast(acc * weights, tf.float32)) / nonpad_seq
			return tf.cast(accuracy, tf.float32)

	def create_optimizer(self):
		optimizer = self.optimizer_t.lower()
		with tf.name_scope("optimizer"):
			if optimizer == "adam":
				self.optimizer = tf.keras.optimizers.Adam(self.learning_rate, beta_1=0.9, beta_2=0.98,
				                                          epsilon=1e-9)
			elif optimizer == "adadelta":
				self.optimizer = tf.keras.optimizers.Adadelta(self.learning_rate)
			elif optimizer == "rms":
				self.optimizer = tf.keras.optimizers.RMSprop(self.learning_rate)
			else:
				self.optimizer = tf.keras.optimizers.SGD(self.learning_rate)
			return self.optimizer

	def get_loss(self, real, pred):
		with tf.name_scope("loss_layer"):
			mask = tf.math.logical_not(tf.math.equal(real, 0))
			loss_ = self.loss_object(real, pred)

			with tf.name_scope("loss_masking"):
				mask = tf.cast(mask, dtype=loss_.dtype)
				loss_ *= mask
			loss_ = tf.reduce_sum(loss_, axis=1)
			sequence_avg_loss = loss_ / tf.reduce_sum(mask, axis=1)
			return sequence_avg_loss

	def create_checkpoint_manager(self, checkpoint_path, max_to_keep=5, load_model=True):
		with tf.name_scope('checkpoint_manager'):
			ckpt = tf.train.Checkpoint(optimizer=self.optimizer, model=self)
			self.ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=max_to_keep)

			if load_model:  # If want to load trained weights
				ckpt.restore(self.ckpt_manager.latest_checkpoint)
				print('Latest checkpoint restored...............')
			else:
				print("Initializing model from scratch..........")

	def load_model(self, filepath):
		ckpt = tf.train.Checkpoint(model=self)
		ckpt_manager = tf.train.CheckpointManager(ckpt, filepath)
		ckpt.restore(ckpt_manager.latest_checkpoint)
		print("Model Restored..........................")

	def create_summary_writer(self, summary_path):
		train_summary_path = summary_path + "/train"
		test_summary_path = summary_path + "/test"

		with tf.name_scope('summary'):
			self.train_writer = tf.summary.create_file_writer(train_summary_path)
			self.test_writer = tf.summary.create_file_writer(test_summary_path)

			return self.train_writer, self.test_writer

	# @tf.function(input_signature=train_step_signature)
	def train_step(self, inputs, targets, step, grad_clip=True, clip_value=2.5):

		with tf.GradientTape() as tape:
			predictions, _ = self(inputs, training=True)
			loss = tf.reduce_mean(self.get_loss(targets, predictions))

		with tf.name_scope("gradients"):
			gradients = tape.gradient(loss, self.trainable_variables)
			if grad_clip:
				gradients = [(tf.clip_by_value(grad, -clip_value, clip_value))
				             for grad in gradients]
			self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

		accuracy = self.get_padded_accuracy(targets, predictions)

		with tf.name_scope("summary_writer"):
			with self.train_writer.as_default():
				tf.summary.scalar("loss", loss, step=tf.cast(step, tf.int64))
				tf.summary.scalar("accuracy", accuracy, step=tf.cast(step, tf.int64))

		return loss, accuracy

	@tf.function
	def distributed_train_step(self, inputs, targets, step, grad_clip=True, clip_value=1.0):
		def step_fn(inp, tar):
			with tf.GradientTape() as tape:
				logits = self(inputs)
				cross_entropy = self.get_loss(targets, logits)
				loss = tf.reduce_mean(cross_entropy)

			with tf.name_scope("gradients"):
				gradients = tape.gradient(loss, self.trainable_variables)
				if grad_clip:
					gradients = [(tf.clip_by_value(grad, -clip_value, clip_value))
					             for grad in gradients]
				self.optimizer.apply_gradients(list(zip(gradients, self.trainable_variables)))
			return cross_entropy

		per_example_losses = self.mirrored_strategy.experimental_run_v2(
			step_fn, args=(inputs, targets))
		mean_loss = self.mirrored_strategy.reduce(
			tf.distribute.ReduceOp.MEAN, per_example_losses, axis=0)

		with tf.name_scope("summary_writer"):
			with self.train_writer.as_default():
				tf.summary.scalar("loss", mean_loss, step=step)
		return mean_loss

	def fit(self, dataset):
		if self.mirrored_strategy is None:
			train_dataset, test_dataset = dataset
			tf.summary.trace_on(graph=True, profiler=True)
			for (step, (inputs, targets)) in enumerate(train_dataset):

				print(inputs)
				print(targets)

				train_loss, train_acc = self.train_step(inputs, targets, step)
				if step % 10 == 0:
					print('Step {} Train_Loss {:.4f} Train_Accuracy {:.4f}'.format(
						step, train_loss, train_acc))

				if step == 0:
					with self.train_writer.as_default():
						tf.summary.trace_export(
							name="gpt-2",
							step=0,
							profiler_outdir=LOG_DIR)

				if step % 1000 == 0:
					ckpt_save_path = self.ckpt_manager.save()
					print('Saving checkpoint for step {} at {}'.format(step,
					                                                   ckpt_save_path))
		else:
			with self.mirrored_strategy.scope():
				tf.summary.trace_on(graph=True, profiler=True)
				for (step, (inputs)) in enumerate(dataset):
					train_loss = self.distributed_train_step(inputs, step)
					if step == 0:
						with self.train_writer.as_default():
							tf.summary.trace_export(
								name="gpt-2",
								step=0,
								profiler_outdir=LOG_DIR)
					if step % 100 == 0:
						print('Step {} Train_Loss {:.4f}'.format(
							step, train_loss))
					if step % 1000 == 0:
						ckpt_save_path = self.ckpt_manager.save()
						print('Saving checkpoint for step {} at {}'.format(step,
						                                                   ckpt_save_path))


class OutputLayer(tf.keras.layers.Layer):
	def __init__(self, output_dim, proj_weights=None, kernel_initializer=None):
		super(OutputLayer, self).__init__()
		self.proj_weights = proj_weights
		self.output_dim = output_dim
		self.layer_weights = None
		self.kernel_initializer = kernel_initializer

	def build(self, input_shape):
		if self.proj_weights is None:
			input_dim = tensor_shape.dimension_value(input_shape[-1])
			self.layer_weights = self.add_weight(
				'output_layer_weights',
				shape=[input_dim, self.output_dim],
				initializer=self.kernel_initializer,
				trainable=True)
		super(OutputLayer, self).build(input_shape)

	def call(self, x):
		batch, sequence, d_model = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[-1]
		h_flat = tf.reshape(x, [-1, d_model])

		if self.proj_weights is None:
			out = tf.matmul(h_flat, self.layer_weights)
		else:
			out = tf.matmul(h_flat, self.porj_weights, transpose_b=True)
		out = tf.reshape(out, [batch, sequence, self.output_dim])
		return out


class Encoder(tf.keras.layers.Layer):
	def __init__(self,
	             num_layers,
	             d_model,
	             num_heads,
	             dff,
	             act,
	             inp_vocab_size,
	             max_seq_len,
	             dr_rate=0.1,
	             pos_n_time_train=False):
		super(Encoder, self).__init__()
		self.num_layers = num_layers
		self.act = act
		self.max_seq_len = max_seq_len
		self.dr_rate = dr_rate

		self.embedding_layer = EmbeddingLayer(inp_vocab_size, d_model)
		self.pos_embedding_layer = PositionEmbeddingLayer(self.max_seq_len,
		                                                  d_model,
		                                                  trainable=pos_n_time_train)
		self.time_embedding_layer = PositionEmbeddingLayer(self.num_layers,
		                                                   d_model,
		                                                   trainable=pos_n_time_train)

		self.dropout = tf.keras.layers.Dropout(self.dr_rate)
		self.encoder_layer = EncoderLayer(d_model, num_heads, dff,
		                                  dr_rate=self.dr_rate)

	def call(self, x, training, mask, past=None):
		with tf.name_scope("embeddings"):
			out = self.embedding_layer(x)
			out = out + self.pos_embedding_layer(x)
			out = out + self.time_embedding_layer()[:0:]

			# Applying embedding dropout
			out = self.dropout(out, training=training)

		if self.act:
			raise Exception("Not implemented")
		else:
			for layer in range(self.num_layers):
				# Adding time signal at start of every layer
				out = out + self.time_embedding_layer()[:layer:]
				out = self.encoder_layer(out, training, mask)

		return out


class Decoder(tf.keras.layers.Layer):
	def __init__(self,
	             num_layers,
	             d_model,
	             num_heads,
	             dff,
	             act,
	             inp_vocab_size,
	             max_seq_len,
	             dr_rate=0.1,
	             pos_n_time_train=False):
		super(Decoder, self).__init__()
		self.num_layers = num_layers
		self.d_model = d_model
		self.act = act
		self.max_seq_len = max_seq_len
		self.dr_rate = dr_rate

		self.embedding_layer = EmbeddingLayer(inp_vocab_size, self.d_model)
		self.pos_embedding_layer = PositionEmbeddingLayer(self.max_seq_len,
		                                                  self.d_model,
		                                                  trainable=pos_n_time_train)
		self.time_embedding_layer = PositionEmbeddingLayer(self.num_layers,
		                                                   self.d_model,
		                                                   trainable=pos_n_time_train)

		self.dropout = tf.keras.layers.Dropout(self.dr_rate)
		self.decoder_layer = DecoderLayer(d_model, num_heads, dff,
		                                  dr_rate=self.dr_rate)

	def call(self, x, enc_output, training, mask):
		with tf.name_scope("embeddings"):
			out = self.embedding_layer(x)
			out = out + self.pos_embedding_layer(x)
			out = out + self.time_embedding_layer()[:0:]

			# Applying embedding dropout
			out = self.dropout(out, training=training)

		if self.act:
			raise Exception("Not implemented")
		else:
			for layer in range(self.num_layers):
				# Adding time signal at start of every layer
				out = out + self.time_embedding_layer()[:layer:]
				out = self.decoder_layer(out, enc_output, training, mask)

		return out
