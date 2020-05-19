import tensorflow as tf
import tensorflow_datasets as tfds

# Adopted this data pipeline from https://www.tensorflow.org/tutorials/text/transformer
def load_data():
	examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en', with_info=True,
	                               as_supervised=True)
	train_examples, val_examples = examples['train'], examples['validation']

	return train_examples, val_examples


def build_sub_word_tokenizer(train_examples, vocab_size=32000):
	tokenizer_en = tfds.features.text.SubwordTextEncoder.build_from_corpus(
		(en.numpy() for pt, en in train_examples), target_vocab_size=vocab_size)

	tokenizer_pt = tfds.features.text.SubwordTextEncoder.build_from_corpus(
		(pt.numpy() for pt, en in train_examples), target_vocab_size=vocab_size)

	return tokenizer_en, tokenizer_pt


def encode(lang1, lang2, tokenizer_pt, tokenizer_en):
	lang1 = [tokenizer_pt.vocab_size] + tokenizer_pt.encode(
		lang1.numpy()) + [tokenizer_pt.vocab_size + 1]

	lang2 = [tokenizer_en.vocab_size] + tokenizer_en.encode(
		lang2.numpy()) + [tokenizer_en.vocab_size + 1]

	return lang1, lang2


def tf_encode(pt, en):
	result_pt, result_en = tf.py_function(encode, [pt, en], [tf.int64, tf.int64])
	result_pt.set_shape([None])
	result_en.set_shape([None])

	return result_pt, result_en


def filter_max_length(x, y, max_length=500):
	return tf.logical_and(tf.size(x) <= max_length,
	                      tf.size(y) <= max_length)


def make_dataset(buffer_size=20000):
	train_examples, val_examples = load_data()
	train_dataset = train_examples.map(tf_encode)
	train_dataset = train_dataset.filter(filter_max_length)
	train_dataset = train_dataset.shuffle(buffer_size)
    val_dataset = val_examples.map(tf_encode).filter(filter_max_length)
    return train_dataset, val_dataset

def input_fn(batch_size=32, padded_shapes=([-1], [-1]), epoch=10, buffer_size=10000):
    dataset = make_dataset(buffer_size)
    dataset = dataset.padded_batch(batch_size, padded_shapes=padded_shapes)
    dataset = dataset.repeat(epoch)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset