
import random
import numpy as np
import tensorflow as tf
import os
from tqdm import tqdm

from .. import Config
from ..emb import LanguageModel
from . import data_util
from ..tf_util import checkpoint_utils
from .. import train

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def create_lm_embeddings():
  config = Config.flags
  seed = config.seed
  random.seed(seed)
  np.random.seed(seed)

  if config.gpu_memory_growth:
    gpu_config = tf.ConfigProto()
    gpu_config.gpu_options.allow_growth = True
  else:
    gpu_config = None

  with tf.Graph().as_default(), tf.Session(config=gpu_config) as session:
    tf.set_random_seed(seed)
    print('Loading tokens...')
    tokens_dict = data_util.load_metathesaurus_token_data(config.data_dir)
    entity_count, max_token_length = tokens_dict['token_ids'].shape
    truncated_lengths = np.clip(tokens_dict['token_lengths'], 1, max_token_length)

    dataset = tf.data.Dataset.from_tensor_slices(np.arange(entity_count))
    dataset = dataset.batch(config.batch_size).prefetch(buffer_size=2)
    iterator = dataset.make_one_shot_iterator()
    entity_ids = iterator.get_next()

    token_ids = tf.get_variable(
      name='token_ids',
      shape=[entity_count, max_token_length],
      dtype=tf.int64,
      initializer=tf.constant_initializer(tokens_dict['token_ids'])
    )
    token_lengths = tf.get_variable(
      name='token_lengths',
      shape=[entity_count],
      dtype=tf.int64,
      initializer=tf.constant_initializer(truncated_lengths)
    )
    entity_token_ids = tf.nn.embedding_lookup(token_ids, entity_ids)
    entity_token_lengths = tf.nn.embedding_lookup(token_lengths, entity_ids)

    print('Loading bert...')
    lm = LanguageModel.BertLanguageModel(
      bert_config_path=config.bert_config,
      train_bert=False
    )
    entity_seq_embeddings = lm.encode(entity_token_ids, entity_token_lengths)

    checkpoint_utils.init_from_checkpoint(config.encoder_checkpoint)

    tf.global_variables_initializer().run()
    tf.local_variables_initializer().run()

    embedding_dir = os.path.join(config.secondary_data_dir, 'lm_embeddings')
    if not os.path.exists(embedding_dir):
      os.mkdir(embedding_dir)

    nrof_batches = int(np.ceil(entity_count / config.batch_size))
    print('Saving language model sequence embeddings...')
    for b_idx in tqdm(range(nrof_batches), total=nrof_batches):
      entity_batch, entity_batch_embeddings, entity_batch_token_ids, entity_batch_token_lengths = session.run(
        [entity_ids, entity_seq_embeddings, entity_token_ids, entity_token_lengths]
      )
      for e_idx, e_emb, e_b_t_ids, e_b_t_l in \
            zip(entity_batch, entity_batch_embeddings, entity_batch_token_ids, entity_batch_token_lengths):
        e_file = os.path.join(embedding_dir, f'{e_idx}.tfexample')
        feature = {
          'lm_embedding': tf.train.Feature(float_list=tf.train.FloatList(value=np.reshape(e_emb[:e_b_t_l], e_b_t_l * e_emb.shape[1]))),
          'lm_embedding_size': _int64_feature(e_emb.shape[1]),
          'token_ids': tf.train.Feature(int64_list=tf.train.Int64List(value=e_b_t_ids[:e_b_t_l])),
          'token_length': _int64_feature(e_b_t_l),
          'entity_id': _int64_feature(e_idx)
        }

        with open(e_file, 'wb') as f:
          example_proto_str = tf.train.Example(
            features=tf.train.Features(feature=feature)
          ).SerializeToString()
          f.write(example_proto_str)


if __name__ == '__main__':
  create_lm_embeddings()
