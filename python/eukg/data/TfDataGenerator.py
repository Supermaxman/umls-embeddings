
import os
import tensorflow as tf
import numpy as np
import zlib

from . import data_util
from tqdm import tqdm

from collections import defaultdict


class TfDataGenerator:
  def __init__(self, data_dir, batch_size, num_workers, buffer_size):
    self.data_dir = data_dir
    self.batch_size = batch_size

    self.num_workers = num_workers
    self.buffer_size = buffer_size

  def load_train(self, session):
    session.run(
      self.iterator.initializer,
      feed_dict={
        self.data_filepath: os.path.join(self.data_dir, 'triples', 'train.tfrecords'),
      }
    )

  def load_val(self, session):
    session.run(
      self.iterator.initializer,
      feed_dict={
        self.data_filepath: os.path.join(self.data_dir, 'triples', 'val.tfrecords'),
      }
    )

  def load_test(self, session):
    session.run(
      self.iterator.initializer,
      feed_dict={
        self.data_filepath: os.path.join(self.data_dir, 'triples', 'test.tfrecords'),
      }
    )

  def create_iterator(self):
    # Get all concepts which have some relation for negative sampling (these should be dense

    self.data_filepath = tf.placeholder(tf.string, [])

    dataset = tf.data.TFRecordDataset(filenames=self.data_filepath)

    features = {
      'r_idx': tf.io.FixedLenFeature([], tf.int64),
      'subj_id': tf.io.FixedLenFeature([], tf.int64),
      'subj_token_ids': tf.io.VarLenFeature(tf.int64),
      'subj_token_length': tf.io.FixedLenFeature([], tf.int64),
      'rt_id': tf.io.FixedLenFeature([], tf.int64),
      'rt_token_ids': tf.io.VarLenFeature(tf.int64),
      'rt_token_length': tf.io.FixedLenFeature([], tf.int64),
      'obj_id': tf.io.FixedLenFeature([], tf.int64),
      'obj_token_ids': tf.io.VarLenFeature(tf.int64),
      'obj_token_length': tf.io.FixedLenFeature([], tf.int64),
    }

    def get_padding_values():
      concept_pad = {
        'token_ids': [None],
        'token_length': [],
        'entity_id': []
      }
      rel_pad = {
        'token_ids': [None],
        'token_length': [],
        'entity_id': []
      }
      return concept_pad, rel_pad, concept_pad

    def parse_single_example(example_proto):
      example = tf.io.parse_single_example(
        example_proto,
        features
      )
      # TODO move this somewhere else
      max_token_length_truncate = 40
      example['subj_token_length'] = tf.minimum(example['subj_token_length'], max_token_length_truncate)
      example['obj_token_length'] = tf.minimum(example['obj_token_length'], max_token_length_truncate)
      example['rt_token_length'] = tf.minimum(example['rt_token_length'], max_token_length_truncate)

      example['subj_token_ids'] = tf.sparse.to_dense(example['subj_token_ids'])
      example['obj_token_ids'] = tf.sparse.to_dense(example['obj_token_ids'])
      example['rt_token_ids'] = tf.sparse.to_dense(example['rt_token_ids'])
      example['subj_token_ids'] = example['subj_token_ids'][:max_token_length_truncate]
      example['obj_token_ids'] = example['obj_token_ids'][:max_token_length_truncate]
      example['rt_token_ids'] = example['rt_token_ids'][:max_token_length_truncate]

      max_token_length = example['subj_token_length']
      max_token_length = tf.maximum(example['obj_token_length'], max_token_length)
      max_token_length = tf.maximum(example['rt_token_length'], max_token_length)

      example['subj_token_ids'] = tf.pad(
        example['subj_token_ids'],
        paddings=[
          [0, max_token_length - example['subj_token_length']]
        ]
      )
      example['obj_token_ids'] = tf.pad(
        example['obj_token_ids'],
        paddings=[
          [0, max_token_length - example['obj_token_length']]
        ]
      )
      example['rt_token_ids'] = tf.pad(
        example['rt_token_ids'],
        paddings=[
          [0, max_token_length - example['rt_token_length']]
        ]
      )
      subj_ex = {
        'entity_id': example['subj_id'],
        'token_ids': example['subj_token_ids'],
        'token_length': example['subj_token_length']
      }
      obj_ex = {
        'entity_id': example['obj_id'],
        'token_ids': example['obj_token_ids'],
        'token_length': example['obj_token_length']
      }
      rt_ex = {
        'entity_id': example['rt_id'],
        'token_ids': example['rt_token_ids'],
        'token_length': example['rt_token_length']
      }
      return subj_ex, rt_ex, obj_ex

    dataset = dataset.map(
      map_func=parse_single_example,
      num_parallel_calls=self.num_workers
    )
    dataset = dataset.shuffle(
      buffer_size=10000,
      reshuffle_each_iteration=True
    )
    dataset = dataset.padded_batch(
      batch_size=self.batch_size,
      padded_shapes=get_padding_values()
    )

    dataset = dataset.prefetch(
      buffer_size=self.buffer_size
    )

    iterator = dataset.make_initializable_iterator()

    self.iterator = iterator

    batch = iterator.get_next()
    return batch



class TfTestDataGenerator:
  def __init__(self, data, data_dir, secondary_data_dir, batch_size,
               lm_encoder_size, num_workers, buffer_size):
    self.data = data
    self.data_dir = data_dir
    self.secondary_data_dir = secondary_data_dir
    self.batch_size = batch_size

    self.lm_encoder_size = lm_encoder_size
    self.num_workers = num_workers
    self.buffer_size = buffer_size

  def load_concepts(self, session):
    test_data = data_util.load_metathesaurus_test_data(self.data_dir)
    concepts = np.unique(
      np.concatenate(
        [self.data['obj'], test_data['obj'], self.data['subj'], test_data['subj']]
      )
    )
    session.run(
      self.concept_iterator.initializer,
      feed_dict={
        self.concepts_placeholder: concepts
      }
    )

  def load_rels(self, session):
    test_data = data_util.load_metathesaurus_test_data(self.data_dir)
    rels = np.unique(
      np.concatenate(
        [self.data['rel'], test_data['rel']]
      )
    )
    session.run(
      self.rel_iterator.initializer,
      feed_dict={
        self.rels_placeholder: rels
      }
    )

  def create_concept_iterator(self):

    self.concepts_placeholder = tf.placeholder(tf.int32, [None])

    dataset = tf.data.Dataset.from_tensor_slices(self.concepts_placeholder)
    lm_embedding_dir = os.path.join(self.secondary_data_dir, 'lm_embeddings')

    features = {
      'lm_embedding': tf.io.VarLenFeature(tf.float32),
      'lm_embedding_size': tf.io.FixedLenFeature([], tf.int64),
      # 'token_ids': tf.io.VarLenFeature(tf.int64),
      'token_length': tf.io.FixedLenFeature([], tf.int64),
      'entity_id': tf.io.FixedLenFeature([], tf.int64)
    }

    def transform_to_path(x):
      return tf.strings.join([lm_embedding_dir + '/', tf.strings.as_string(x), '.tfexample'])

    def read_file(file_path):
      results = tf.io.read_file(file_path)
      results = tf.io.decode_compressed(results, compression_type='ZLIB')
      return results

    def parse_batch_example(b_concept_ids):
      bsize = tf.shape(b_concept_ids)[0]

      b_paths = transform_to_path(b_concept_ids)

      b_concepts = tf.map_fn(
        fn=read_file,
        elems=b_paths,
        dtype=tf.string,
        parallel_iterations=self.num_workers
      )
      b_concept_exs = tf.io.parse_example(b_concepts, features=features)
      b_concept_lengths = tf.reshape(
        b_concept_exs['token_length'],
        shape=[bsize]
      )
      b_max_token_length = tf.cast(tf.reduce_max(b_concept_lengths), tf.int32)
      b_concept_embs = tf.reshape(
        tf.sparse_tensor_to_dense(
          b_concept_exs['lm_embedding'],
          default_value=0
        ),
        shape=[bsize, b_max_token_length, self.lm_encoder_size]
      )

      return b_concept_embs, b_concept_lengths, b_concept_ids

    dataset = dataset.batch(
      batch_size=self.batch_size
    )
    dataset = dataset.map(
      map_func=parse_batch_example,
      num_parallel_calls=self.num_workers
    )

    dataset = dataset.prefetch(
      buffer_size=self.buffer_size
    )

    iterator = dataset.make_initializable_iterator()

    self.concept_iterator = iterator
    batch = iterator.get_next()
    b_concept_embs, b_concept_lengths, b_concept_ids = batch

    b_concept_embs.set_shape([None, None, self.lm_encoder_size])

    self.b_concept_embs = b_concept_embs
    self.b_concept_lengths = b_concept_lengths
    self.b_concept_ids = b_concept_ids

  def create_rel_iterator(self):
    self.rels_placeholder = tf.placeholder(tf.int32, [None])

    dataset = tf.data.Dataset.from_tensor_slices(self.rels_placeholder)
    lm_embedding_dir = os.path.join(self.secondary_data_dir, 'lm_embeddings')

    features = {
      'lm_embedding': tf.io.VarLenFeature(tf.float32),
      'lm_embedding_size': tf.io.FixedLenFeature([], tf.int64),
      # 'token_ids': tf.io.VarLenFeature(tf.int64),
      'token_length': tf.io.FixedLenFeature([], tf.int64),
      'entity_id': tf.io.FixedLenFeature([], tf.int64)
    }

    def transform_to_path(x):
      return tf.strings.join([lm_embedding_dir + '/', tf.strings.as_string(x), '.tfexample'])

    def read_file(file_path):
      results = tf.io.read_file(file_path)
      results = tf.io.decode_compressed(results, compression_type='ZLIB')
      return results

    def parse_batch_example(b_concept_ids):
      bsize = tf.shape(b_concept_ids)[0]

      b_paths = transform_to_path(b_concept_ids)

      b_concepts = tf.map_fn(
        fn=read_file,
        elems=b_paths,
        dtype=tf.string,
        parallel_iterations=self.num_workers
      )
      b_concept_exs = tf.io.parse_example(b_concepts, features=features)
      b_concept_lengths = tf.reshape(
        b_concept_exs['token_length'],
        shape=[bsize]
      )
      b_max_token_length = tf.cast(tf.reduce_max(b_concept_lengths), tf.int32)
      b_concept_embs = tf.reshape(
        tf.sparse_tensor_to_dense(
          b_concept_exs['lm_embedding'],
          default_value=0
        ),
        shape=[bsize, b_max_token_length, self.lm_encoder_size]
      )

      return b_concept_embs, b_concept_lengths, b_concept_ids

    dataset = dataset.batch(
      batch_size=self.batch_size
    )
    dataset = dataset.map(
      map_func=parse_batch_example,
      num_parallel_calls=self.num_workers
    )

    dataset = dataset.prefetch(
      buffer_size=self.buffer_size
    )

    iterator = dataset.make_initializable_iterator()

    self.rel_iterator = iterator
    batch = iterator.get_next()
    b_concept_embs, b_concept_lengths, b_concept_ids = batch

    b_concept_embs.set_shape([None, None, self.lm_encoder_size])

    self.b_rel_embs = b_concept_embs
    self.b_rel_lengths = b_concept_lengths
    self.b_rel_ids = b_concept_ids




class TfEvalDataGenerator:
  def __init__(self, data_dir, batch_size, num_workers, buffer_size):
    self.data_dir = data_dir
    self.batch_size = batch_size
    self.num_workers = num_workers
    self.buffer_size = buffer_size

  def load_eval(self):
    cui2id, train_data, _, _ = data_util.load_metathesaurus_data(self.data_dir, 0.0)
    test_data = data_util.load_metathesaurus_test_data(self.data_dir)

    self.concepts = set()
    self.sr2o = defaultdict(set)
    self.or2s = defaultdict(set)
    for s, r, o in zip(train_data['subj'], train_data['rel'], train_data['obj']):
      self.concepts.add(s)
      self.concepts.add(o)
      self.sr2o[(s, r)].add(o)
      self.or2s[(o, r)].add(s)

    self.test_sr2o = defaultdict(set)
    self.test_or2s = defaultdict(set)
    for s, r, o in zip(test_data['subj'], test_data['rel'], test_data['obj']):
      self.concepts.add(s)
      self.concepts.add(o)
      self.test_sr2o[(s, r)].add(o)
      self.test_or2s[(o, r)].add(s)
      self.sr2o[(s, r)].add(o)
      self.or2s[(o, r)].add(s)

    self.concepts = np.asarray(list(self.concepts), dtype=np.int32)
    self.nrof_sr = len(self.test_sr2o)
    self.nrof_or = len(self.test_or2s)

    self.nrof_triples = len(test_data['subj'])
    self.test_data = test_data

    self.subjs = tf.placeholder(tf.int32, [None])
    self.rels = tf.placeholder(tf.int32, [None])
    self.objs = tf.placeholder(tf.int32, [None])

    self.all_concepts = tf.constant(self.concepts, dtype=tf.int32)

  def load_sub_rel_eval(self, session):
    subj_rels = np.array([(s, r) for (s, r) in self.test_sr2o.keys()], dtype=np.int32)
    session.run(
      self.eval_sr_iterator.initializer,
      feed_dict={
        self.subjs: subj_rels[:, 0],
        self.rels: subj_rels[:, 1],
      }
    )

  def load_obj_rel_eval(self, session):
    obj_rels = np.array([(o, r) for (o, r) in self.test_or2s.keys()], dtype=np.int32)
    session.run(
      self.eval_or_iterator.initializer,
      feed_dict={
        self.objs: obj_rels[:, 0],
        self.rels: obj_rels[:, 1],
      }
    )

  def create_sub_rel_eval_iterator(self):

    dataset = tf.data.Dataset.from_tensor_slices((self.subjs, self.rels))

    dataset = dataset.batch(
      batch_size=self.batch_size
    )

    dataset = dataset.prefetch(
      buffer_size=self.buffer_size
    )

    iterator = dataset.make_initializable_iterator()

    self.eval_sr_iterator = iterator
    batch = iterator.get_next()
    subjs, rels = batch

    self.b_sr_subjs = subjs
    self.b_sr_rels = rels

  def create_obj_rel_eval_iterator(self):

    dataset = tf.data.Dataset.from_tensor_slices((self.objs, self.rels))

    dataset = dataset.batch(
      batch_size=self.batch_size
    )

    dataset = dataset.prefetch(
      buffer_size=self.buffer_size
    )

    iterator = dataset.make_initializable_iterator()

    self.eval_or_iterator = iterator
    batch = iterator.get_next()
    objs, rels = batch

    self.b_or_objs = objs
    self.b_or_rels = rels
