
import os
import tensorflow as tf
import numpy as np
import zlib

from . import data_util
from tqdm import tqdm

from collections import defaultdict


class TfDataGenerator:
  def __init__(self, data, train_idx, val_idx, data_dir, secondary_data_dir, num_generator_samples, batch_size,
               num_epochs, lm_encoder_size, num_workers, buffer_size, test_mode=False):
    self.data = data
    self.train_idx = train_idx
    self.val_idx = val_idx
    self.data_dir = data_dir
    self.secondary_data_dir = secondary_data_dir
    self.num_generator_samples = num_generator_samples
    self.batch_size = batch_size
    self.num_epochs = num_epochs

    self.test_mode = test_mode

    self.lm_encoder_size = lm_encoder_size
    self.num_workers = num_workers
    self.buffer_size = buffer_size

  def load_train(self, session):
    np.random.shuffle(self.train_idx)
    session.run(
      self.iterator.initializer,
      feed_dict={
        self.subjs_placeholder: self.data['subj'][self.train_idx],
        self.rels_placeholder: self.data['rel'][self.train_idx],
        self.objs_placeholder: self.data['obj'][self.train_idx]
      }
    )

  def load_val(self, session):
    session.run(
      self.iterator.initializer,
      feed_dict={
        self.subjs_placeholder: self.data['subj'][self.val_idx],
        self.rels_placeholder: self.data['rel'][self.val_idx],
        self.objs_placeholder: self.data['obj'][self.val_idx]
      }
    )

  def create_iterator(self):
    if self.test_mode:
      _, test_data, _, _ = data_util.load_metathesaurus_data(self.data_dir, 0.)
    else:
      test_data = data_util.load_metathesaurus_test_data(self.data_dir)
    # valid_triples = set()
    # for s, r, o in zip(self.data['subj'], self.data['rel'], self.data['obj']):
    #   valid_triples.add((s, r, o))
    # for s, r, o in zip(test_data['subj'], test_data['rel'], test_data['obj']):
    #   valid_triples.add((s, r, o))

    # Get all concepts which have some relation for negative sampling (these should be dense
    total_concepts = np.unique(
      np.concatenate(
        [self.data['obj'], test_data['obj'], self.data['subj'], test_data['subj']]
      )
    )
    concept_count = len(total_concepts)
    concept_lookup = tf.constant(total_concepts)

    self.subjs_placeholder = tf.placeholder(tf.int32, [None])
    self.rels_placeholder = tf.placeholder(tf.int32, [None])
    self.objs_placeholder = tf.placeholder(tf.int32, [None])

    dataset = tf.data.Dataset.from_tensor_slices((self.subjs_placeholder, self.rels_placeholder, self.objs_placeholder))
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

    # def parse_example(b_subj, b_rel, b_obj):
    #
    #   # sample 50/50 subjects and objects
    #   subj_sample_count = self.num_generator_samples // 2
    #   obj_sample_count = self.num_generator_samples - subj_sample_count
    #
    #   # Very low probability that I sample a correct triple (0.03% chance)
    #   b_nsubjs_sample_idxs = tf.random.uniform(shape=[subj_sample_count], maxval=concept_count, dtype=tf.int32)
    #   b_nobjs_sample_idxs = tf.random.uniform(shape=[obj_sample_count], maxval=concept_count, dtype=tf.int32)
    #
    #   # this is necessary because rel ids are shared with concept ids, so we only want to sample concepts, not rels
    #   b_nsubjs_samples = tf.nn.embedding_lookup(concept_lookup, b_nsubjs_sample_idxs)
    #   b_nobjs_samples = tf.nn.embedding_lookup(concept_lookup, b_nobjs_sample_idxs)
    #
    #   # get total number of concepts
    #   b_concept_count = 3 + subj_sample_count + obj_sample_count
    #
    #   # convert concept ids to paths and read features
    #   b_concept_ids = tf.concat([tf.stack([b_subj, b_rel, b_obj], axis=0), b_nsubjs_samples, b_nobjs_samples], axis=0)
    #   b_paths = transform_to_path(b_concept_ids)
    #
    #   b_concepts = tf.map_fn(
    #     fn=read_file,
    #     elems=b_paths,
    #     dtype=tf.string,
    #     parallel_iterations=self.num_workers
    #   )
    #
    #   b_concept_exs = tf.io.parse_example(b_concepts, features=features)
    #   b_concept_lengths = b_concept_exs['token_length']
    #   b_max_token_length = tf.reduce_max(b_concept_lengths)
    #   b_concept_embs = tf.reshape(
    #     tf.sparse_tensor_to_dense(
    #       b_concept_exs['lm_embedding'],
    #       default_value=0
    #     ),
    #     shape=[b_concept_count, b_max_token_length, self.lm_encoder_size]
    #   )
    #   # TODO dynamic batch seq_len padding
    #   # TODO read this from somewhere.
    #   max_seq_len = 31
    #   # transform [concept_count, max_token_length, emb_size] to
    #   # [concept_count, max_seq_len, emb_size]
    #   # and pad with zeros.
    #   b_s_padding = tf.zeros(shape=[b_concept_count, max_seq_len - b_max_token_length, self.lm_encoder_size])
    #   b_concept_embs = tf.concat([b_concept_embs, b_s_padding], axis=1)
    #
    #   b_subj_emb = b_concept_embs[0]
    #   b_subj_lengths = b_concept_lengths[0]
    #   b_rels_emb = b_concept_embs[1]
    #   b_rels_lengths = b_concept_lengths[1]
    #   b_objs_emb = b_concept_embs[2]
    #   b_objs_lengths = b_concept_lengths[2]
    #
    #   b_sample_embs = b_concept_embs[3:]
    #   b_sample_lengths = b_concept_lengths[3:]
    #   b_nsubjs_samples_embs = b_sample_embs[:subj_sample_count]
    #   b_nsubjs_sample_lengths = b_sample_lengths[:subj_sample_count]
    #   b_nobjs_samples_embs = b_sample_embs[subj_sample_count:]
    #   b_nobjs_samples_lengths = b_sample_lengths[subj_sample_count:]
    #
    #   b_nsubjs_embs = tf.concat(
    #     [
    #       b_nsubjs_samples_embs,
    #       tf.tile(
    #         tf.expand_dims(b_subj_emb, axis=0),
    #         [obj_sample_count, 1, 1]
    #       )
    #     ],
    #     axis=0
    #   )
    #   b_nsubjs_lengths = tf.concat(
    #     [
    #       b_nsubjs_sample_lengths,
    #       tf.tile(
    #         tf.expand_dims(b_subj_lengths, axis=0),
    #         [obj_sample_count]
    #       )
    #     ],
    #     axis=0
    #   )
    #   b_nobjs_embs = tf.concat(
    #     [
    #       tf.tile(
    #         tf.expand_dims(b_objs_emb, axis=0),
    #         [subj_sample_count, 1, 1]
    #       ),
    #       b_nobjs_samples_embs
    #     ],
    #     axis=0
    #   )
    #   b_nobjs_lengths = tf.concat(
    #     [
    #       tf.tile(
    #         tf.expand_dims(b_objs_lengths, axis=0),
    #         [subj_sample_count]
    #       ),
    #       b_nobjs_samples_lengths
    #     ],
    #     axis=0
    #   )
    #   # TODO convert this to a dict for easier lookup.
    #   # emb_size, emb_size, emb_size, [sample_size, emb_size], [sample_size, emb_size]
    #   return b_subj_emb, b_rels_emb, b_objs_emb, b_nsubjs_embs, b_nobjs_embs, \
    #     b_subj_lengths, b_rels_lengths, b_objs_lengths, b_nsubjs_lengths, b_nobjs_lengths

    def parse_batch_example(b_subjs, b_rels, b_objs):
      # [bsize], [bsize], [bsize]
      # I am going to utilize already-loaded other batch elements
      # for each batch's negative samples for efficiency and speed

      # other subjs and objs in batch I can utilize
      bsize = tf.shape(b_subjs)[0]

      # get total number of concepts

      # convert concept ids to paths and read features
      # [3 * bsize]
      b_concept_ids = tf.concat([b_subjs, b_rels, b_objs], axis=0)
      b_paths = transform_to_path(b_concept_ids)

      # [3 * bsize]
      b_concepts = tf.map_fn(
        fn=read_file,
        elems=b_paths,
        dtype=tf.string,
        parallel_iterations=self.num_workers
      )
      b_concept_exs = tf.io.parse_example(b_concepts, features=features)
      b_concept_lengths = tf.reshape(
        b_concept_exs['token_length'],
        shape=[3, bsize]
      )
      b_max_token_length = tf.cast(tf.reduce_max(b_concept_lengths), tf.int32)
      b_concept_embs = tf.reshape(
        tf.sparse_tensor_to_dense(
          b_concept_exs['lm_embedding'],
          default_value=0
        ),
        shape=[3, bsize, b_max_token_length, self.lm_encoder_size]
      )
      # shape [bsize, seq_len, emb_size]
      b_subj_emb = b_concept_embs[0]
      b_subj_lengths = b_concept_lengths[0]
      b_rels_emb = b_concept_embs[1]
      b_rels_lengths = b_concept_lengths[1]
      b_objs_emb = b_concept_embs[2]
      b_objs_lengths = b_concept_lengths[2]

      # TODO need to create tensor of shape [bsize, bsize - 1] where, for each bsize it is only the remaining indices
      # shape [bsize, bsize, seq_len, emb_size]
      b_nsubjs_samples_embs = tf.tile(
        tf.expand_dims(b_subj_emb, axis=0),
        [bsize, 1, 1, 1]
      )
      print(b_nsubjs_samples_embs.get_shape())
      # shape [bsize, bsize]
      b_nsubjs_sample_lengths = tf.tile(
        tf.expand_dims(b_subj_lengths, axis=0),
        [bsize, 1]
      )
      print(b_nsubjs_sample_lengths.get_shape())

      # shape [bsize, bsize, seq_len, emb_size]
      b_nobjs_samples_embs = tf.tile(
        tf.expand_dims(b_objs_emb, axis=0),
        [bsize, 1, 1, 1]
      )
      # shape [bsize, bsize]
      b_nobjs_sample_lengths = tf.tile(
        tf.expand_dims(b_objs_lengths, axis=0),
        [bsize, 1]
      )

      # mask out same batch elements
      b_sample_mask = tf.logical_not(tf.eye(bsize, dtype=tf.bool))
      print(b_sample_mask.get_shape())

      # only dropping the equal element in batch, so keep others for samples
      subj_sample_count = bsize - 1
      obj_sample_count = bsize - 1

      # utilize boolean mask to get embeddings
      # shape [bsize, bsize-1, b_max_token_length, lm_encoder_size]
      b_nsubjs_samples_embs = tf.reshape(
        tf.boolean_mask(b_nsubjs_samples_embs, b_sample_mask),
        shape=[bsize, subj_sample_count, b_max_token_length, self.lm_encoder_size]
      )
      print(b_nsubjs_samples_embs.get_shape())
      # shape [bsize, bsize-1]
      b_nsubjs_sample_lengths = tf.reshape(
        tf.boolean_mask(b_nsubjs_sample_lengths, b_sample_mask),
        shape=[bsize, subj_sample_count]
      )
      print(b_nsubjs_sample_lengths.get_shape())

      # shape [bsize, bsize-1, b_max_token_length, lm_encoder_size]
      b_nobjs_samples_embs = tf.reshape(
        tf.boolean_mask(b_nobjs_samples_embs, b_sample_mask),
        shape=[bsize, obj_sample_count, b_max_token_length, self.lm_encoder_size]
      )
      # shape [bsize, bsize-1]
      b_nobjs_sample_lengths = tf.reshape(
        tf.boolean_mask(b_nobjs_sample_lengths, b_sample_mask),
        shape=[bsize, obj_sample_count]
      )

      # concat real objs for negative subj samples
      # shape [bsize,
      # concatenate
      # [bsize, subj_sample_count, b_max_token_length, lm_encoder_size]
      # with
      # [bsize, obj_sample_count, b_max_token_length, lm_encoder_size]
      # to get
      # [bsize, total_sample_count, b_max_token_length, lm_encoder_size]
      b_nsubjs_embs = tf.concat(
        [
          b_nsubjs_samples_embs,
          # tile to [bsize, obj_sample_count, seq_len, emb_size]
          tf.tile(
            # expand to [bsize, 1, seq_len, emb_size]
            tf.expand_dims(b_subj_emb, axis=1),
            [1, obj_sample_count, 1, 1]
          )
        ],
        axis=1
      )
      print(b_nsubjs_embs.get_shape())
      b_nsubjs_lengths = tf.concat(
        [
          b_nsubjs_sample_lengths,
          tf.tile(
            tf.expand_dims(b_subj_lengths, axis=1),
            [1, obj_sample_count]
          )
        ],
        axis=1
      )
      print(b_nsubjs_lengths.get_shape())
      b_nobjs_embs = tf.concat(
        [
          tf.tile(
            tf.expand_dims(b_objs_emb, axis=1),
            [1, subj_sample_count, 1, 1]
          ),
          b_nobjs_samples_embs
        ],
        axis=1
      )
      b_nobjs_lengths = tf.concat(
        [
          tf.tile(
            tf.expand_dims(b_objs_lengths, axis=1),
            [1, subj_sample_count]
          ),
          b_nobjs_sample_lengths
        ],
        axis=1
      )
      # TODO convert this to a dict for easier lookup.
      # emb_size, emb_size, emb_size, [sample_size, emb_size], [sample_size, emb_size]
      return b_subj_emb, b_rels_emb, b_objs_emb, b_nsubjs_embs, b_nobjs_embs, \
        b_subj_lengths, b_rels_lengths, b_objs_lengths, b_nsubjs_lengths, b_nobjs_lengths

    # Shuffling is done every epoch, no need to do it here
    # dataset = dataset.shuffle(
    #   buffer_size=10240
    # )

    # dataset = dataset.apply(
    #   tf.data.experimental.map_and_batch(
    #     parse_example,
    #     self.batch_size,
    #     num_parallel_batches=self.num_workers
    #   )
    # )
    # dataset = dataset.map(
    #   map_func=parse_example,
    #   num_parallel_calls=self.num_workers
    # )
    dataset = dataset.batch(
      batch_size=self.batch_size
    )
    dataset = dataset.map(
      map_func=parse_batch_example,
      num_parallel_calls=self.num_workers
    )
    # dataset = dataset.apply(
    #   tf.data.experimental.prefetch_to_device(
    #     device='gpu:0',
    #     buffer_size=self.buffer_size
    #   )
    # )
    #
    dataset = dataset.prefetch(
      buffer_size=self.buffer_size
    )

    iterator = dataset.make_initializable_iterator()

    self.iterator = iterator
    # TODO convert this to a dict for easier lookup.
    batch = iterator.get_next()
    subjs_emb, rels_emb, objs_emb, nsubjs_embs, nobjs_embs = batch[:5]
    subjs_lengths, rels_lengths, objs_lengths, nsubjs_lengths, nobjs_lengths = batch[5:]

    subjs_emb.set_shape([None, None, self.lm_encoder_size])
    rels_emb.set_shape([None, None, self.lm_encoder_size])
    objs_emb.set_shape([None, None, self.lm_encoder_size])
    nsubjs_embs.set_shape([None, None, None, self.lm_encoder_size])
    nobjs_embs.set_shape([None, None, None, self.lm_encoder_size])

    self.subjs_emb = subjs_emb
    self.rels_emb = rels_emb
    self.objs_emb = objs_emb
    self.nsubjs_embs = nsubjs_embs
    self.nobjs_embs = nobjs_embs

    self.subjs_lengths = subjs_lengths
    self.rels_lengths = rels_lengths
    self.objs_lengths = objs_lengths
    self.nsubjs_lengths = nsubjs_lengths
    self.nobjs_lengths = nobjs_lengths


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

    valid_triples = set()
    for s, r, o in zip(train_data['subj'], train_data['rel'], train_data['obj']):
      valid_triples.add((s, r, o))
    for s, r, o in zip(test_data['subj'], test_data['rel'], test_data['obj']):
      valid_triples.add((s, r, o))

    self.sr2o = defaultdict(set)
    self.or2s = defaultdict(set)
    self.concepts = set()
    for s, r, o in tqdm(valid_triples, desc='building triple maps', total=len(valid_triples)):
      self.sr2o[(s, r)].add(o)
      self.or2s[(o, r)].add(s)
      self.concepts.update([s, o])
    self.nrof_sr = len(self.sr2o)
    self.nrof_or = len(self.or2s)
    self.concepts = np.asarray(list(self.concepts), dtype=np.int32)
    self.nrof_triples = len(test_data['subj'])
    self.test_data = test_data

    self.subjs = tf.placeholder(tf.int32, [None])
    self.rels = tf.placeholder(tf.int32, [None])
    self.objs = tf.placeholder(tf.int32, [None])

    self.all_concepts = tf.constant(self.concepts, dtype=tf.int32)

  def load_sub_rel_eval(self, session):
    subj_rels = np.array([(s, r) for (s, r) in self.sr2o.keys()], dtype=np.int32)
    session.run(
      self.eval_sr_iterator.initializer,
      feed_dict={
        self.subjs: subj_rels[:, 0],
        self.rels: subj_rels[:, 1],
      }
    )

  def load_obj_rel_eval(self, session):
    obj_rels = np.array([(o, r) for (o, r) in self.or2s.keys()], dtype=np.int32)
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
