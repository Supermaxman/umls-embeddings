
import os
import tensorflow as tf
import numpy as np
import zlib

from . import data_util
from tqdm import tqdm

from collections import defaultdict


class TfDataGenerator:
  def __init__(self, data, train_idx, val_idx, data_dir, secondary_data_dir, num_generator_samples, batch_size,
               num_epochs, lm_encoder_size, num_atom_samples, num_workers, buffer_size, test_mode=False):
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
    self.num_atom_samples = num_atom_samples
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

    concept_features = {
      'lm_embeddings': tf.io.VarLenFeature(tf.float32),
      'token_lengths': tf.io.VarLenFeature(tf.int64),
      'p_atom_idx': tf.io.FixedLenFeature([], tf.int64),
      'nrof_atoms': tf.io.FixedLenFeature([], tf.int64),
      'concept_token_pad': tf.io.FixedLenFeature([], tf.int64),
      'lm_emb_size': tf.io.FixedLenFeature([], tf.int64),
      'entity_id': tf.io.FixedLenFeature([], tf.int64)
    }
    rt_features = {
      'lm_embedding': tf.io.VarLenFeature(tf.float32),
      'token_length': tf.io.FixedLenFeature([], tf.int64),
      'entity_id': tf.io.FixedLenFeature([], tf.int64)
    }

    def transform_to_path(x, entity_type):
      return tf.strings.join([lm_embedding_dir + f'/{entity_type}/', tf.strings.as_string(x), '.tfexample'])

    def read_file(file_path):
      results = tf.io.read_file(file_path)
      results = tf.io.decode_compressed(results, compression_type='ZLIB')
      return results

    # TODO pre-load all rels, cut down on disk reads significantly

    def get_padding_values():
      concept_pad = {
        'lm_embeddings': [None, None, self.lm_encoder_size],
        'token_lengths': [None],
        'p_atom_idx': [],
        'nrof_atoms': [],
        'concept_token_pad': [],
        'lm_emb_size': [],
        'entity_id': []
      }
      rel_pad = {
        'lm_embedding': [None, self.lm_encoder_size],
        'token_length': [],
        'entity_id': []
      }
      return concept_pad, rel_pad, concept_pad

    def parse_single_example(subj, rel, obj):
      rt_ex = tf.io.parse_single_example(
        read_file(
          transform_to_path(rel, 'rt')
        ),
        features=rt_features
      )
      max_token_length = rt_ex['token_length']
      subj_ex = tf.io.parse_single_example(
        read_file(
          transform_to_path(subj, 'concept')
        ),
        features=concept_features
      )
      # [nrof_atoms, concept_token_pad, lm_size]
      subj_ex['lm_embeddings'] = tf.reshape(
        tf.sparse.to_dense(subj_ex['lm_embeddings']),
        [subj_ex['nrof_atoms'], subj_ex['concept_token_pad'], self.lm_encoder_size]
      )

      max_token_length = tf.maximum(subj_ex['concept_token_pad'], max_token_length)
      max_atom_count = subj_ex['nrof_atoms']
      obj_ex = tf.io.parse_single_example(
        read_file(
          transform_to_path(obj, 'concept')
        ),
        features=concept_features
      )
      obj_ex['lm_embeddings'] = tf.reshape(
        tf.sparse.to_dense(obj_ex['lm_embeddings']),
        [obj_ex['nrof_atoms'], obj_ex['concept_token_pad'], self.lm_encoder_size]
      )
      rt_ex['lm_embedding'] = tf.reshape(
        tf.sparse.to_dense(rt_ex['lm_embedding']),
        [rt_ex['token_length'], self.lm_encoder_size]
      )

      max_atom_count = tf.maximum(obj_ex['nrof_atoms'], max_atom_count)
      max_token_length = tf.maximum(obj_ex['concept_token_pad'], max_token_length)
      # [slen lm_size]
      rt_ex['lm_embedding'] = tf.pad(
        rt_ex['lm_embedding'],
        paddings=[
          [0, max_token_length-rt_ex['token_length']],
          [0, 0]
        ]
      )
      # [atoms, slen, lm_size]
      subj_ex['lm_embeddings'] = tf.pad(
        subj_ex['lm_embeddings'],
        paddings=[
          [0, max_atom_count-subj_ex['nrof_atoms']],
          [0, max_token_length-subj_ex['concept_token_pad']],
          [0, 0]
        ]
      )
      subj_ex['token_lengths'] = tf.pad(
        tf.sparse.to_dense(subj_ex['token_lengths']),
        paddings=[
          [0, max_atom_count - subj_ex['nrof_atoms']]
        ]
      )
      obj_ex['lm_embeddings'] = tf.pad(
        obj_ex['lm_embeddings'],
        paddings=[
          [0, max_atom_count - obj_ex['nrof_atoms']],
          [0, max_token_length - obj_ex['concept_token_pad']],
          [0, 0]
        ]
      )
      obj_ex['token_lengths'] = tf.pad(
        tf.sparse.to_dense(obj_ex['token_lengths']),
        paddings=[
          [0, max_atom_count - obj_ex['nrof_atoms']]
        ]
      )
      # everything should be padded to the same shape by here, so only
      # remaining padding needs to be done by tf dataset.padded_batch
      return subj_ex, rt_ex, obj_ex

    def sample_non_primary(embs, lengths, a_counts, p_idxs, k):
      print('sample_non_primary')
      # TODO validate indexing
      p_embs = tf.gather(
        embs,
        tf.expand_dims(p_idxs, axis=-1),
        batch_dims=1,
        axis=1
      )[:, 0]
      print(f'p_embs: {p_embs.get_shape()}')
      # TODO validate indexing
      p_lengths = tf.gather(
        lengths,
        tf.expand_dims(p_idxs, axis=-1),
        batch_dims=1,
        axis=1
      )[:, 0]
      print(f'p_lengths: {p_lengths.get_shape()}')

      # [bsize, num_atoms]
      # will be 1 for primary atom
      p_mask = tf.one_hot(
        p_idxs,
        depth=tf.shape(lengths)[1],
        on_value=0.0,
        off_value=1.0,
        dtype=tf.float32
      )
      print(f'p_mask: {p_mask.get_shape()}')
      # [bsize, num_atoms]
      a_mask = tf.sequence_mask(
        a_counts,
        dtype=tf.float32
      )
      print(f'a_mask: {a_mask.get_shape()}')
      # valid samples for each concept
      # [bsize, num_atoms]
      sample_mask = p_mask * a_mask
      # allow sampling primary atom when concept has no other secondary atoms
      sample_mask = tf.where(
        # [bsize, num_atoms]
        tf.tile(
          tf.expand_dims(tf.greater(tf.reduce_sum(sample_mask, axis=-1), 0.0), axis=-1),
          multiples=[1, tf.shape(sample_mask)[1]]
        ),
        # [bsize, num_atoms]
        sample_mask,
        # [bsize, num_atoms]
        a_mask
      )

      # [bsize, num_atoms]
      sample_energies = (1.0 - sample_mask) * -1e9
      # [bsize, k]
      sample_idxs = tf.random.categorical(sample_energies, k)
      print(f'sample_idxs: {sample_idxs.get_shape()}')
      # TODO validate indexing
      # [bsize, k, seq_len, emb_size]
      s_embs = tf.gather(
        # [bsize, num_atoms, seq_len, emb_size]
        embs,
        # [bsize, k]
        sample_idxs,
        batch_dims=1,
        axis=1
      )
      print(f's_embs: {s_embs.get_shape()}')
      # TODO validate indexing
      # [bsize, k]
      s_lengths = tf.gather(
        # [bsize, num_atoms]
        lengths,
        # [bsize, k]
        sample_idxs,
        batch_dims=1,
        axis=1
      )
      print(f's_lengths: {s_lengths.get_shape()}')

      return p_embs, p_lengths, s_embs, s_lengths

    def parse_batch_example(b_subjs_ex, b_rt_exs, b_objs_ex):
      # [bsize], [bsize], [bsize]
      # I am going to utilize already-loaded other batch elements
      # for each batch's negative samples for efficiency and speed
      b_rels_emb = b_rt_exs['lm_embedding']
      b_rels_lengths = b_rt_exs['token_length']
      # other subjs and objs in batch I can utilize
      bsize = tf.shape(b_subjs_ex['lm_embeddings'])[0]
      b_max_token_length = tf.shape(b_subjs_ex['lm_embeddings'])[1]
      # [bsize, num_atoms, seq_len, emb_size]
      b_subj_emb = b_subjs_ex['lm_embeddings']
      # [bsize, num_atoms]
      b_subj_lengths = b_subjs_ex['token_lengths']
      # [bsize]
      b_subj_atom_counts = b_subjs_ex['nrof_atoms']
      # [bsize]
      b_subj_p_idxs = b_subjs_ex['p_atom_idx']

      # [bsize, seq_len, emb_size], [bsize], [bsize, k, seq_len, emb_size], [bsize, k]
      b_subj_emb, b_subj_lengths, b_s_subj_embs, b_s_subj_lengths = sample_non_primary(
        b_subj_emb,
        b_subj_lengths,
        b_subj_atom_counts,
        b_subj_p_idxs,
        self.num_atom_samples
      )

      # [bsize, num_atoms, seq_len, emb_size]
      b_objs_emb = b_objs_ex['lm_embeddings']
      # [bsize, num_atoms]
      b_objs_lengths = b_objs_ex['token_lengths']
      # [bsize]
      b_obj_atom_counts = b_objs_ex['nrof_atoms']
      # [bsize]
      b_obj_p_idxs = b_objs_ex['p_atom_idx']
      # [bsize, seq_len, emb_size], [bsize], [bsize, k, seq_len, emb_size], [bsize, k]
      b_objs_emb, b_objs_lengths, b_s_objs_embs, b_s_objs_lengths = sample_non_primary(
        b_objs_emb,
        b_objs_lengths,
        b_obj_atom_counts,
        b_obj_p_idxs,
        self.num_atom_samples
      )

      # need to create tensor of shape [bsize, bsize - 1] where, for each bsize it is only the remaining indices
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

      b_subj_emb.set_shape([None, None, self.lm_encoder_size])
      b_s_subj_embs.set_shape([None, self.num_atom_samples, None, self.lm_encoder_size])
      b_nsubjs_embs.set_shape([None, None, None, self.lm_encoder_size])
      b_objs_emb.set_shape([None, None, self.lm_encoder_size])
      b_s_objs_embs.set_shape([None, self.num_atom_samples, None, self.lm_encoder_size])
      b_nobjs_embs.set_shape([None, None, None, self.lm_encoder_size])
      b_rels_emb.set_shape([None, None, self.lm_encoder_size])

      b_data = {
        'b_subj_emb': b_subj_emb,
        'b_subj_lengths': b_subj_lengths,
        'b_s_subj_embs': b_s_subj_embs,
        'b_s_subj_lengths': b_s_subj_lengths,

        'b_nsubjs_embs': b_nsubjs_embs,
        'b_nsubjs_lengths': b_nsubjs_lengths,


        'b_objs_emb': b_objs_emb,
        'b_objs_lengths': b_objs_lengths,
        'b_s_objs_embs': b_s_objs_embs,
        'b_s_objs_lengths': b_s_objs_lengths,

        'b_nobjs_embs': b_nobjs_embs,
        'b_nobjs_lengths': b_nobjs_lengths,

        'b_rels_emb': b_rels_emb,
        'b_rels_lengths': b_rels_lengths
      }

      # TODO convert this to a dict for easier lookup.
      # emb_size, emb_size, emb_size, [sample_size, emb_size], [sample_size, emb_size]
      # return b_subj_emb, b_rels_emb, b_objs_emb, b_nsubjs_embs, b_nobjs_embs, \
      #   b_subj_lengths, b_rels_lengths, b_objs_lengths, b_nsubjs_lengths, b_nobjs_lengths
      return b_data

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

    dataset = dataset.map(
      map_func=parse_single_example,
      num_parallel_calls=self.num_workers
    )
    # dataset = dataset.batch(
    #   batch_size=self.batch_size
    # )
    dataset = dataset.padded_batch(
      batch_size=self.batch_size,
      padded_shapes=get_padding_values()
    )
    dataset = dataset.map(
      map_func=parse_batch_example
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

    batch = iterator.get_next()

    self.subjs_emb = batch['b_subj_emb']
    self.s_subjs_emb = batch['b_s_subj_embs']
    self.nsubjs_embs = batch['b_nsubjs_embs']
    self.objs_emb = batch['b_objs_emb']
    self.s_objs_emb = batch['b_s_objs_emb']
    self.nobjs_embs = batch['b_nobjs_embs']
    self.rels_emb = batch['b_rels_emb']

    self.subjs_lengths = batch['b_subj_lengths']
    self.s_subjs_lengths = batch['b_s_subj_lengths']
    self.nsubjs_lengths = batch['b_nsubjs_lengths']
    self.objs_lengths = batch['b_objs_lengths']
    self.s_objs_lengths = batch['b_s_objs_lengths']
    self.nobjs_lengths = batch['b_nobjs_lengths']
    self.rels_lengths = batch['b_rels_lengths']


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
