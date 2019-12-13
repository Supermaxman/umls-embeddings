
import os
import tensorflow as tf
import numpy as np

from . import data_util


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

  # must include test data in negative sampler
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
    total_objs = np.unique(np.concatenate([self.data['obj'], test_data['obj']]))
    total_subj = np.unique(np.concatenate([self.data['subj'], test_data['subj']]))
    total_rels = np.unique(np.concatenate([self.data['rel'], test_data['rel']]))
    obj_count = len(total_objs)
    subj_count = len(total_subj)
    rel_count = len(total_rels)
    # TODO make this into initializable system so we can swap train/val data
    data_indices_placeholder = tf.placeholder(tf.int32, [None])
    dataset = tf.data.Dataset.from_tensor_slices(data_indices_placeholder)
    lm_embedding_dir = os.path.join(self.secondary_data_dir, 'lm_embeddings')

    features = {
      'lm_embedding': tf.io.VarLenFeature(tf.float32),
      'lm_embedding_size': tf.io.FixedLenFeature([], tf.int64),
      'token_ids': tf.io.VarLenFeature(tf.int64),
      'token_length': tf.io.FixedLenFeature([], tf.int64),
      'entity_id': tf.io.FixedLenFeature([], tf.int64)
    }

    # TODO make this initializable
    triple_count = len(self.data['subj'])
    subjs = tf.constant(self.data['subj'])
    objs = tf.constant(self.data['obj'])
    rels = tf.constant(self.data['rel'])

    def parse_example(example_idx):
      # go from index to concept id
      b_subj = tf.nn.embedding_lookup(subjs, example_idx)
      b_obj = tf.nn.embedding_lookup(objs, example_idx)
      b_rel = tf.nn.embedding_lookup(rels, example_idx)

      # sample 50/50 subjects and objects
      subj_sample_count = self.num_generator_samples // 2
      obj_sample_count = self.num_generator_samples - subj_sample_count
      # TODO consider validating or not validating these
      b_nsubjs_samples = tf.random.uniform(shape=[subj_sample_count], maxval=subj_count, dtype=tf.int32)
      b_nobjs_samples = tf.random.uniform(shape=[obj_sample_count], maxval=obj_count, dtype=tf.int32)

      def transform_to_path(x):
        return tf.strings.join([lm_embedding_dir, '/', tf.strings.as_string(x), '.tfexample'])

      # convert concept ids to paths and read features
      b_subj_path = transform_to_path(b_subj)
      b_rel_path = transform_to_path(b_rel)
      b_obj_path = transform_to_path(b_obj)
      b_nsubj_samples_paths = transform_to_path(b_nsubjs_samples)
      b_nobj_samples_paths = transform_to_path(b_nobjs_samples)

      b_concepts = []
      b_subj_ex = tf.io.read_file(b_subj_path)
      b_concepts.append(b_subj_ex)
      b_rel_ex = tf.io.read_file(b_rel_path)
      b_concepts.append(b_rel_ex)
      b_obj_ex = tf.io.read_file(b_obj_path)
      b_concepts.append(b_obj_ex)
      b_nsubj_exs = []
      for i in range(subj_sample_count):
        b_nsubj_ex = tf.io.read_file(b_nsubj_samples_paths[i])
        b_nsubj_exs.append(b_nsubj_ex)
        b_concepts.append(b_nsubj_ex)

      b_nobj_exs = []
      for i in range(obj_sample_count):
        b_nobj_ex = tf.io.read_file(b_nobj_samples_paths[i])
        b_nobj_exs.append(b_nobj_ex)
        b_concepts.append(b_nobj_ex)

      b_concept_count = len(b_concepts)
      b_concepts = tf.stack(b_concepts, axis=0)
      b_concept_exs = tf.io.parse_example(b_concepts, features=features)
      b_concept_lengths = b_concept_exs['token_length']
      b_max_token_length = tf.reduce_max(b_concept_lengths)
      b_emb_size = b_concept_exs['lm_embedding_size'][0]
      b_concept_embs = tf.reshape(
        tf.sparse_tensor_to_dense(
          b_concept_exs['lm_embedding'],
          default_value=0
        ),
        shape=[b_concept_count, b_max_token_length, b_emb_size]
      )
      # TODO dynamic batch seq_len padding
      max_seq_len = 31
      # transform [concept_count, max_token_length, emb_size] to
      # [concept_count, max_seq_len, emb_size]
      # and pad with zeros.
      b_s_padding = tf.zeros(shape=[b_concept_count, max_seq_len - b_max_token_length, b_emb_size])
      b_concept_embs = tf.concat([b_concept_embs, b_s_padding], axis=1)

      b_subj_emb = b_concept_embs[0]
      b_subj_lengths = b_concept_lengths[0]
      b_rels_emb = b_concept_embs[1]
      b_rels_lengths = b_concept_lengths[1]
      b_objs_emb = b_concept_embs[2]
      b_objs_lengths = b_concept_lengths[2]

      # TODO need to extract b_concept_lengths
      b_sample_embs = b_concept_embs[3:]
      b_sample_lengths = b_concept_lengths[3:]
      b_nsubjs_samples_embs = b_sample_embs[:subj_sample_count]
      b_nsubjs_sample_lengths = b_sample_lengths[:subj_sample_count]
      b_nobjs_samples_embs = b_sample_embs[subj_sample_count:]
      b_nobjs_samples_lengths = b_sample_lengths[subj_sample_count:]

      b_nsubjs_embs = tf.concat(
        [
          b_nsubjs_samples_embs,
          tf.tile(
            tf.expand_dims(b_subj_emb, axis=0),
            [obj_sample_count, 1, 1]
          )
        ],
        axis=0
      )
      b_nsubjs_lengths = tf.concat(
        [
          b_nsubjs_sample_lengths,
          tf.tile(
            tf.expand_dims(b_subj_lengths, axis=0),
            [obj_sample_count]
          )
        ],
        axis=0
      )
      b_nobjs_embs = tf.concat(
        [
          tf.tile(
            tf.expand_dims(b_objs_emb, axis=0),
            [subj_sample_count, 1, 1]
          ),
          b_nobjs_samples_embs
        ],
        axis=0
      )
      b_nobjs_lengths = tf.concat(
        [
          tf.tile(
            tf.expand_dims(b_objs_lengths, axis=0),
            [subj_sample_count]
          ),
          b_nobjs_samples_lengths
        ],
        axis=0
      )

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
    dataset = dataset.map(
      map_func=parse_example,
      num_parallel_calls=self.num_workers
    )
    dataset = dataset.batch(
      batch_size=self.batch_size
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

    self.data_indices_placeholder = data_indices_placeholder
    self.iterator = iterator

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


