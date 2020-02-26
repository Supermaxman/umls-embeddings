import tensorflow as tf
import numpy as np

from .Discriminator import BaseModel
from ..emb import Smoothing
from ..data import DataGenerator


class DisGen(BaseModel):
  def __init__(self, config, dis_embedding_model, gen_embedding_model, data_generator, ace_model):
    super().__init__(config, dis_embedding_model, data_generator)
    self.dis_embedding_model = dis_embedding_model
    self.gen_embedding_model = gen_embedding_model
    self.num_samples = config.num_generator_samples
    self.gan_mode = False

    # dataset2: s, r, o, ns, no
    # self.num_samples
    self.neg_subj = tf.placeholder(dtype=tf.int32, shape=[None, None], name="neg_subj")
    self.neg_obj = tf.placeholder(dtype=tf.int32, shape=[None, None], name="neg_obj")
    self.discounted_reward = tf.placeholder(dtype=tf.float32, shape=[], name="discounted_reward")
    self.gan_loss_sample = tf.placeholder(dtype=tf.int32, shape=[None, 2], name="gan_loss_sample")

    self.d_model, self.g_model = self.model.split('-')
    # semantic network vars
    self.type_probabilities = None
    self.data_provider = data_generator
    self.ace_model = ace_model
    self.lm_encoder_size = config.lm_encoder_size

  def build_test(self):
    self.build_test_concepts()
    self.build_test_rels()

  def build_test_concepts(self):

    self.data_generator.create_concept_iterator()

    self.b_concept_embs = self.data_generator.b_concept_embs
    self.b_concept_lengths = self.data_generator.b_concept_lengths
    self.concept_ids = self.data_generator.b_concept_ids

    self.concept_embeddings = self.dis_embedding_model.embed(self.ace_model.encode(self.b_concept_embs, self.b_concept_lengths, 'concept'), 'concept')

  def build_test_rels(self):
    self.data_generator.create_rel_iterator()

    self.b_rel_embs = self.data_generator.b_rel_embs
    self.b_rel_lengths = self.data_generator.b_rel_lengths
    self.rel_ids = self.data_generator.b_rel_ids

    self.relation_embeddings = self.dis_embedding_model.embed(self.ace_model.encode(self.b_rel_embs, self.b_rel_lengths, 'rel'), 'rel')

  def build_eval(self):
    self.data_generator.load_eval()
    self.data_generator.create_sub_rel_eval_iterator()
    self.data_generator.create_obj_rel_eval_iterator()

    self.b_sr_subjs = self.data_generator.b_sr_subjs
    self.b_sr_rels = self.data_generator.b_sr_rels
    self.b_or_rels = self.data_generator.b_or_rels
    self.b_or_objs = self.data_generator.b_or_objs
    self.all_concepts = self.data_generator.all_concepts

    with tf.variable_scope('dis_energy'):

      self.subj_rel_all_energy = self.dis_embedding_model.energy(
        tf.expand_dims(self.b_sr_subjs, axis=1),
        tf.expand_dims(self.b_sr_rels, axis=1),
        tf.expand_dims(self.all_concepts, axis=0),
        norm_ord=self.energy_norm
      )

      self.obj_rel_all_energy = self.dis_embedding_model.energy(
        tf.expand_dims(self.all_concepts, axis=0),
        tf.expand_dims(self.b_or_rels, axis=1),
        tf.expand_dims(self.b_or_objs, axis=1),
        norm_ord=self.energy_norm
      )

  def build_pairwise_eval(self):
    self.pos_energy = self.dis_embedding_model.energy(
      self.pos_subj,
      self.relations,
      self.pos_obj,
      norm_ord=self.energy_norm
    )

    self.neg_energy = self.dis_embedding_model.energy(
      self.neg_subj,
      tf.expand_dims(self.relations, axis=-1),
      self.neg_obj,
      norm_ord=self.energy_norm
    )

  def build(self):

    summary = []

    self._build_embeddings()

    g_e_neg_subj, g_e_neg_obj, g_e_pos_subj, g_e_pos_obj, g_e_s_subj, g_e_s_obj = self._un_flatten_gen(self.g_e_concepts)
    uniform_sampls = tf.random.uniform([self.bsize, 1], maxval=tf.cast(self.nsamples, tf.int64), dtype=tf.int64)
    d_e_neg_subj, d_e_neg_obj, d_e_pos_subj, d_e_pos_obj, d_e_s_subj, d_e_s_obj = self._un_flatten_dis(self.d_e_concepts, uniform_sampls)

    print(f'd_e_neg_subj:{d_e_neg_subj[0].get_shape()}')
    print(f'd_e_neg_obj:{d_e_neg_obj[0].get_shape()}')
    print(f'd_e_pos_subj:{d_e_pos_subj[0].get_shape()}')
    print(f'd_e_pos_obj:{d_e_pos_obj[0].get_shape()}')
    print(f'd_e_s_subj:{d_e_s_subj[0].get_shape()}')
    print(f'd_e_s_obj:{d_e_s_obj[0].get_shape()}')

    self.concept_embeddings = self.d_e_concepts
    self.relation_embeddings = self.d_e_rels
    # [batch_size, num_samples]
    with tf.variable_scope("gen_energy"):
      self.g_sampl_energies = self.gen_embedding_model.energy(
        g_e_neg_subj,
        tf.expand_dims(self.g_e_rels, axis=1),
        g_e_neg_obj
      )
      self.g_true_energies = self.gen_embedding_model.energy(
        g_e_pos_subj,
        self.g_e_rels,
        g_e_pos_obj
      )
      self.g_avg_pos_energy = tf.reduce_mean(self.g_true_energies)
      self.g_avg_neg_energy = tf.reduce_mean(self.g_sampl_energies)

    with tf.variable_scope("gen_loss"):
      # [batch_size]
      sm_numerator = tf.exp(self.g_true_energies)
      # [batch_size]
      exp_sampl_nergies = tf.exp(self.g_sampl_energies)
      sm_denominator = tf.reduce_sum(exp_sampl_nergies, axis=-1) + sm_numerator
      # [batch_size]
      self.g_probabilities = sm_numerator / sm_denominator
      self.g_loss = -tf.reduce_mean(tf.log(self.g_probabilities))

      g_energies = tf.concat(
        [tf.expand_dims(self.g_true_energies, axis=1), self.g_sampl_energies],
        axis=1
      )
      self.g_predictions = tf.argmax(
        g_energies,
        axis=1,
        output_type=tf.int32
      )
      # loss
      self.g_accuracy = tf.reduce_mean(tf.to_float(tf.equal(self.g_predictions, 0)))
      self.g_avg_prob = tf.reduce_mean(self.g_probabilities)

    summary += [
      tf.summary.scalar('gen_loss', self.g_loss),
      tf.summary.scalar('gen_margin', self.g_avg_pos_energy - self.g_avg_neg_energy),
      tf.summary.scalar('gen_accuracy', self.g_accuracy),
      tf.summary.scalar('gen_avg_prob', self.g_avg_prob)
    ]

    # regularization for distmult
    if self.g_model == "distmult":
      reg = self.regulatization_parameter * self.gen_embedding_model.regularization(
        [self.g_e_concepts],
        [self.g_e_rels]
      )
      self.g_loss += reg
      summary += [
        tf.summary.scalar('gen_reg', reg)
      ]
    with tf.variable_scope('dis_energy'):
      print(f'pos_subj[{d_e_pos_subj[0].get_shape()}], '
            f'rels[{self.d_e_rels[0].get_shape()}], '
            f'pos_obj[{d_e_pos_obj[0].get_shape()}]')
      self.d_pos_energy = self.dis_embedding_model.energy(
        d_e_pos_subj,
        self.d_e_rels,
        d_e_pos_obj,
        norm_ord=self.energy_norm
      )
      # input()
      print(f'neg_subj[{d_e_neg_subj[0].get_shape()}], '
            f'rels[{self.d_e_rels[0].get_shape()}], '
            f'neg_obj[{d_e_neg_obj[0].get_shape()}]')
      self.d_neg_energy = self.dis_embedding_model.energy(
        d_e_neg_subj,
        self.d_e_rels,
        d_e_neg_obj,
        norm_ord=self.energy_norm
      )
      # input()
      self.pos_energy = self.d_pos_energy
      self.neg_energy = self.d_neg_energy
      self.d_avg_pos_energy = tf.reduce_mean(self.d_pos_energy)
      self.d_avg_neg_energy = tf.reduce_mean(self.d_neg_energy)

    with tf.variable_scope("dis_loss"):
      self.d_predictions = tf.argmin(
        tf.stack([self.d_pos_energy, self.d_neg_energy], axis=1), axis=1, output_type=tf.int32)
      self.d_reward = tf.reduce_mean(self.d_neg_energy, name='reward')
      # loss
      self.d_margin = self.d_pos_energy - self.d_neg_energy
      print(self.d_margin.get_shape())
      # TODO move constants to config for loss balancing.
      self.d_loss = tf.reduce_mean(tf.nn.relu(self.gamma + self.d_margin), name='loss')
      self.d_active_percent = tf.reduce_mean(tf.to_float(-self.d_margin < self.gamma))

      self.d_accuracy = tf.reduce_mean(tf.to_float(tf.equal(self.d_predictions, 0)))

      if self.np_atom_loss_type == 'p_dist':
        def np_atom_loss(p_emb, np_emb):
          p_emb = tf.expand_dims(tf.stop_gradient(tf.concat(p_emb, axis=-1)), axis=1)
          np_emb = tf.concat(np_emb, axis=-1)
          # [bsize, s_nsample, e_dim]
          p_np_diff = p_emb - np_emb
          # [bsize, s_nsample]
          pd_loss = tf.reduce_mean(p_np_diff * p_np_diff, axis=-1)
          # [bsize]
          pd_loss = tf.reduce_mean(pd_loss, axis=-1)
          pd_loss = tf.reduce_mean(pd_loss, axis=-1)
          return pd_loss
      elif self.np_atom_loss_type == 'p_dist_symmetric':
        def np_atom_loss(p_emb, np_emb):
          p_emb = tf.expand_dims(tf.concat(p_emb, axis=-1), axis=1)
          np_emb = tf.concat(np_emb, axis=-1)
          # [bsize, s_nsample, e_dim]
          p_np_diff = p_emb - np_emb
          # [bsize, s_nsample]
          pd_loss = tf.reduce_mean(p_np_diff * p_np_diff, axis=-1)
          # [bsize]
          pd_loss = tf.reduce_mean(pd_loss, axis=-1)
          pd_loss = tf.reduce_mean(pd_loss, axis=-1)
          return pd_loss
      else:
        raise ValueError(f'Unknown non-primary atom loss type: {self.np_atom_loss_type}')

      p_dist_subj_loss = np_atom_loss(d_e_pos_subj, d_e_s_subj)
      p_dist_obj_loss = np_atom_loss(d_e_pos_obj, d_e_s_obj)
      self.np_loss = self.np_atom_loss_factor * (p_dist_subj_loss + p_dist_obj_loss)

    summary += [
      tf.summary.scalar('dis_loss', self.d_loss),
      tf.summary.scalar('np_loss', self.np_loss),
      tf.summary.scalar('dis_avg_margin', self.d_avg_pos_energy - self.d_avg_neg_energy),
      tf.summary.scalar('dis_margin', tf.reduce_mean(self.d_margin)),
      tf.summary.scalar('dis_accuracy', self.d_accuracy),
      tf.summary.scalar('dis_active_percent', self.d_active_percent)
    ]

    self.loss = self.g_loss + self.d_loss + self.np_loss
    summary += [
      tf.summary.scalar('loss', self.loss)
    ]

    # backprop
    optimizer = self.optimizer()
    self.train_op = optimizer.minimize(self.loss, tf.train.get_or_create_global_step())

    # summary
    self.summary = tf.summary.merge(summary)

  def _build_embeddings(self):
    self.data_generator.create_iterator()

    self.subjs_emb = self.data_generator.subjs_emb
    self.rels_emb = self.data_generator.rels_emb
    self.objs_emb = self.data_generator.objs_emb
    self.nsubjs_embs = self.data_generator.nsubjs_embs
    self.nobjs_embs = self.data_generator.nobjs_embs

    self.s_subjs_emb = self.data_generator.s_subjs_emb
    self.s_objs_emb = self.data_generator.s_objs_emb

    self.subjs_lengths = self.data_generator.subjs_lengths
    self.rels_lengths = self.data_generator.rels_lengths
    self.objs_lengths = self.data_generator.objs_lengths
    self.nsubjs_lengths = self.data_generator.nsubjs_lengths
    self.nobjs_lengths = self.data_generator.nobjs_lengths

    self.s_subjs_lengths = self.data_generator.s_subjs_lengths
    self.s_objs_lengths = self.data_generator.s_objs_lengths

    neg_shape = tf.shape(self.nsubjs_embs)
    self.bsize, self.nsamples, self.seq_len = neg_shape[0], neg_shape[1], neg_shape[2]
    self.total_neg_size = self.bsize * self.nsamples
    self.s_nsamples = tf.shape(self.s_subjs_emb)[1]
    print(f's_subjs_emb: {self.s_subjs_emb.get_shape()}')
    print(f's_objs_emb: {self.s_objs_emb.get_shape()}')
    print(f's_subjs_lengths: {self.s_subjs_lengths.get_shape()}')
    print(f's_objs_lengths: {self.s_objs_lengths.get_shape()}')
    self.total_s_size = self.bsize * self.s_nsamples

    # [bsize * num_samples]
    neg_subj_flat = tf.reshape(
      self.nsubjs_embs,
      [self.total_neg_size, self.seq_len, self.lm_encoder_size], name='neg_subj_flat')

    neg_subj_length_flat = tf.reshape(self.nsubjs_lengths, [self.total_neg_size], name='neg_subj_flat_len')
    # [bsize * num_samples]
    neg_obj_flat = tf.reshape(
      self.nobjs_embs,
      [self.total_neg_size, self.seq_len, self.lm_encoder_size], name='neg_obj_flat')

    neg_obj_length_flat = tf.reshape(self.nobjs_lengths, [self.total_neg_size], name='neg_obj_flat_len')

    s_subjs_flat = tf.reshape(
      self.s_subjs_emb,
      [self.total_s_size, self.seq_len, self.lm_encoder_size], name='s_subjs_flat'
    )
    s_objs_flat = tf.reshape(
      self.s_objs_emb,
      [self.total_s_size, self.seq_len, self.lm_encoder_size], name='s_objs_flat'
    )
    s_subjs_lengths_flat = tf.reshape(self.s_subjs_lengths, [self.total_s_size], name='s_subjs_lengths_flat')
    s_objs_lengths_flat = tf.reshape(self.s_objs_lengths, [self.total_s_size], name='s_objs_lengths_flat')

    # [bsize * num_samples + bsize * num_samples + b_size + b_size + 2 * b_size * num_atom_samples, enc_size]
    concept_embs = tf.concat(
      [neg_subj_flat,
       neg_obj_flat,
       self.subjs_emb,
       self.objs_emb,
       s_subjs_flat,
       s_objs_flat],
      axis=0,
      name='concept_flat_embs'
    )
    concept_lengths = tf.concat(
      [neg_subj_length_flat,
       neg_obj_length_flat,
       self.subjs_lengths,
       self.objs_lengths,
       s_subjs_lengths_flat,
       s_objs_lengths_flat],
      axis=0,
      name='concept_flat_lengths'
    )

    concept_encodes = self.ace_model.encode(concept_embs, concept_lengths, 'concept')
    rel_encodes = self.ace_model.encode(self.rels_emb, self.rels_lengths, 'rel')

    self.g_e_concepts = self.gen_embedding_model.embed(concept_encodes, 'concept')
    # TODO can be more efficient, not all concept dis embeddings are needed, only gen sampled and [0] for random.
    self.d_e_concepts = self.dis_embedding_model.embed(concept_encodes, 'concept')

    self.g_e_rels = self.gen_embedding_model.embed(rel_encodes, 'rel')
    self.d_e_rels = self.dis_embedding_model.embed(rel_encodes, 'rel')

  def _un_flatten_gen(self, e_concepts):
    with tf.variable_scope('emb_indexing'):
      emb_size = e_concepts.get_shape()[-1]

      pos_subj_start = 2 * self.total_neg_size
      s_subj_start = pos_subj_start + 2 * self.bsize

      # first bsize * num_samples
      e_neg_subj = tf.reshape(
        e_concepts[:self.total_neg_size],
        [self.bsize, self.nsamples, emb_size],
        name='e_neg_subj'
      )
      # second bsize * num_samples
      e_neg_obj = tf.reshape(
        e_concepts[self.total_neg_size:2 * self.total_neg_size],
        [self.bsize, self.nsamples, emb_size],
        name='e_neg_obj'
      )
      # bsize
      e_pos_subj = e_concepts[pos_subj_start:pos_subj_start + self.bsize]
      # bsize
      e_pos_obj = e_concepts[pos_subj_start + self.bsize:s_subj_start]

      # first bsize * num_samples
      e_s_subj = tf.reshape(
        e_concepts[s_subj_start:s_subj_start + self.total_s_size],
        [self.bsize, self.s_nsamples, emb_size],
        name='e_s_subj'
      )
      # second bsize * num_samples
      e_s_obj = tf.reshape(
        e_concepts[s_subj_start + self.total_s_size:],
        [self.bsize, self.s_nsamples, emb_size],
        name='e_s_obj'
      )

    return e_neg_subj, e_neg_obj, e_pos_subj, e_pos_obj, e_s_subj, e_s_obj

  def _un_flatten_dis(self, e_concepts, g_sampls=None):
    if g_sampls is None:
      g_sampls = tf.zeros(shape=[self.bsize, 1], dtype=tf.int64)
    if isinstance(e_concepts, tuple):
      e_concepts, e_concepts_proj = e_concepts
      e_neg_subj, e_neg_obj, e_pos_subj, e_pos_obj, e_s_subj, e_s_obj = self._un_flatten_gen(e_concepts)
      e_neg_subj_proj, e_neg_obj_proj, e_pos_subj_proj, e_pos_obj_proj, e_s_subj_proj, e_s_obj_proj = self._un_flatten_gen(e_concepts_proj)
      # only take first negative sample for discriminator loss
      e_neg_subj = tf.gather(e_neg_subj, g_sampls, batch_dims=1, axis=1)[:, 0]
      e_neg_subj_proj = tf.gather(e_neg_subj_proj, g_sampls, batch_dims=1, axis=1)[:, 0]
      e_neg_obj = tf.gather(e_neg_obj, g_sampls, batch_dims=1, axis=1)[:, 0]
      e_neg_obj_proj = tf.gather(e_neg_obj_proj, g_sampls, batch_dims=1, axis=1)[:, 0]

      e_neg_subj = e_neg_subj, e_neg_subj_proj
      e_neg_obj = e_neg_obj, e_neg_obj_proj
      e_pos_subj = e_pos_subj, e_pos_subj_proj
      e_pos_obj = e_pos_obj, e_pos_obj_proj

      e_s_subj = e_s_subj, e_s_subj_proj
      e_s_obj = e_s_obj, e_s_obj_proj

    else:
      e_neg_subj, e_neg_obj, e_pos_subj, e_pos_obj, e_s_subj, e_s_obj = self._un_flatten_gen(e_concepts)
      e_neg_subj = tf.gather(e_neg_subj, g_sampls, batch_dims=1, axis=1)[:, 0]
      e_neg_obj = tf.gather(e_neg_obj, g_sampls, batch_dims=1, axis=1)[:, 0]

    return e_neg_subj, e_neg_obj, e_pos_subj, e_pos_obj, e_s_subj, e_s_obj

  def fetches(self, is_training, verbose=False):
    fetches = [self.summary, self.loss]
    if verbose:
      fetches += [self.d_avg_pos_energy, self.d_avg_neg_energy,
                  self.g_probabilities, self.g_avg_pos_energy, self.g_avg_neg_energy]
    if is_training:
      fetches += [self.train_op]
    return fetches

  def prepare_feed_dict(self, batch, is_training, **kwargs):
    return {}

  def progress_update(self, batch, fetched, **kwargs):
    print('Avg loss of last batch: %.4f' % np.average(fetched[1]))
    print('Avg dis_pos energy of last batch: %.4f' % np.average(fetched[2]))
    print('Avg dis_neg energy of last batch: %.4f' % np.average(fetched[3]))
    print('Avg gen_probability of last batch: %.4f' % np.average(fetched[4]))
    print('Avg gen_pos energy of last batch: %.4f' % np.average(fetched[5]))
    print('Avg gen_neg energy of last batch: %.4f' % np.average(fetched[6]))


class DisGenGan(DisGen):
  def __init__(self, config, dis_embedding_model, gen_embedding_model, data_generator, ace_model):
    super().__init__(config, dis_embedding_model, gen_embedding_model, data_generator, ace_model)
    self.baseline = tf.Variable(
      0.0,
      trainable=False,
      name='baseline'
    )
    self.baseline_type = config.baseline_type
    self.baseline_momentum = config.baseline_momentum

  def build(self):
    summary = []
    optimizer = self.optimizer()

    self._build_embeddings()

    g_e_neg_subj, g_e_neg_obj, g_e_pos_subj, g_e_pos_obj, g_e_s_subj, g_e_s_obj = self._un_flatten_gen(self.g_e_concepts)

    uniform_sampls = tf.random.uniform([self.bsize, 1], maxval=tf.cast(self.nsamples, tf.int64), dtype=tf.int64)
    d_e_neg_subj_uniform, d_e_neg_obj_uniform, _, _, _, _ = self._un_flatten_dis(self.d_e_concepts, uniform_sampls)

    self.concept_embeddings = self.d_e_concepts
    self.relation_embeddings = self.d_e_rels

    # [batch_size, num_samples]
    with tf.variable_scope("gen_energy"):
      self.g_sampl_energies = self.gen_embedding_model.energy(
        g_e_neg_subj,
        tf.expand_dims(self.g_e_rels, axis=1),
        g_e_neg_obj
      )
      self.g_avg_neg_energy = tf.reduce_mean(self.g_sampl_energies)
      # [bsize, 1]
      # index into [bsize, num_samples]
      # TODO double check these sample energies are my logits
      # TODO this needs to be negative because of how the pre-trained model was trained with negative softmax inputs
      # TODO WHYYYY
      self.g_sampls = tf.stop_gradient(
        tf.random.categorical(self.g_sampl_energies, 1, name='g_sampls')
      )

    # This gets the negative subj and obj sampled from the generator along with the positive subj and obj
    d_e_neg_subj, d_e_neg_obj, d_e_pos_subj, d_e_pos_obj, d_e_s_subj, d_e_s_obj = self._un_flatten_dis(self.d_e_concepts, self.g_sampls)

    with tf.variable_scope('dis_energy'):
      self.d_pos_energy = self.dis_embedding_model.energy(
        d_e_pos_subj,
        self.d_e_rels,
        d_e_pos_obj,
        norm_ord=self.energy_norm
      )

      self.d_neg_energy = self.dis_embedding_model.energy(
        d_e_neg_subj,
        self.d_e_rels,
        d_e_neg_obj,
        norm_ord=self.energy_norm
      )

      self.d_neg_energy_uniform = self.dis_embedding_model.energy(
        d_e_neg_subj_uniform,
        self.d_e_rels,
        d_e_neg_obj_uniform,
        norm_ord=self.energy_norm
      )

      self.pos_energy = self.d_pos_energy
      self.neg_energy = self.d_neg_energy
      self.neg_energy_uniform = self.d_neg_energy_uniform
      self.d_avg_pos_energy = tf.reduce_mean(self.d_pos_energy)
      self.d_avg_neg_energy = tf.reduce_mean(self.d_neg_energy)

    with tf.variable_scope("dis_loss"):
      self.d_predictions = tf.argmin(
        tf.stack([self.d_pos_energy, self.d_neg_energy], axis=1), axis=1, output_type=tf.int32)
      self.d_predictions_uniform = tf.argmin(
        tf.stack([self.d_pos_energy, self.neg_energy_uniform], axis=1), axis=1, output_type=tf.int32)
      # TODO double check this is correct with REINFORCE
      # TODO also double check this shouldn't be negative here
      #
      self.d_reward = tf.identity(-self.d_neg_energy, name='reward')
      # loss
      # loss wants high neg energy and low pos energy
      self.d_margin = self.d_pos_energy - self.d_neg_energy
      self.d_loss = tf.reduce_mean(tf.nn.relu(self.gamma + self.d_margin), name='loss')
      self.d_accuracy = tf.reduce_mean(tf.to_float(tf.equal(self.d_predictions, 0)))
      self.d_accuracy_uniform = tf.reduce_mean(tf.to_float(tf.equal(self.d_predictions_uniform, 0)))
      # self.d_train_op = d_optimizer.minimize(self.d_loss, name='d_train_op')
      self.d_active_percent = tf.reduce_mean(tf.to_float(-self.d_margin < self.gamma))

    summary += [
      tf.summary.scalar('dis_loss', self.d_loss),
      tf.summary.scalar('dis_avg_margin', self.d_avg_pos_energy - self.d_avg_neg_energy),
      tf.summary.scalar('dis_margin', tf.reduce_mean(self.d_margin)),
      tf.summary.scalar('dis_accuracy', self.d_accuracy_uniform),
      tf.summary.scalar('dis_gen_accuracy', self.d_accuracy),
      tf.summary.scalar('dis_active_percent', self.d_active_percent)
    ]

    with tf.variable_scope("gen_loss"):
      # TODO determine if this is a good baseline method
      self.discounted_reward = tf.stop_gradient(self.d_reward - self.baseline)
      self.avg_reward = tf.reduce_mean(self.d_reward)
      self.avg_discounted_reward = tf.reduce_mean(self.discounted_reward)
      # [batch_size, num_samples] - this is for sampling during GAN training
      # TODO this needs to be negative because of how the pre-trained model was trained with negative softmax inputs
      # TODO WHYYYY
      self.g_probability_distributions = tf.nn.softmax(self.g_sampl_energies, axis=-1)
      self.g_probabilities = tf.gather(
        self.g_probability_distributions,
        self.g_sampls,
        batch_dims=1,
        axis=1,
        name='sampl_probs'
      )[:, 0]
      # TODO ask if self.discounted_reward should be [bsize] neg energies, then multiplied to each of these
      # TODO losses before sum/avg instead of sum and multiplying by avg neg energy of discriminator.
      # g_loss = -tf.reduce_sum(tf.log(self.g_probabilities))
      # we want to maximize -f(neg) * log(p(neg)) so we minimize -[-f(neg) * log(p(neg))]
      g_loss = -tf.log(self.g_probabilities)
      avg_g_loss = tf.reduce_mean(g_loss)
      self.g_loss = tf.reduce_mean(self.discounted_reward * g_loss)
      self.g_avg_prob = tf.reduce_mean(self.g_probabilities)

      if self.g_model == "distmult":
        reg = self.regulatization_parameter * self.gen_embedding_model.regularization(
          [self.g_e_concepts],
          [self.g_e_rels]
        )
        summary += [
          tf.summary.scalar('gen_reg', reg)
        ]
        self.g_loss += reg

    self.loss = self.d_loss + self.g_loss

    self.train_op = optimizer.minimize(
      self.loss,
      global_step=tf.train.get_or_create_global_step(),
      name='shared_train_op'
    )

    with tf.variable_scope('gen_loss'):
      with tf.control_dependencies([self.train_op]):
        # TODO determine if this is a good baseline method, maybe running mean or something
        # TODO use baseline_type to change to running avg, etc.
        if self.baseline_type == 'avg_prev_batch':
          self.new_baseline = self.avg_reward
        elif self.baseline_type == 'avg_prev_batch_momentum':
          momentum = self.baseline_momentum
          self.new_baseline = ((1.0 - momentum) * self.avg_reward) + (momentum * self.baseline)
        else:
          raise ValueError(f'Unknown baseline type: {self.baseline_type}')
        self.update_baseline_op = tf.assign(
          self.baseline,
          self.new_baseline
        )

    self.train_op = tf.group(
      self.train_op,
      self.update_baseline_op,
      name='train_op'
    )
    summary += [
      tf.summary.scalar('gen_loss', avg_g_loss),
      tf.summary.scalar('gen_avg_sampled_prob', self.g_avg_prob),
      tf.summary.scalar('gen_discounted_reward', self.avg_discounted_reward),
      tf.summary.scalar('gen_reward', self.avg_reward),
      tf.summary.scalar('gen_baseline', self.baseline)
    ]

    # summary
    self.summary = tf.summary.merge(summary)

  def fetches(self, is_training, verbose=False):
    fetches = [self.summary, self.g_loss, self.d_loss]
    if verbose:
      fetches += [self.d_accuracy]
    if is_training:
      fetches += [self.train_op]
    return fetches

  def prepare_feed_dict(self, batch, is_training, **kwargs):
    return {}

  def progress_update(self, batch, fetched, **kwargs):
    print('Avg gen loss of last batch: %.4f' % np.average(fetched[1]))
    print('Avg dis loss of last batch: %.4f' % np.average(fetched[2]))
    print('Avg dis accuracy of last batch: %.4f' % np.average(fetched[3]))



