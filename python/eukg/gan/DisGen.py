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
    self.shared_encoder = config.shared_encoder

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

    g_e_neg_subj, g_e_neg_obj, g_e_pos_subj, g_e_pos_obj = self._un_flatten_gen(self.g_e_concepts)

    self.nsamples = tf.shape(g_e_neg_subj)[1]
    uniform_sampls = tf.random.uniform([self.bsize, 1], maxval=tf.cast(self.nsamples, tf.int64), dtype=tf.int64)
    d_e_neg_subj, d_e_neg_obj, d_e_pos_subj, d_e_pos_obj = self._un_flatten_dis(self.d_e_concepts, uniform_sampls)

    self.concept_embeddings = self.d_e_concepts
    self.relation_embeddings = self.d_e_rels
    # [batch_size, num_samples]
    with tf.variable_scope("gen_energies"):
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

    with tf.variable_scope("gen_losses"):
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
      # TODO use already computed embeddings here
      reg = self.regulatization_parameter * self.gen_embedding_model.regularization(
        [self.g_e_concepts],
        [self.g_e_rels]
      )
      self.g_loss += reg
      summary += [
        tf.summary.scalar('gen_reg', reg)
      ]
    with tf.variable_scope('dis_energies'):
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

    with tf.variable_scope("dis_losses"):
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

    summary += [
      tf.summary.scalar('dis_loss', self.d_loss),
      tf.summary.scalar('dis_avg_margin', self.d_avg_pos_energy - self.d_avg_neg_energy),
      tf.summary.scalar('dis_margin', tf.reduce_mean(self.d_margin)),
      tf.summary.scalar('dis_uniform_accuracy', self.d_accuracy),
      tf.summary.scalar('dis_active_percent', self.d_active_percent)
    ]

    self.loss = self.g_loss + self.d_loss
    summary += [
      tf.summary.scalar('loss', self.loss)
    ]

    # backprop
    optimizer = self.optimizer()
    self.train_op = optimizer.minimize(self.loss, tf.train.get_or_create_global_step())

    # summary
    self.summary = tf.summary.merge(summary)

  def _build_embeddings(self):
    batch = self.data_generator.create_iterator()

    self.subjs_emb = batch['b_subj_emb']
    self.rels_emb = batch['b_rels_emb']
    self.objs_emb = batch['b_objs_emb']

    self.subjs_lengths = batch['b_subj_lengths']
    self.rels_lengths = batch['b_rels_lengths']
    self.objs_lengths = batch['b_objs_lengths']

    self.bsize = tf.shape(self.subjs_emb)[0]
    self.seq_len = tf.shape(self.subjs_emb)[1]
    concept_tensors = [self.subjs_emb, self.objs_emb]
    concept_length_tensors = [self.subjs_lengths, self.objs_lengths]
    concept_embs = tf.concat(
      concept_tensors,
      axis=0,
      name='concept_flat_embs'
    )
    concept_lengths = tf.concat(
      concept_length_tensors,
      axis=0,
      name='concept_flat_lengths'
    )

    if self.shared_encoder:
      concept_encodes = self.ace_model.encode(concept_embs, concept_lengths, 'concept')
      rel_encodes = self.ace_model.encode(self.rels_emb, self.rels_lengths, 'rel')
      g_concept_encodes = concept_encodes
      d_concept_encodes = concept_encodes
      g_rel_encodes = rel_encodes
      d_rel_encodes = rel_encodes
    else:
      g_concept_encodes = self.ace_model.encode(concept_embs, concept_lengths, 'concept', 'gen')
      d_concept_encodes = self.ace_model.encode(concept_embs, concept_lengths, 'concept', 'dis')
      g_rel_encodes = self.ace_model.encode(self.rels_emb, self.rels_lengths, 'rel', 'gen')
      d_rel_encodes = self.ace_model.encode(self.rels_emb, self.rels_lengths, 'rel', 'dis')

    self.g_e_concepts = self.gen_embedding_model.embed(g_concept_encodes, 'concept')
    if self.gen_embedding_model == self.dis_embedding_model:
      self.d_e_concepts = self.g_e_concepts
    else:
      # TODO can be more efficient, not all concept dis embeddings are needed, only gen sampled and [0] for random.
      self.d_e_concepts = self.dis_embedding_model.embed(d_concept_encodes, 'concept')

    self.g_e_rels = self.gen_embedding_model.embed(g_rel_encodes, 'rel')
    if self.gen_embedding_model == self.dis_embedding_model:
      self.d_e_rels = self.g_e_rels
    else:
      self.d_e_rels = self.dis_embedding_model.embed(d_rel_encodes, 'rel')

  def _un_flatten_gen(self, e_concepts):
    with tf.variable_scope('emb_indexing'):
      s_subj_start = 2 * self.bsize

      # bsize
      e_pos_subj = e_concepts[:self.bsize]
      # bsize
      e_pos_obj = e_concepts[self.bsize:s_subj_start]

      e_neg_subj, e_neg_obj = self._get_neg_samples(e_pos_subj, e_pos_obj)
    return e_neg_subj, e_neg_obj, e_pos_subj, e_pos_obj

  def _un_flatten_dis(self, e_concepts, g_sampls=None):
    if g_sampls is None:
      g_sampls = tf.zeros(shape=[self.bsize, 1], dtype=tf.int64)
    if isinstance(e_concepts, tuple):
      e_concepts, e_concepts_proj = e_concepts
      e_neg_subj, e_neg_obj, e_pos_subj, e_pos_obj = self._un_flatten_gen(e_concepts)
      e_neg_subj_proj, e_neg_obj_proj, e_pos_subj_proj, e_pos_obj_proj = self._un_flatten_gen(e_concepts_proj)
      # only take first negative sample for discriminator loss
      e_neg_subj = tf.gather(e_neg_subj, g_sampls, batch_dims=1, axis=1)[:, 0]
      e_neg_subj_proj = tf.gather(e_neg_subj_proj, g_sampls, batch_dims=1, axis=1)[:, 0]
      e_neg_obj = tf.gather(e_neg_obj, g_sampls, batch_dims=1, axis=1)[:, 0]
      e_neg_obj_proj = tf.gather(e_neg_obj_proj, g_sampls, batch_dims=1, axis=1)[:, 0]

      e_neg_subj = e_neg_subj, e_neg_subj_proj
      e_neg_obj = e_neg_obj, e_neg_obj_proj
      e_pos_subj = e_pos_subj, e_pos_subj_proj
      e_pos_obj = e_pos_obj, e_pos_obj_proj

    else:
      e_neg_subj, e_neg_obj, e_pos_subj, e_pos_obj = self._un_flatten_gen(e_concepts)

      e_neg_subj = tf.gather(e_neg_subj, g_sampls, batch_dims=1, axis=1)[:, 0]
      e_neg_obj = tf.gather(e_neg_obj, g_sampls, batch_dims=1, axis=1)[:, 0]

    return e_neg_subj, e_neg_obj, e_pos_subj, e_pos_obj

  def _get_neg_samples(self, b_subj_emb, b_objs_emb):
    bsize = tf.shape(b_subj_emb)[0]
    emb_size = b_subj_emb.get_shape()[-1]
    # need to create tensor of shape [bsize, bsize - 1] where, for each bsize it is only the remaining indices
    # shape [bsize, bsize, emb_size]
    b_nsubjs_samples_embs = tf.tile(
      tf.expand_dims(b_subj_emb, axis=0),
      [bsize, 1, 1]
    )
    # shape [bsize, bsize, emb_size]
    b_nobjs_samples_embs = tf.tile(
      tf.expand_dims(b_objs_emb, axis=0),
      [bsize, 1, 1]
    )

    # mask out same batch elements
    b_sample_mask = tf.logical_not(tf.eye(bsize, dtype=tf.bool))

    # only dropping the equal element in batch, so keep others for samples
    subj_sample_count = bsize - 1
    obj_sample_count = bsize - 1

    # utilize boolean mask to get embeddings
    # shape [bsize, bsize-1, emb_size]
    b_nsubjs_samples_embs = tf.reshape(
      tf.boolean_mask(b_nsubjs_samples_embs, b_sample_mask),
      shape=[bsize, subj_sample_count, emb_size],
      name='b_nsubjs_samples_embs'
    )

    # shape [bsize, bsize-1, emb_size]
    b_nobjs_samples_embs = tf.reshape(
      tf.boolean_mask(b_nobjs_samples_embs, b_sample_mask),
      shape=[bsize, obj_sample_count, emb_size],
      name='b_nobjs_samples_embs'
    )

    # concat real objs for negative subj samples
    # shape [bsize,
    # concatenate
    # [bsize, subj_sample_count, lm_encoder_size]
    # with
    # [bsize, obj_sample_count, lm_encoder_size]
    # to get
    # [bsize, total_sample_count, lm_encoder_size]
    b_nsubjs_embs = tf.concat(
      [
        b_nsubjs_samples_embs,
        # tile to [bsize, obj_sample_count, emb_size]
        tf.tile(
          # expand to [bsize, 1, emb_size]
          tf.expand_dims(b_subj_emb, axis=1),
          [1, obj_sample_count, 1]
        )
      ],
      axis=1
    )
    b_nobjs_embs = tf.concat(
      [
        tf.tile(
          tf.expand_dims(b_objs_emb, axis=1),
          [1, subj_sample_count, 1]
        ),
        b_nobjs_samples_embs
      ],
      axis=1
    )

    return b_nsubjs_embs, b_nobjs_embs

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
    self.reward_type = config.reward_type
    self.dis_loss_type = config.dis_loss_type
    self.baseline_momentum = config.baseline_momentum

  def build(self):
    summary = []
    optimizer = self.optimizer()

    self._build_embeddings()

    g_e_neg_subj, g_e_neg_obj, g_e_pos_subj, g_e_pos_obj = self._un_flatten_gen(self.g_e_concepts)

    self.nsamples = tf.shape(g_e_neg_subj)[1]
    uniform_sampls = tf.random.uniform([self.bsize, 1], maxval=tf.cast(self.nsamples, tf.int64), dtype=tf.int64)
    d_e_neg_subj_uniform, d_e_neg_obj_uniform, _, _ = self._un_flatten_dis(self.d_e_concepts, uniform_sampls)

    self.concept_embeddings = self.d_e_concepts
    self.relation_embeddings = self.d_e_rels

    # [batch_size, num_samples]
    with tf.variable_scope("gen_energies"):
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
    d_e_neg_subj, d_e_neg_obj, d_e_pos_subj, d_e_pos_obj = self._un_flatten_dis(self.d_e_concepts, self.g_sampls)

    with tf.variable_scope('dis_energies'):
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

    with tf.variable_scope("dis_losses"):
      # loss wants high neg energy and low pos energy
      self.d_margin = self.d_pos_energy - self.d_neg_energy
      self.d_predictions = tf.argmin(
        tf.stack([self.d_pos_energy, self.d_neg_energy], axis=1), axis=1, output_type=tf.int32)
      self.d_predictions_uniform = tf.argmin(
        tf.stack([self.d_pos_energy, self.neg_energy_uniform], axis=1), axis=1, output_type=tf.int32)
      # TODO double check this is correct with REINFORCE
      # TODO also double check this shouldn't be negative here
      #
      if self.reward_type == 'neg_energy':
        self.d_reward = tf.identity(-self.d_neg_energy, name='reward')
      elif self.reward_type == 'neg_margin':
        self.d_reward = tf.identity(self.d_margin, name='reward')
      else:
        raise ValueError(f'Unknown reward type: {self.reward_type}')

      # loss
      if self.dis_loss_type == 'gen':
        self.d_loss = tf.reduce_mean(tf.nn.relu(self.gamma + self.d_margin), name='d_loss')
      elif self.dis_loss_type == 'gen_and_uniform':
        d_gen_loss = tf.reduce_mean(tf.nn.relu(self.gamma + self.d_margin), name='d_gen_loss')
        d_uniform_margin = self.d_pos_energy - self.d_neg_energy_uniform
        d_uniform_loss = tf.reduce_mean(tf.nn.relu(self.gamma + d_uniform_margin), name='d_uniform_loss')
        self.d_loss = tf.identity(d_gen_loss + d_uniform_loss, name='d_loss')
      # TODO better negative sampling loss? Used in RotatE paper.
      else:
        raise ValueError(f'Unknown dis loss type: {self.dis_loss_type}')

      self.d_accuracy = tf.reduce_mean(tf.to_float(tf.equal(self.d_predictions, 0)))
      self.d_accuracy_uniform = tf.reduce_mean(tf.to_float(tf.equal(self.d_predictions_uniform, 0)))
      # self.d_train_op = d_optimizer.minimize(self.d_loss, name='d_train_op')
      self.d_active_percent = tf.reduce_mean(tf.to_float(-self.d_margin < self.gamma))

    summary += [
      tf.summary.scalar('dis_loss', self.d_loss),
      tf.summary.scalar('dis_avg_margin', self.d_avg_pos_energy - self.d_avg_neg_energy),
      tf.summary.scalar('dis_margin', tf.reduce_mean(self.d_margin)),
      tf.summary.scalar('dis_uniform_accuracy', self.d_accuracy_uniform),
      tf.summary.scalar('dis_gen_accuracy', self.d_accuracy),
      tf.summary.scalar('dis_active_percent', self.d_active_percent)
    ]

    with tf.variable_scope("gen_losses"):
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
      avg_g_prob_loss = tf.reduce_mean(g_loss)
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

    with tf.variable_scope('gen_baselines'):
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
      tf.summary.scalar('gen_avg_prob_loss', avg_g_prob_loss),
      tf.summary.scalar('gen_loss', self.g_loss),
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


class DisSelfGen(DisGen):
  def __init__(self, config, dis_embedding_model, data_generator, ace_model):
    super().__init__(config, dis_embedding_model, dis_embedding_model, data_generator, ace_model)
    self.dis_loss_type = config.dis_loss_type
    self.adversarial_temp = config.adversarial_temp

  def build(self):
    summary = []
    optimizer = self.optimizer()

    self._build_embeddings()

    d_e_neg_subj, d_e_neg_obj, d_e_pos_subj, d_e_pos_obj = self._un_flatten_gen(self.d_e_concepts)

    self.nsamples = tf.shape(d_e_neg_subj)[1]
    uniform_sampls = tf.random.uniform([self.bsize, 1], maxval=tf.cast(self.nsamples, tf.int64), dtype=tf.int64)
    d_e_neg_subj_uniform, d_e_neg_obj_uniform, _, _ = self._un_flatten_dis(self.d_e_concepts, uniform_sampls)

    self.concept_embeddings = self.d_e_concepts
    self.relation_embeddings = self.d_e_rels

    with tf.variable_scope('dis_energies'):
      self.d_pos_energy = self.dis_embedding_model.energy(
        d_e_pos_subj,
        self.d_e_rels,
        d_e_pos_obj
      )

      self.d_neg_energy = self.dis_embedding_model.energy(
        d_e_neg_subj,
        tf.expand_dims(self.d_e_rels, axis=1),
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

    with tf.variable_scope("dis_losses"):
      self.g_probabilities = tf.nn.softmax(self.adversarial_temp * (self.gamma - self.d_neg_energy), axis=-1)
      self.d_loss = -tf.log_sigmoid(self.gamma - self.d_pos_energy)
      self.g_loss = -tf.reduce_sum(self.g_probabilities * tf.log_sigmoid(self.d_neg_energy - self.gamma), axis=-1)
      self.loss = tf.reduce_mean(self.d_loss + self.g_loss)

      # loss wants high neg energy and low pos energy
      self.d_predictions_uniform = tf.argmin(
        tf.stack([self.d_pos_energy, self.neg_energy_uniform], axis=1), axis=1, output_type=tf.int32)

      expected_accuracy = tf.reduce_sum(
        self.g_probabilities * tf.to_float(tf.expand_dims(self.d_pos_energy, axis=-1) < self.d_neg_energy),
        axis=-1
      )
      self.d_accuracy = tf.reduce_mean(expected_accuracy)
      self.d_accuracy_uniform = tf.reduce_mean(tf.to_float(tf.equal(self.d_predictions_uniform, 0)))

    summary += [
      tf.summary.scalar('dis_loss', self.d_loss),
      tf.summary.scalar('dis_avg_margin', self.d_avg_pos_energy - self.d_avg_neg_energy),
      tf.summary.scalar('dis_accuracy', self.d_accuracy),
      tf.summary.scalar('dis_uniform_accuracy', self.d_accuracy_uniform),
      tf.summary.scalar('gen_loss', self.g_loss),
      tf.summary.scalar('loss', self.loss),
    ]

    self.train_op = optimizer.minimize(
      self.loss,
      global_step=tf.train.get_or_create_global_step(),
      name='shared_train_op'
    )

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



