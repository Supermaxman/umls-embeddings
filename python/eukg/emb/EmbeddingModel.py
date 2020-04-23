import tensorflow as tf
import math

from tensorflow.contrib import layers


class BaseModel:
  def __init__(self, config):
    self.embedding_size = config.embedding_size
    self.embedding_device = config.embedding_device
    self.vocab_size = config.vocab_size

    self.embeddings = None
    self.ids_to_update = tf.placeholder(dtype=tf.int32, shape=[None], name='used_vectors')
    self.norm_op = None

  def energy(self, head, rel, tail, norm_ord='euclidean'):
    """
    Calculates the energies of the batch of triples corresponding to head, rel and tail.
    Energy should be in (-inf, 0]
    :param head: [batch_size] vector of head entity ids
    :param rel: [batch_size] vector of relation ids
    :param tail: [batch_size] vector of tail entity ids
    :param norm_ord: order of the normalization function
    :return: [batch_size] vector of energies of the passed triples
    """
    pass
    # raise NotImplementedError("subclass should implement")

  def embed(self, ids, emb_type=None):
    """
    returns embedding vectors or tuple of embedding vectors for the passed ids
    :param ids: ids of embedding vectors in an embedding matrix
    :param emb_type: embedding type of ids
    :return: embedding vectors or tuple of embedding vectors for the passed ids
    """
    pass
    # raise NotImplementedError("subclass should implement")

  def normalize_parameters(self):
    """
    Returns the op that enforces the constraint that each embedding vector have a norm <= 1
    :return: the op that enforces the constraint that each embedding vector have a norm <= 1
    """
    pass
    # raise NotImplementedError("subclass should implement")

  # noinspection PyMethodMayBeStatic
  def normalize(self, rows, mat, ids):
    """
    Normalizes the rows of the matrix mat corresponding to ids s.t. |row|_2 <= 1 for each row.
    :param rows: rows from the matrix mat, embedding vectors
    :param mat: matrix of embedding vectors
    :param ids: ids of rows in mat
    :return: the scatter update op that updates only the rows of mat corresponding to ids
    """
    norm = tf.norm(rows)
    scaling = 1. / tf.maximum(norm, 1.)
    scaled = scaling * rows
    return tf.scatter_update(mat, ids, scaled)


class TransE(BaseModel):
  def __init__(self, config, embeddings_dict=None):
    BaseModel.__init__(self, config)
    with tf.device("/%s:0" % self.embedding_device):
      if embeddings_dict is None:
        self.embeddings = tf.get_variable("embeddings",
                                          shape=[self.vocab_size, self.embedding_size],
                                          dtype=tf.float32,
                                          initializer=layers.xavier_initializer())
      else:
        self.embeddings = tf.Variable(embeddings_dict['embs'], name="embeddings")

  def energy(self, head, rel, tail, norm_ord='euclidean'):
    h = self.embedding_lookup(head)
    r = self.embedding_lookup(rel)
    t = self.embedding_lookup(tail)

    return self.energy_from_embeddings(h, r, t, norm_ord)

  def energy_from_embeddings(self, head, rel, tail, norm_ord='euclidean'):
    return tf.norm(head + rel - tail,
            ord=norm_ord,
            axis=-1,
            keepdims=False,
            name='energy')

  def embedding_lookup(self, ids, emb_type=None):
    with tf.device("/%s:0" % self.embedding_device):
      return tf.nn.embedding_lookup(self.embeddings, ids)

  def normalize_parameters(self):
    """
    Enforces the contraint that the embedding vectors corresponding to ids_to_update <= 1.0
    """
    params1 = self.embedding_lookup(self.ids_to_update)
    self.norm_op = self.normalize(params1, self.embeddings, self.ids_to_update)

    return self.norm_op


class TransD(BaseModel):
  def __init__(self, config, embeddings_dict=None):
    BaseModel.__init__(self, config)
    with tf.device("/%s:0" % self.embedding_device):
      if embeddings_dict is None:
        print('Initializing embeddings.')
        self.embeddings = tf.get_variable("embeddings",
                                          shape=[self.vocab_size, self.embedding_size],  # 364373
                                          dtype=tf.float32,
                                          initializer=layers.xavier_initializer())
      else:
        print('Loading embeddings.')
        self.embeddings = tf.Variable(embeddings_dict['embs'], name="embeddings")
    with tf.device("/%s:%d" % (self.embedding_device, 0 if self.embedding_device == 'cpu' else 0)):
      if embeddings_dict is None or 'p_embs' not in embeddings_dict:
        print('Initializing projection embeddings.')
        if config.p_init == 'zeros':
          p_init = tf.initializers.zeros()
        elif config.p_init == 'xavier':
          p_init = layers.xavier_initializer()
        elif config.p_init == 'uniform':
          p_init = tf.initializers.random_uniform(minval=-0.1, maxval=0.1, dtype=tf.float32)
        else:
          raise Exception('unrecognized p initializer: %s' % config.p_init)

        # projection embeddings initialized to zeros
        self.p_embeddings = tf.get_variable("p_embeddings",
                                            shape=[self.vocab_size, self.embedding_size],
                                            dtype=tf.float32,
                                            initializer=p_init)
      else:
        print('Loading projection embeddings.')
        self.p_embeddings = tf.Variable(embeddings_dict['p_embs'], name="p_embeddings")

  def energy(self, head, rel, tail, norm_ord='euclidean'):
    """
        Computes the TransD energy of a relation triple
        :param head: head concept embedding ids [batch_size]
        :param rel: relation embedding ids [batch_size]
        :param tail: tail concept embedding ids [batch_size]
        :param norm_ord: norm order ['euclidean', 'fro', 'inf', 1, 2, 3, etc.]
        :return: [batch_size] vector of energies
        """
    # x & x_proj both [batch_size, embedding_size]
    h = self.embedding_lookup(head)
    r = self.embedding_lookup(rel)
    t = self.embedding_lookup(tail)

    # [batch_size]
    return self.energy_from_embeddings(h, r, t, norm_ord)

  def energy_from_embeddings(self, head, rel, tail, norm_ord='euclidean'):
    (h, h_proj) = head
    (r, r_proj) = rel
    (t, t_proj) = tail

    h_p = self.project(h, h_proj, r_proj)

    t_p = self.project(t, t_proj, r_proj)
    h_r_t_diff = h_p + r - t_p
    if norm_ord == 2:
      # L2 norm squared
      # https://www.aclweb.org/anthology/P15-1067.pdf
      h_r_t_energy = tf.reduce_sum(
        h_r_t_diff * h_r_t_diff,
        axis=-1,
        keepdims=False,
        name='energy'
      )
    else:
      h_r_t_energy = tf.norm(
        h_r_t_diff,
        ord=norm_ord,
        axis=-1,
        keepdims=False,
        name="energy"
      )

    return h_r_t_energy

  # noinspection PyMethodMayBeStatic
  def project(self, c, c_proj, r_proj):
    """
    Computes the projected concept embedding for relation r according to TransD:
      (c_proj^T*c)*r_proj + c
    :param c: concept embeddings [batch_size, embedding_size]
    :param c_proj: concept projection embeddings [batch_size, embedding_size]
    :param r_proj: relation projection embeddings [batch_size, embedding_size]
    :return: projected concept embedding [batch_size, embedding_size]
    """
    c_p = c + tf.reduce_sum(c * c_proj, axis=-1, keepdims=True) * r_proj
    # https://www.aclweb.org/anthology/P15-1067.pdf
    # normalize projection embeddings
    c_p_norm = tf.norm(c_p, ord=2, axis=-1, keepdims=True)
    c_p = c_p / tf.maximum(c_p_norm, 1.0)
    return c_p

  def embedding_lookup(self, ids, emb_type=None):
    with tf.device("/%s:0" % self.embedding_device):
      params1 = tf.nn.embedding_lookup(self.embeddings, ids)
    with tf.device("/%s:%d" % (self.embedding_device, 1 if self.embedding_device == 'cpu' else 0)):
      params2 = tf.nn.embedding_lookup(self.p_embeddings, ids)
    return params1, params2

  def normalize_parameters(self):
    """
    Normalizes the vectors of embeddings corresponding to the passed ids
    :return: the normalization op
    """
    with tf.device("/%s:0" % self.embedding_device):
      params1 = tf.nn.embedding_lookup(self.embeddings, self.ids_to_update)
    with tf.device("/%s:%d" % (self.embedding_device, 1 if self.embedding_device == 'cpu' else 0)):
      params2 = tf.nn.embedding_lookup(self.p_embeddings, self.ids_to_update)

    n1 = self.normalize(params1, self.embeddings, self.ids_to_update)
    n2 = self.normalize(params2, self.p_embeddings, self.ids_to_update)
    self.norm_op = n1, n2

    return self.norm_op


class DistMult(TransE):
  def __init__(self, config, embeddings_dict=None):
    BaseModel.__init__(self, config)
    if config.energy_activation == 'relu':
      self.energy_activation = tf.nn.relu
    elif config.energy_activation == 'tanh':
      self.energy_activation = tf.nn.tanh
    elif config.energy_activation == 'sigmoid':
      self.energy_activation = tf.nn.sigmoid
    elif config.energy_activation is None:
      self.energy_activation = lambda x: x
    else:
      raise Exception('Unrecognized activation: %s' % config.energy_activation)

    with tf.device("/%s:0" % self.embedding_device):
      if embeddings_dict is None:
        self.embeddings = tf.get_variable("embeddings",
                                          shape=[self.vocab_size, self.embedding_size],
                                          dtype=tf.float32,
                                          initializer=tf.initializers.random_uniform(minval=-0.5,
                                                                                     maxval=0.5,
                                                                                     dtype=tf.float32))
      else:
        self.embeddings = tf.Variable(embeddings_dict['embs'], name="embeddings")

  def energy(self, head, rel, tail, norm_ord='euclidean'):
    h = self.embedding_lookup(head)
    r = self.embedding_lookup(rel)
    t = self.embedding_lookup(tail)

    return self.energy_from_embeddings(h, r, t, norm_ord)

  def energy_from_embeddings(self, head, rel, tail, norm_ord='euclidean'):

    pre_activation = tf.reduce_sum(head * rel * tail, axis=-1)
    post_activation = self.energy_activation(pre_activation)

    return post_activation

  def normalize_parameters(self):
    return tf.no_op()

  def regularization(self, c_parameters, r_parameters):
    reg_term = 0.0
    reg_count = 0
    all_params = c_parameters + r_parameters
    for p in all_params:
      reg_norm = tf.norm(p, axis=-1)
      reg_term += tf.reduce_sum(reg_norm)
      reg_count += tf.size(reg_norm)
    reg_term = reg_term / tf.cast(reg_count, tf.float32)
    return reg_term


class TransDACE(BaseModel):
  def __init__(self, config):
    BaseModel.__init__(self, config)

  def energy(self, head, rel, tail, norm_ord='euclidean'):
    h, h_proj = head
    r, r_proj = rel
    t, t_proj = tail

    h_p = self.project(h, h_proj, r_proj)
    print(f'h_p:{h_p.get_shape()}')
    t_p = self.project(t, t_proj, r_proj)
    print(f't_p:{t_p.get_shape()}')
    h_r_t_diff = h_p + r - t_p
    print(f'h_r_t_diff:{h_r_t_diff.get_shape()}')

    if norm_ord == 2:
      # L2 norm squared
      # https://www.aclweb.org/anthology/P15-1067.pdf
      h_r_t_energy = tf.reduce_sum(
        h_r_t_diff * h_r_t_diff,
        axis=-1,
        keepdims=False,
        name='energy'
      )
    else:
      h_r_t_energy = tf.norm(
        h_r_t_diff,
        ord=norm_ord,
        axis=-1,
        keepdims=False,
        name="energy"
      )

    print(f'h_r_t_energy:{h_r_t_energy.get_shape()}')
    return h_r_t_energy

  # noinspection PyMethodMayBeStatic
  def project(self, c, c_proj, r_proj):
    """
    Computes the projected concept embedding for relation r according to TransD:
      (c_proj^T*c)*r_proj + c
    :param c: concept embeddings [batch_size, embedding_size]
    :param c_proj: concept projection embeddings [batch_size, embedding_size]
    :param r_proj: relation projection embeddings [batch_size, embedding_size]
    :return: projected concept embedding [batch_size, embedding_size]
    """
    c_p = c + tf.reduce_sum(c * c_proj, axis=-1, keepdims=True) * r_proj
    # https://www.aclweb.org/anthology/P15-1067.pdf
    # normalize projection embeddings
    c_p_norm = tf.norm(c_p, ord=2, axis=-1, keepdims=True)
    c_p = c_p / tf.maximum(c_p_norm, 1.0)
    return c_p

  def embed(self, input_encodings, emb_type=None):

    ids_shape = tf.shape(input_encodings)
    feature_size = ids_shape[-1]
    input_shape = ids_shape[:-1]
    input_shape_count = input_shape.get_shape().as_list()[0]
    if input_shape_count > 1:
      total_flat_size = tf.math.reduce_prod(input_shape)
      input_encodings = tf.reshape(
        input_encodings,
        [total_flat_size, feature_size],
        name='input_embeddings_flat'
      )

    with tf.variable_scope('transd_embeddings'):
      with tf.variable_scope(f'{emb_type}_embeddings', reuse=tf.AUTO_REUSE):
        embeddings = tf.layers.dense(
          inputs=input_encodings,
          units=self.embedding_size,
          activation=None,
          name='embeddings'
        )
        embeddings_proj = tf.layers.dense(
          inputs=input_encodings,
          units=self.embedding_size,
          activation=None,
          name='embeddings_proj'
        )
    # https://www.aclweb.org/anthology/P15-1067.pdf
    # normalize all lookups
    embeddings_norm = tf.norm(embeddings, ord=2, axis=-1, keepdims=True)
    embeddings = embeddings / tf.maximum(embeddings_norm, 1.0)

    # embeddings_proj_norm = tf.norm(embeddings_proj, ord=2, axis=-1, keepdims=True)
    # embeddings_proj = embeddings_proj / tf.maximum(embeddings_proj_norm, 1.0)

    # embeddings = tf.nn.l2_normalize(embeddings, axis=-1)
    # embeddings_proj = tf.nn.l2_normalize(embeddings_proj, axis=-1)

    if input_shape_count > 1:
      embeddings = tf.reshape(
        embeddings,
        input_shape + [self.embedding_size],
        'embeddings_reshaped'
      )
      embeddings_proj = tf.reshape(
        embeddings_proj,
        input_shape + [self.embedding_size],
        'embeddings_proj_reshaped'
      )

    return embeddings, embeddings_proj

  def normalize_parameters(self):
    self.norm_op = tf.no_op()
    return self.norm_op


class DistMultACE(BaseModel):
  def __init__(self, config):
    BaseModel.__init__(self, config)
    if config.energy_activation == 'relu':
      self.energy_activation = tf.nn.relu
    elif config.energy_activation == 'tanh':
      self.energy_activation = tf.nn.tanh
    elif config.energy_activation == 'sigmoid':
      self.energy_activation = tf.nn.sigmoid
    elif config.energy_activation is None or config.energy_activation == 'none':
      self.energy_activation = lambda x: x
    else:
      raise Exception('Unrecognized activation: %s' % config.energy_activation)

  def energy(self, head, rel, tail, norm_ord='euclidean'):

    pre_activation = tf.reduce_sum(head * rel * tail, axis=-1)
    post_activation = self.energy_activation(pre_activation)

    return post_activation

  def normalize_parameters(self):
    return tf.no_op()

  def regularization(self, c_parameters, r_parameters):
    reg_term = 0
    reg_count = 0
    for p in c_parameters:
      reg_norm = tf.norm(p, axis=-1)
      reg_term += tf.reduce_sum(reg_norm)
      reg_count += tf.size(reg_norm)
    for p in r_parameters:
      reg_norm = tf.norm(p, axis=-1)
      reg_term += tf.reduce_sum(reg_norm)
      reg_count += tf.size(reg_norm)

    reg_term = reg_term / tf.cast(reg_count, tf.float32)
    return reg_term

  def embed(self, input_encodings, emb_type=None):
    assert emb_type is not None

    ids_shape = tf.shape(input_encodings)
    feature_size = ids_shape[-1]
    input_shape = ids_shape[:-1]
    input_shape_count = input_shape.get_shape().as_list()[0]
    if input_shape_count > 1:
      total_flat_size = tf.math.reduce_prod(input_shape)
      input_encodings = tf.reshape(
        input_encodings,
        [total_flat_size, feature_size],
        name='input_embeddings_flat'
      )

    with tf.variable_scope('dm_embeddings'):
      with tf.variable_scope(f'{emb_type}_embeddings', reuse=tf.AUTO_REUSE):
        embeddings = tf.layers.dense(
          inputs=input_encodings,
          units=self.embedding_size,
          activation=None,
          name='embeddings'
        )

    if input_shape_count > 1:
      embeddings = tf.reshape(
        embeddings,
        input_shape + [self.embedding_size],
        'embeddings_reshaped'
      )

    return embeddings


class RotatEACE(BaseModel):
  def __init__(self, config):
    BaseModel.__init__(self, config)

  def energy(self, head, rel, tail, norm_ord=1):
    h_re, h_im = head
    r, _ = rel
    t_re, t_im = tail

    # TODO limit range of r values somehow
    # r_range = 1.0
    # r = tf.minimum(r, r_range)
    # r = tf.maximum(r, -r_range)
    # r_phase = r / (r_range / math.pi)

    # (-pi, pi) range
    r_phase = tf.tanh(r) * math.pi

    r_re = tf.cos(r_phase)
    r_im = tf.sin(r_phase)

    re_score = (h_re * r_re - h_im * r_im) - t_re
    im_score = (h_re * r_im + h_im * r_re) - t_im
    h_r_t_energy = tf.concat([re_score, im_score], axis=-1)
    h_r_t_energy = tf.norm(
      h_r_t_energy,
      ord=1,
      axis=-1,
      keepdims=False,
      name='energy'
    )

    return h_r_t_energy

  def embed(self, input_encodings, emb_type=None):

    ids_shape = tf.shape(input_encodings)
    feature_size = ids_shape[-1]
    input_shape = ids_shape[:-1]
    input_shape_count = input_shape.get_shape().as_list()[0]
    if input_shape_count > 1:
      total_flat_size = tf.math.reduce_prod(input_shape)
      input_encodings = tf.reshape(
        input_encodings,
        [total_flat_size, feature_size],
        name='input_embeddings_flat'
      )

    with tf.variable_scope('transd_embeddings'):
      with tf.variable_scope(f'{emb_type}_embeddings', reuse=tf.AUTO_REUSE):
        embeddings = tf.layers.dense(
          inputs=input_encodings,
          units=self.embedding_size,
          activation=None,
          name='embeddings'
        )
        embeddings_proj = tf.layers.dense(
          inputs=input_encodings,
          units=self.embedding_size,
          activation=None,
          name='embeddings_proj'
        )

    if input_shape_count > 1:
      embeddings = tf.reshape(
        embeddings,
        input_shape + [self.embedding_size],
        'embeddings_reshaped'
      )
      embeddings_proj = tf.reshape(
        embeddings_proj,
        input_shape + [self.embedding_size],
        'embeddings_proj_reshaped'
      )

    return embeddings, embeddings_proj

  def normalize_parameters(self):
    self.norm_op = tf.no_op()
    return self.norm_op
