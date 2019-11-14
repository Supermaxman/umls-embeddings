
import tensorflow as tf
import numpy as np
from bert import modeling


class ACEModel(object):
  def __init__(self, config, tokens_dict):
    self.embedding_size = config.embedding_size
    self.embedding_device = config.embedding_device
    self.vocab_size = config.vocab_size
    self.bert_config = modeling.BertConfig.from_json_file(config.bert_config)
    self.bert_rnn_layers = config.encoder_rnn_layers
    self.bert_rnn_size = config.encoder_rnn_size
    self.train_bert = config.train_bert
    with tf.variable_scope('ace_encoder') as scope:
      self.scope = scope
    with tf.variable_scope('bert', reuse=tf.AUTO_REUSE) as bert_scope:
      self.bert_scope = bert_scope

    print('Loading tokens.')
    with tf.variable_scope(self.scope):
      with tf.device("/%s:0" % self.embedding_device):
        max_token_length = tokens_dict['token_ids'].shape[1]
        truncated_lengths = np.clip(tokens_dict['token_lengths'], 1, max_token_length)
        self.token_ids = tf.constant(
          tokens_dict['token_ids'],
          dtype=tf.int64
        )
        self.token_lengths = tf.constant(
          truncated_lengths,
          dtype=tf.int64
        )
    self.tensor_cache = {}

  def tokens_to_embeddings(self, token_ids, token_lengths, emb_type):
    # dynamic token id sizing so we don't waste compute
    max_length = tf.reduce_max(token_lengths)
    token_ids = token_ids[:, :max_length]


    # TODO determine proper reuse of bert, prob keep same weights for both concepts & relations
    with tf.variable_scope(self.bert_scope) as scope:
      input_mask = tf.sequence_mask(
          token_lengths,
          maxlen=tf.shape(token_ids)[1],
          dtype=tf.int32
      )
      bert_model = modeling.BertModel(
        config=self.bert_config,
        is_training=False, # TODO determine if we want to use dropout during training
        input_ids=token_ids,
        input_mask=input_mask,
        scope=scope
      )
      encoder_seq_out = bert_model.get_sequence_output()
      # If we do not want to train BERT at all and leave it frozen then stop gradients at output.
      if not self.train_bert:
        encoder_seq_out = tf.stop_gradient(encoder_seq_out)

    with tf.variable_scope(self.scope):
      # TODO determine proper reuse of rnn layer (reuse between concepts, maybe new for rels?)
      with tf.variable_scope(f'rnn_{emb_type}_encoder', reuse=tf.AUTO_REUSE) as scope:
        encoder_out = rnn_encoder(
          encoder_seq_out,
          token_lengths,
          nrof_layers=self.bert_rnn_layers,
          nrof_units=self.bert_rnn_size,
          reuse=tf.AUTO_REUSE
        )
      return encoder_out

  def embedding_lookup(self, ids, emb_type):
    """
    returns embedding vectors or tuple of embedding vectors for the passed ids
    :param ids: ids of embedding vectors in an embedding matrix
    :param emb_type: type of id embedding (concept, rel).
    :return: embedding vectors or tuple of embedding vectors for the passed ids
    """
    assert emb_type is not None

    # TODO relations with expand dims breaks this cache
    if ids.name not in self.tensor_cache:
      token_ids = tf.nn.embedding_lookup(self.token_ids, ids)
      token_lengths = tf.nn.embedding_lookup(self.token_lengths, ids)
      t_embs = self.tokens_to_embeddings(token_ids, token_lengths, emb_type)
      self.tensor_cache[ids.name] = t_embs
    return self.tensor_cache[ids.name]

  def init_from_checkpoint(self, init_checkpoint):
    t_vars = tf.trainable_variables()
    (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(
      t_vars,
      init_checkpoint
    )
    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    for trainable_var in t_vars:
      init_string = ""
      if trainable_var.name in initialized_variable_names:
        init_string = '*INIT_FROM_CKPT*'
      print(f'{trainable_var.name}: {trainable_var.get_shape()} {init_string}')


def rnn_encoder(input_embs, input_lengths, nrof_layers, nrof_units, reuse=tf.AUTO_REUSE):
  seq_output_indices = input_lengths - 1
  with tf.variable_scope('forward', reuse=reuse) as scope:
    rnn_forward = tf.contrib.cudnn_rnn.CudnnGRU(
      num_layers=nrof_layers,
      num_units=nrof_units
    )
    input_embs_seq_major = tf.transpose(input_embs, [1, 0, 2])
    encoder_forward_seq_major, _ = rnn_forward(input_embs_seq_major, scope=scope)
    encoder_forward = tf.transpose(encoder_forward_seq_major, [1, 0, 2])
    encoder_forward = extract_last_seq_axis(encoder_forward, seq_output_indices)

  with tf.variable_scope('backward', reuse=reuse) as scope:
    rnn_backward = tf.contrib.cudnn_rnn.CudnnGRU(
      num_layers=nrof_layers,
      num_units=nrof_units
    )
    input_embs_rev = tf.reverse_sequence(
      input_embs,
      input_lengths,
      seq_axis=1,
      batch_axis=0)
    input_embs_seq_major_rev = tf.transpose(input_embs_rev, [1, 0, 2])
    encoder_backward_seq_major_rev, _ = rnn_backward(input_embs_seq_major_rev, scope=scope)
    encoder_backward = tf.transpose(encoder_backward_seq_major_rev, [1, 0, 2])
    encoder_backward = extract_last_seq_axis(encoder_backward, seq_output_indices)

  encoder_out = tf.concat([encoder_forward, encoder_backward], axis=1, name='encoder_out')
  return encoder_out


def extract_last_seq_axis(data, ind):
    """
    Get specified elements along the first axis of tensor.
    :param data: Tensorflow tensor that will be subsetted.
    :param ind: Indices to take (one for each element along axis 0 of data).
    :return: Subsetted tensor.
    """

    batch_range = tf.cast(tf.range(tf.shape(data)[0]), tf.int64)
    indices = tf.stack([batch_range, tf.cast(ind, tf.int64)], axis=1)
    res = tf.gather_nd(data, indices)

    return res
