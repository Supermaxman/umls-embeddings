
import tensorflow as tf


class ACEModel(object):
  def __init__(self, config):
    self.encoder_rnn_layers = config.encoder_rnn_layers
    self.encoder_rnn_size = config.encoder_rnn_size
    try:
      self.encoder_rnn_type = config.encoder_rnn_type.lower()
    except AttributeError:
      print('No encoder rnn specified, defaulting to lstm')
      self.encoder_rnn_type = 'lstm'

  def encode(self, encoder_seq_out, token_lengths, emb_type):
    assert emb_type is not None

    max_seq_length = tf.reduce_max(token_lengths)
    # if the max batch length is less than the padded length then we can save computation here
    # by indexing up to the max seq length
    encoder_seq_out = encoder_seq_out[:, :max_seq_length]

    with tf.variable_scope('ace_encoder'):
      with tf.variable_scope(f'rnn_{emb_type}_encoder', reuse=tf.AUTO_REUSE):
        encoder_out = rnn_encoder(
          encoder_seq_out,
          token_lengths,
          nrof_layers=self.encoder_rnn_layers,
          nrof_units=self.encoder_rnn_size,
          rnn_type=self.encoder_rnn_type,
          reuse=tf.AUTO_REUSE
        )
      return encoder_out


def rnn_encoder(input_embs, input_lengths, nrof_layers, nrof_units, rnn_type, reuse=tf.AUTO_REUSE):
  seq_output_indices = input_lengths - 1
  if rnn_type == 'gru':
    cell = tf.contrib.cudnn_rnn.CudnnGRU
  elif rnn_type == 'lstm':
    cell = tf.contrib.cudnn_rnn.CudnnLSTM
  else:
    raise ValueError(f'Unknown rnn type: {rnn_type}')
  with tf.variable_scope('forward', reuse=reuse) as scope:
    rnn_forward = cell(
      num_layers=nrof_layers,
      num_units=nrof_units
    )
    input_embs_seq_major = tf.transpose(input_embs, [1, 0, 2])
    encoder_forward_seq_major, _ = rnn_forward(input_embs_seq_major, scope=scope)
    encoder_forward = tf.transpose(encoder_forward_seq_major, [1, 0, 2])
    encoder_forward = extract_last_seq_axis(encoder_forward, seq_output_indices)

  with tf.variable_scope('backward', reuse=reuse) as scope:
    rnn_backward = cell(
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
