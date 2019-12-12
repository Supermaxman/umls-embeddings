

import tensorflow as tf
import numpy as np
from bert import modeling


class LanguageModel(object):
  def __init__(self, train):
    self.train = train

  def encode(self, token_ids, token_lengths):
    pass


class BertLanguageModel(LanguageModel):
  def __init__(self, bert_config_path, train_bert=False):
    super().__init__(train=train_bert)
    self.bert_config = modeling.BertConfig.from_json_file(bert_config_path)

  def encode(self, token_ids, token_lengths):
    with tf.variable_scope('bert', reuse=tf.AUTO_REUSE) as scope:
      input_mask = tf.sequence_mask(
        token_lengths,
        maxlen=tf.shape(token_ids)[1],
        dtype=tf.int32
      )
      bert_model = modeling.BertModel(
        config=self.bert_config,
        is_training=False,
        input_ids=token_ids,
        input_mask=input_mask,
        scope=scope
      )
      encoder_seq_out = bert_model.get_sequence_output()
      # If we do not want to train BERT at all and leave it frozen then stop gradients at output.
      if not self.train:
        encoder_seq_out = tf.stop_gradient(encoder_seq_out)
      return encoder_seq_out

