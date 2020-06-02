import json
import os
from collections import namedtuple
import tensorflow as tf

from .emb import EmbeddingModel
from .emb import AceModel
from .emb import LanguageModel
from .tf_util import checkpoint_utils


def load_ace(ace_path):
  config = load_config(os.path.join(ace_path, 'config.json'))
  bert_checkpoint = config.encoder_checkpoint
  latest_model_checkpoint = tf.train.latest_checkpoint(ace_path)
  lm = LanguageModel.BertWPTModel(
    bert_config_path=config.bert_config,
    train=False
  )
  ace_model = AceModel.ACEModel(config)
  embedding_size = config.embedding_size // 2
  config = config._replace(embedding_size=embedding_size)
  d_em = EmbeddingModel.TransDACE(config)

  def model(t_ids, e_type):
    t_mask = tf.cast(tf.not_equal(t_ids, 0), tf.int32)
    t_lengths = tf.reduce_sum(t_mask, axis=-1)
    t_lm_embs = lm.encode(t_ids, t_lengths)
    checkpoint_utils.init_from_checkpoint(bert_checkpoint)
    b_enc = ace_model.encode(t_lm_embs, t_lengths, e_type)
    b_embs = d_em.embed(b_enc, e_type)
    checkpoint_utils.init_from_checkpoint(latest_model_checkpoint)

    return b_embs

  return model


def load_config(config_path):
  with open(config_path, 'r') as f:
    config_dict = json.load(f)
  config = namedtuple('x', config_dict.keys())(*config_dict.values())
  return config
