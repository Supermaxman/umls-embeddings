import os
import tensorflow as tf
import numpy as np
import math

from .tf_util import Trainer, ModelSaver

from .emb import EmbeddingModel
from .gan import Generator, train_gan, Discriminator, DisGen
from . import Config
from .data import data_util, DataGenerator, TfDataGenerator
from .emb import AceModel
from .tf_util import checkpoint_utils

import random
import numpy as np


def train():
  config = Config.flags
  seed = config.seed
  random.seed(seed)
  np.random.seed(seed)

  if config.mode == 'gan':
    train_gan.train()
    exit()

  # init model dir
  all_models_dir = config.model_dir
  config.model_dir = os.path.join(config.model_dir, config.model, config.run_name)
  if not os.path.exists(config.model_dir):
    os.makedirs(config.model_dir)

  # init summaries dir
  config.summaries_dir = os.path.join(config.summaries_dir, config.run_name)
  if not os.path.exists(config.summaries_dir):
    os.makedirs(config.summaries_dir)

  # save the config
  data_util.save_config(config.model_dir, config)

  # load data
  cui2id, data, train_idx, val_idx = data_util.load_metathesaurus_data(config.data_dir, config.val_proportion)
  config.val_progress_update_interval = int(math.floor(float(len(val_idx)) / config.val_batch_size))
  config.batches_per_epoch = int(math.floor(float(len(train_idx)) / config.batch_size))
  if not config.no_semantic_network:
    type2cuis = data_util.load_semantic_network_data(config.data_dir, data)
  else:
    type2cuis = None

  # data_generator = DataGenerator.QueuedDataGenerator(
  #   data, train_idx, val_idx, config, type2cuis,
  #   nrof_queued_batches=config.nrof_queued_batches,
  #   nrof_queued_workers=config.nrof_queued_workers
  # )

  data_generator = TfDataGenerator.TfDataGenerator(
    data,
    train_idx,
    val_idx,
    config.data_dir,
    config.secondary_data_dir,
    config.num_generator_samples,
    config.batch_size,
    config.num_epochs,
    config.lm_encoder_size
  )

  # data_generator = DataGenerator.DataGenerator(
  #   data, train_idx, val_idx, config, type2cuis
  # )

  # config map
  config_map = config.flag_values_dict()
  config_map['data'] = data
  config_map['train_idx'] = train_idx
  config_map['val_idx'] = val_idx
  if not config_map['no_semantic_network']:
    config_map['type2cuis'] = type2cuis

  if config.gpu_memory_growth:
    gpu_config = tf.ConfigProto()
    gpu_config.gpu_options.allow_growth = True
  else:
    gpu_config = None

  with tf.Graph().as_default(), tf.Session(config=gpu_config) as session:
    tf.set_random_seed(seed)
    if config.ace_model:
      ace_model = AceModel.ACEModel(config)
    else:
      ace_model = None

    # with tf.variable_scope(config.run_name):
    model = init_model(config, data_generator, ace_model)
    # session.run(model.train_init_op)

    # init models
    # if config.ace_model:
    #   if config.pre_run_name is not None:
    #     pre_model_ckpt = tf.train.latest_checkpoint(
    #       os.path.join(all_models_dir, config.model, config.pre_run_name))
    #     ace_model.init_from_checkpoint(pre_model_ckpt)
    #   else:
    #     ace_model.init_from_checkpoint(config.encoder_checkpoint)
    if config.pre_run_name is not None:
      pre_model_ckpt = tf.train.latest_checkpoint(
        os.path.join(all_models_dir, config.model, config.pre_run_name))
      checkpoint_utils.init_from_checkpoint(pre_model_ckpt)

    tf.global_variables_initializer().run()
    tf.local_variables_initializer().run()

    # if config.ace_model:
    #   ace_model.initialize_tokens(session)

    # init saver
    tf_saver = tf.train.Saver(max_to_keep=10)

    saver = init_saver(config, tf_saver, session)

    # load model
    global_step = 0
    if config.load:
      ckpt = tf.train.latest_checkpoint(config.model_dir)
      print('Loading checkpoint: %s' % ckpt)
      global_step = int(os.path.split(ckpt)[-1].split('-')[-1])
      tf_saver.restore(session, ckpt)

    # finalize graph
    tf.get_default_graph().finalize()

    # define normalization step
    def find_unique(tensor_list):
      if max([len(t.shape) for t in tensor_list[:10]]) == 1:
        return np.unique(np.concatenate(tensor_list[:10]))
      else:
        return np.unique(np.concatenate([t.flatten() for t in tensor_list[:10]]))
    normalize = lambda _, batch: session.run(model.norm_op,
                                             {model.ids_to_update: find_unique(batch)})

    # define streaming_accuracy reset per epoch
    print('local variables that will be reinitialized every epoch: %s' % tf.local_variables())
    reset_local_vars = lambda: session.run(model.reset_streaming_metrics_op)

    # train
    Trainer.train(config_map, session, model, saver,
                  train_post_step=[normalize],
                  train_post_epoch=[reset_local_vars],
                  val_post_epoch=[reset_local_vars],
                  global_step=global_step,
                  max_batches_per_epoch=config_map['max_batches_per_epoch'])


def init_model(config, data_generator, ace_model=None, eval=False, emb_mode=False):
  print('Initializing %s embedding model in %s mode...' % (config.model, config.mode))
  npz = np.load(config.embedding_file) if config.load_embeddings else None

  if not config.ace_model:
    if config.model == 'transe':
      em = EmbeddingModel.TransE(config, embeddings_dict=npz)
    elif config.model == 'transd':
      config.embedding_size = config.embedding_size // 2
      em = EmbeddingModel.TransD(config, embeddings_dict=npz)
    elif config.model == 'distmult':
      em = EmbeddingModel.DistMult(config, embeddings_dict=npz)
    else:
      raise ValueError('Unrecognized model type: %s' % config.model)
  else:
    if config.model == 'transd':
      config.embedding_size = config.embedding_size // 2
      em = EmbeddingModel.TransDACE(config)
    elif config.model == 'distmult':
      em = EmbeddingModel.DistMultACE(config)
    elif config.model == 'transd-distmult':
      g_em = EmbeddingModel.DistMultACE(config)
      config.embedding_size = config.embedding_size // 2
      d_em = EmbeddingModel.TransDACE(config)
      em = d_em, g_em
    else:
      raise ValueError('Unrecognized model type: %s' % config.model)

  if config.mode == 'disc':
    model = Discriminator.BaseModel(config, em, data_generator)
  elif config.mode == 'gen':
    model = Generator.Generator(config, em, data_generator)
  elif config.mode == 'disgen':
    d_em, g_em = em
    model = DisGen.DisGen(config, d_em, g_em, data_generator, ace_model)
  elif config.mode == 'gan-joint':
    d_em, g_em = em
    model = DisGen.DisGenGan(config, d_em, g_em, data_generator, ace_model)
  else:
    raise ValueError('Unrecognized mode: %s' % config.mode)

  if npz:
    # noinspection PyUnresolvedReferences
    npz.close()
  if eval:
    model.build_eval()
    print('Built eval model.')
  elif emb_mode:
    model.build_emb()
    print('Built emb model.')
  else:
    model.build()
    print('Built model.')
  print('use semnet: %s' % model.use_semantic_network)
  return model


def init_saver(config, tf_saver, session):
    model_file = os.path.join(config.model_dir, config.model)
    if config.save_strategy == 'timed':
      print('Models will be saved every %d seconds' % config.save_interval)
      return ModelSaver.TimedSaver(tf_saver, session, model_file, config.save_interval)
    elif config.save_strategy == 'epoch':
      print('Models will be saved every training epoch')
      return ModelSaver.EpochSaver(tf_saver, session, model_file)
    else:
      raise ValueError('Unrecognized save strategy: %s' % config.save_strategy)


def batch(items, batch_size, shuffle=False):
  if shuffle:
    np.random.shuffle(items)
  nrof_batches = int(np.ceil(len(items) / batch_size))
  for i in range(nrof_batches):
    yield items[i * batch_size: (i+1) * batch_size]


if __name__ == "__main__":
  train()
