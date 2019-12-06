import tensorflow as tf
import numpy as np
import random
import os
import json
from tqdm import tqdm

from ..data import data_util, DataGenerator
from .. import Config, train
from ..emb import AceModel

config = Config.flags


def save_embeddings():
  random.seed(config.seed)
  np.random.seed(config.seed)
  config.no_semantic_network = True
  all_models_dir = config.model_dir

  cui2id, train_data, _, _ = data_util.load_metathesaurus_data(config.data_dir, config.val_proportion)
  id2cui = {v: k for k, v in cui2id.items()}
  test_data = data_util.load_metathesaurus_test_data(config.data_dir)
  print('Loaded %d test triples from %s' % (len(test_data['rel']), config.data_dir))
  concept_ids = np.unique(np.concatenate([train_data['subj'], train_data['obj'], test_data['subj'], test_data['obj']]))
  rel_ids = np.unique(train_data['rel'])
  concept_count = len(concept_ids)
  rel_count = len(rel_ids)
  print(f'{concept_count} total unique concepts')
  print(f'{rel_count} total unique relation types')

  model_name = config.run_name

  if config.gpu_memory_growth:
    gpu_config = tf.ConfigProto()
    gpu_config.gpu_options.allow_growth = True
  else:
    gpu_config = None

  with tf.Graph().as_default(), tf.Session(config=gpu_config) as session:
    # init model
    # with tf.variable_scope(scope):
    tf.set_random_seed(config.seed)
    # init model
    # with tf.variable_scope(scope):
    if config.ace_model:
      t_data = data_util.load_metathesaurus_token_data(config.data_dir)
      ace_model = AceModel.ACEModel(config, t_data)
    else:
      ace_model = None
    model = train.init_model(config, None, ace_model, emb_mode=True)

    if config.ace_model and not config.load and config.pre_run_name is not None:
      pre_model_ckpt = tf.train.latest_checkpoint(
        os.path.join(all_models_dir, config.model, config.pre_run_name))
      ace_model.init_from_checkpoint(pre_model_ckpt)

    tf.global_variables_initializer().run()
    tf.local_variables_initializer().run()

    if config.ace_model:
      ace_model.initialize_tokens(session)

    if config.load:
      # init saver
      tf_saver = tf.train.Saver(max_to_keep=10)

      # load model
      ckpt = tf.train.latest_checkpoint(os.path.join(config.model_dir, config.model, model_name))
      print('Loading checkpoint: %s' % ckpt)
      tf_saver.restore(session, ckpt)
    tf.get_default_graph().finalize()

    outdir = os.path.join(config.eval_dir, config.run_name)
    if not os.path.exists(outdir):
      os.makedirs(outdir)

    def batch(items, b_size):
      total_batches = int(np.ceil(len(items) / b_size))
      indices = np.arange(len(items))
      for b in range(total_batches):
        yield items[b * batch_size: (b + 1) * batch_size], indices[b * batch_size: (b + 1) * batch_size]

    batch_size = config.batch_size
    emb_size = config.embedding_size
    concept_embeddings = np.zeros((concept_count, 2, emb_size))
    rel_embeddings = np.zeros((rel_count, 2, emb_size))
    print('Creating concept embeddings...')
    for c, idx in tqdm(batch(concept_ids, batch_size), total=int(np.ceil(concept_count / batch_size))):
      c_embs, c_embs_proj = session.run(model.concept_embeddings, {model.concepts: c})
      concept_embeddings[idx, 0] = c_embs
      concept_embeddings[idx, 1] = c_embs_proj

    print('Creating rel embeddings...')
    for r, idx in tqdm(batch(rel_ids, batch_size), total=int(np.ceil(rel_count / batch_size))):
      r_embs, r_embs_proj = session.run(model.relation_embeddings, {model.relations: r})
      rel_embeddings[idx, 0] = r_embs
      rel_embeddings[idx, 1] = r_embs_proj

    print('Saving embeddings...')
    np.savez_compressed(
      os.path.join(outdir, 'test_embeddings.npz'),
      concept_embeddings=concept_embeddings,
      rel_embeddings=rel_embeddings,
      concept_ids=concept_ids,
      rel_ids=rel_ids
    )


if __name__ == "__main__":
  save_embeddings()
