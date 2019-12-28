import tensorflow as tf
import numpy as np
import random
import os
import json
from tqdm import tqdm

from ..data import data_util, TfDataGenerator
from .. import Config, train
from ..emb import AceModel
from ..tf_util import checkpoint_utils

config = Config.flags


def save_embeddings():
  random.seed(config.seed)
  np.random.seed(config.seed)
  config.no_semantic_network = True
  all_models_dir = config.model_dir

  config.model_dir = os.path.join(config.model_dir, config.model, config.run_name)

  cui2id, train_data, _, _ = data_util.load_metathesaurus_data(config.data_dir, config.val_proportion)
  id2cui = {v: k for k, v in cui2id.items()}
  nrof_concepts = max(id2cui.keys()) + 1
  print(f'{nrof_concepts} total unique concepts/rels')

  data_generator = TfDataGenerator.TfTestDataGenerator(
    train_data,
    config.data_dir,
    config.secondary_data_dir,
    config.batch_size,
    config.lm_encoder_size,
    config.num_workers,
    config.buffer_size
  )

  if config.gpu_memory_growth:
    gpu_config = tf.ConfigProto()
    gpu_config.gpu_options.allow_growth = True
  else:
    gpu_config = None

  with tf.Graph().as_default(), tf.Session(config=gpu_config) as session:
    if config.ace_model:
      ace_model = AceModel.ACEModel(config)
    else:
      ace_model = None

    model = train.init_model(config, data_generator, ace_model, test=True)

    if config.pre_run_name is not None:
      pre_model_ckpt = tf.train.latest_checkpoint(
        os.path.join(all_models_dir, config.model, config.pre_run_name))
      checkpoint_utils.init_from_checkpoint(pre_model_ckpt)

    tf.global_variables_initializer().run()
    tf.local_variables_initializer().run()

    # init saver
    tf_saver = tf.train.Saver(max_to_keep=10)

    if config.load:
      ckpt = tf.train.latest_checkpoint(config.model_dir)
      print('Loading checkpoint: %s' % ckpt)
      tf_saver.restore(session, ckpt)

    # finalize graph
    tf.get_default_graph().finalize()

    outdir = os.path.join(config.eval_dir, config.run_name)
    if not os.path.exists(outdir):
      os.makedirs(outdir)

    batch_size = config.batch_size
    emb_size = config.embedding_size
    embeddings = np.zeros((nrof_concepts, emb_size), dtype=np.float32)
    embeddings_proj = np.zeros((nrof_concepts, emb_size), dtype=np.float32)
    seen_concepts = np.zeros(nrof_concepts, dtype=np.bool)
    pbar = tqdm(total=nrof_concepts)
    print('Creating concept embeddings...')
    model.data_generator.load_concepts(session)
    try:
      while True:
        (c_embs, c_embs_proj), c_ids = session.run([model.concept_embeddings, model.concept_ids])
        embeddings[c_ids] = c_embs
        embeddings_proj[c_ids] = c_embs_proj
        pbar.update(len(c_ids))
        seen_concepts[c_ids] = True
    except tf.errors.OutOfRangeError:
      pass

    print('Creating rel embeddings...')
    model.data_generator.load_rels(session)
    try:
      while True:
        (r_embs, r_embs_proj), r_ids = session.run([model.relation_embeddings, model.rel_ids])
        embeddings[r_ids] = r_embs
        embeddings_proj[r_ids] = r_embs_proj
        pbar.update(len(r_ids))
        seen_concepts[r_ids] = True
    except tf.errors.OutOfRangeError:
      pass

    seen_count = int(seen_concepts.astype(np.int32).sum())
    total_count = len(seen_concepts)
    print(f'Seen concepts: {seen_count}/{total_count}')
    assert seen_count == total_count, 'Not all concepts have been seen!'
    print('Saving embeddings...')
    np.savez_compressed(
      os.path.join(outdir, 'test_embeddings.npz'),
      embs=embeddings,
      p_embs=embeddings_proj
    )


if __name__ == "__main__":
  save_embeddings()
