import tensorflow as tf
import numpy as np
import random
import os
from tqdm import tqdm

from ..data import data_util, DataGenerator
from .. import Config, train
from ..emb import AceModel

from sklearn.svm import LinearSVC
from sklearn import metrics

config = Config.flags


def evaluate():
  random.seed(config.seed)
  np.random.seed(config.seed)
  config.no_semantic_network = True
  all_models_dir = config.model_dir

  cui2id, train_data, _, _ = data_util.load_metathesaurus_data(config.data_dir, config.val_proportion)
  test_data = data_util.load_metathesaurus_test_data(config.data_dir)
  print('Loaded %d test triples from %s' % (len(test_data['rel']), config.data_dir))
  concept_ids = np.unique(np.concatenate([train_data['subj'], train_data['obj'], test_data['subj'], test_data['obj']]))
  print('%d total unique concepts' % len(concept_ids))
  val_idx = np.random.permutation(np.arange(len(train_data['rel'])))[:100000]
  val_data_generator = DataGenerator.DataGenerator(train_data,
                                                   train_idx=val_idx,
                                                   val_idx=[],
                                                   config=config,
                                                   test_mode=True)

  valid_triples = set()
  for s, r, o in zip(train_data['subj'], train_data['rel'], train_data['obj']):
    valid_triples.add((s, r, o))
  for s, r, o in zip(test_data['subj'], test_data['rel'], test_data['obj']):
    valid_triples.add((s, r, o))
  print('%d valid triples' % len(valid_triples))

  model_name = config.run_name

  if config.gpu_memory_growth:
    gpu_config = tf.ConfigProto()
    gpu_config.gpu_options.allow_growth = True
  else:
    gpu_config = None

  with tf.Graph().as_default(), tf.Session(config=gpu_config) as session:
    tf.set_random_seed(config.seed)
    # init model
    # with tf.variable_scope(scope):
    model = train.init_model(config, None, eval=True, pairwise=True)

    tf.global_variables_initializer().run()
    tf.local_variables_initializer().run()

    tf.get_default_graph().finalize()

    scores = []
    labels = []
    for r, s, o, ns, no in tqdm(val_data_generator.generate_mt(True), total=val_data_generator.num_train_batches()):
      pscores, nscores = session.run([model.pos_energy, model.neg_energy], {model.relations: r,
                                                                            model.pos_subj: s,
                                                                            model.pos_obj: o,
                                                                            model.neg_subj: np.expand_dims(ns, axis=-1),
                                                                            model.neg_obj: np.expand_dims(no, axis=-1)})
      nscores = nscores[:, 0]
      scores += pscores.tolist()
      labels += np.ones_like(pscores, dtype=np.int).tolist()
      scores += nscores.tolist()
      labels += np.zeros_like(nscores, dtype=np.int).tolist()
    print('Calculated scores. Training SVM.')
    svm = LinearSVC(dual=False)
    svm.fit(np.asarray(scores).reshape(-1, 1), labels)
    print('Done.')

    data_generator = DataGenerator.DataGenerator(test_data,
                                                 train_idx=np.arange(len(test_data['rel'])),
                                                 val_idx=[],
                                                 config=config,
                                                 test_mode=True)
    data_generator._sampler = val_data_generator.sampler
    scores, labels = [], []
    for r, s, o, ns, no in tqdm(data_generator.generate_mt(True), desc='classifying',
                                total=data_generator.num_train_batches()):
      pscores, nscores = session.run([model.pos_energy, model.neg_energy], {model.relations: r,
                                                                            model.pos_subj: s,
                                                                            model.pos_obj: o,
                                                                            model.neg_subj: np.expand_dims(ns, axis=-1),
                                                                            model.neg_obj: np.expand_dims(no, axis=-1)})
      nscores = nscores[:, 0]
      scores += pscores.tolist()
      labels += np.ones_like(pscores, dtype=np.int).tolist()
      scores += nscores.tolist()
      labels += np.zeros_like(nscores, dtype=np.int).tolist()
    predictions = svm.predict(np.asarray(scores).reshape(-1, 1))
    print('pred: %s' % predictions.shape)
    print('lbl: %d' % len(labels))
    print('Relation Triple Classification Accuracy:  %.4f' % metrics.accuracy_score(labels, predictions))
    print('Relation Triple Classification Precision: %.4f' % metrics.precision_score(labels, predictions))
    print('Relation Triple Classification Recall:    %.4f' % metrics.recall_score(labels, predictions))
    print(metrics.classification_report(labels, predictions))


if __name__ == "__main__":
  evaluate()
