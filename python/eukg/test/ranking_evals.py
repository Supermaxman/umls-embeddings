import tensorflow as tf
import numpy as np
import random
import math
import os
import json
from tqdm import tqdm
from collections import defaultdict
from multiprocessing import Process, Queue, Pool

from ..data import data_util, TfDataGenerator
from .. import Config, train
from ..threading_util import synchronized, parallel_stream


config = Config.flags


def sort_and_rank_objs(b_subj, b_rel, b_objs, b_obj_energies, b_real_objs, b_valid_objs):
  ranks = []
  obj_ranks = 0
  prev_rank = 1
  total_rank = 1
  prev_score = float('-inf')
  # sort all triples of (s, r, o) where o is every concept.
  # b_real_objs contains all real (s, r, o) test triples for a given (s, r)
  # b_valid_objs contains all real (s, r, o) train and test triples for a given (s, r)
  # sort all objs for subj, rel by their energies, where lower is better.
  for b_obj, b_obj_energy in sorted(zip(b_objs, b_obj_energies), key=lambda x: x[1]):
    # if the obj is a real test triple of (s, r, o) then save its rank
    if b_obj in b_real_objs:
      ranks.append(((b_subj, b_rel, b_obj), prev_rank, b_obj_energy))
      obj_ranks += 1
      # if we have seen all test triples for (s, r) then we break as we do not care about the remaining non-test
      # triple ranks
      if obj_ranks == len(b_real_objs):
        break
    # do not count true (s, r, o) triples which are part of training data in ranking since we are
    # trying to get a rank for each true o in (s, r) vs all negative objs
    elif b_obj not in b_valid_objs:
      total_rank += 1
      # if energy is higher than previous energy then save new ranking, otherwise use same ranking
      # for all same-energy triples.
      if b_obj_energy > prev_score:
        prev_score = b_obj_energy
        prev_rank = total_rank
  return ranks


def sort_and_rank_subjs(b_obj, b_rel, b_subjs, b_subj_energies, b_real_subjs, b_valid_subjs):
  ranks = []
  subj_ranks = 0
  prev_rank = 1
  total_rank = 1
  prev_score = float('-inf')
  # sort all triples of (s, r, o) where s is every concept.
  # b_real_subjs contains all real (s, r, o) test triples for a given (o, r)
  # b_valid_subjs contains all real (s, r, o) train and test triples for a given (o, r)
  # sort all subjs for obj, rel by their energies, where lower is better.
  for b_subj, b_subj_energy in sorted(zip(b_subjs, b_subj_energies), key=lambda x: x[1]):
    if b_subj in b_real_subjs:
      # if the obj is a real test triple of (s, r, o) then save its rank
      ranks.append(((b_subj, b_rel, b_obj), prev_rank, b_subj_energy))
      subj_ranks += 1
      # if we have seen all test triples for (o, r) then we break as we do not care about the remaining non-test
      # triple ranks
      if subj_ranks == len(b_real_subjs):
        break
    # do not count true (s, r, o) triples which are part of data in ranking since we are
    # trying to get a rank for each true s in (o, r) vs all negative subjs
    elif b_subj not in b_valid_subjs:
      total_rank += 1
      # if energy is higher than previous energy then save new ranking, otherwise use same ranking
      # for all same-energy triples.
      if b_subj_energy > prev_score:
        prev_score = b_subj_energy
        prev_rank = total_rank
  return ranks


def save_ranks():
  random.seed(1337)
  np.random.seed(1337)
  # config.batch_size = 4096

  if config.gpu_memory_growth:
    gpu_config = tf.ConfigProto()
    gpu_config.gpu_options.allow_growth = True
  else:
    gpu_config = None

  with tf.Graph().as_default(), tf.Session(config=gpu_config) as session:
    tf.set_random_seed(config.seed)

    #  embedding_file=, load_embeddings=True, transd
    data_generator = TfDataGenerator.TfEvalDataGenerator(
      config.data_dir,
      config.batch_size,
      config.num_workers,
      config.buffer_size
    )
    # init model
    model = train.init_model(config, data_generator, eval=True)

    tf.global_variables_initializer().run()
    tf.local_variables_initializer().run()

    tf.get_default_graph().finalize()

    outdir = os.path.join(config.eval_dir, config.run_name)
    if not os.path.exists(outdir):
      os.makedirs(outdir)

    pool = Pool(processes=config.num_workers)

    print(f'(s, r): {model.data_generator.nrof_sr}')
    print(f'(o, r): {model.data_generator.nrof_or}')

    with open(os.path.join(outdir, 'subj_ranks.json'), 'w+') as f:
      model.data_generator.load_obj_rel_eval(session)
      pbar = tqdm(total=model.data_generator.nrof_or)
      subj_ranks = {}
      try:
        while True:
          obj_rel_energy, b_objs, b_rels = session.run([model.obj_rel_all_energy, model.b_or_objs, model.b_or_rels])
          bsize = len(b_rels)
          b_promises = []
          for b_obj, b_rel, b_subj_energies in zip(b_objs, b_rels, obj_rel_energy):
            b_real_subjs = model.data_generator.test_or2s[(b_obj, b_rel)]
            b_valid_subjs = model.data_generator.or2s[(b_obj, b_rel)]
            b_subjs = model.data_generator.concepts
            promise = pool.apply_async(sort_and_rank_subjs, (b_obj, b_rel, b_subjs, b_subj_energies, b_real_subjs, b_valid_subjs))
            b_promises.append(promise)
          for promise in b_promises:
            b_ranks = promise.get()
            for (b_subj, b_rel, b_obj), b_rank, b_score in b_ranks:
              subj_ranks[f'({b_subj} {b_rel} {b_obj})'] = [int(b_rank), float(b_score)]
          pbar.update(bsize)
      except tf.errors.OutOfRangeError:
        pass
      json.dump(subj_ranks, f)

    with open(os.path.join(outdir, 'obj_ranks.json'), 'w+') as f:
      model.data_generator.load_sub_rel_eval(session)
      pbar = tqdm(total=model.data_generator.nrof_sr)
      obj_ranks = {}
      try:
        while True:
          subj_rel_energy, b_subjs, b_rels = session.run([model.subj_rel_all_energy, model.b_sr_subjs, model.b_sr_rels])
          bsize = len(b_rels)
          b_promises = []
          for b_subj, b_rel, b_obj_energies in zip(b_subjs, b_rels, subj_rel_energy):
            b_real_objs = model.data_generator.test_sr2o[(b_subj, b_rel)]
            b_valid_objs = model.data_generator.sr2o[(b_subj, b_rel)]
            b_objs = model.data_generator.concepts
            promise = pool.apply_async(sort_and_rank_objs, (b_subj, b_rel, b_objs, b_obj_energies, b_real_objs, b_valid_objs))
            b_promises.append(promise)
          for promise in b_promises:
            b_ranks = promise.get()
            for (b_subj, b_rel, b_obj), b_rank, b_score in b_ranks:
              obj_ranks[f'({b_subj} {b_rel} {b_obj})'] = [int(b_rank), float(b_score)]
          pbar.update(bsize)
      except tf.errors.OutOfRangeError:
        pass
      json.dump(obj_ranks, f)


def calculate_scores(subj, rel, obj, replace_subject, concept_ids, session, model, batch_size):
  num_batches = int(math.ceil(float(len(concept_ids))/batch_size))
  subjects = np.full(batch_size, subj, dtype=np.int32)
  relations = np.full(batch_size, rel, dtype=np.int32)
  objects = np.full(batch_size, obj, dtype=np.int32)
  scores = {}

  for b in range(num_batches):
    concepts = concept_ids[b*batch_size:(b+1)*batch_size]
    feed_dict = {model.pos_subj: subjects,
                 model.relations: relations,
                 model.pos_obj: objects}

    # pad concepts if necessary
    if len(concepts) < batch_size:
      concepts = np.pad(concepts, (0, batch_size - len(concepts)), mode='constant', constant_values=0)

    # replace subj/obj in feed dict
    if replace_subject:
      feed_dict[model.pos_subj] = concepts
    else:
      feed_dict[model.pos_obj] = concepts

    # calculate energies
    energies = session.run(model.pos_energy, feed_dict)

    # store scores
    for i, cid in enumerate(concept_ids[b*batch_size:(b+1)*batch_size]):
      scores[cid] = energies[i]

  ranking = sorted(scores.items(), key=lambda k: k[1])

  rank_map = {}
  prev_rank = 0
  prev_score = -1
  total = 1
  for c, v in ranking:
    # if c has a lower score than prev
    if v > prev_score:
      # increment the rank
      prev_rank = total
      # update score
      prev_score = v
    total += 1
    rank_map[c] = prev_rank

  return scores, rank_map


def mrr(ranks_np):
  return float(np.mean(1. / ranks_np))


def mr(ranks_np):
  return float(np.mean(ranks_np))


def hits_at_10(ranks_np):
  return float(len(ranks_np[ranks_np <= 10])) / len(ranks_np)


def calculate_ranking_evals():
  outdir = os.path.join(config.eval_dir, config.run_name)

  ppa = float(str(next(open(os.path.join(outdir, 'ppa.txt')))).strip())
  print('PPA:  %.4f' % ppa)

  ranks = []
  for fname in os.listdir(outdir):
    full_path = os.path.join(outdir, fname)
    if os.path.isfile(full_path) and fname.startswith('ranks_'):
      for fields in json.load(open(full_path)):
        ranks += [fields[0], fields[1]]
  ranks_np = np.asarray(ranks, dtype=np.float)
  mrr_ = mrr(ranks_np)
  mr_ = mr(ranks_np)
  hat10 = hits_at_10(ranks_np)
  print('MRR:  %.4f' % mrr_)
  print('MR:   %.2f' % mr_)
  print('H@10: %.4f' % hat10)

  with open(os.path.join(outdir, 'ranking_evals.tsv'), 'w+') as f:
    f.write('ppa\t%f\n' % ppa)
    f.write('mrr\t%f\n' % mrr_)
    f.write('mr\t%f\n' % mr_)
    f.write('h@10\t%f' % hat10)


def calculate_ranking_evals_per_rel():
  outdir = os.path.join(config.eval_dir, config.run_name)

  ranks = defaultdict(list)
  for fname in os.listdir(outdir):
    full_path = os.path.join(outdir, fname)
    if os.path.isfile(full_path) and fname.startswith('ranks_'):
      for fields in json.load(open(full_path)):
        ranks[fields[4]] += [fields[0], fields[1]]
  print('Gathered %d rankings' % len(ranks))

  for rel, rl in ranks.items():
    ranks_np = np.asarray(rl, dtype=np.float)
    ranks[rel] = [mrr(ranks_np), mr(ranks_np), hits_at_10(ranks_np), len(rl)]

  relations = ranks.keys()
  sorted(relations, key=lambda x: ranks[x][1])

  with open(os.path.join(outdir, 'ranking_evals_per_rel.tsv'), 'w+') as f:
    for rel in relations:
      [mrr_, mr_, hat10, count] = ranks[rel]
      print('----------%s(%d)----------' % (rel, count))
      print('MRR:  %.4f' % mrr_)
      print('MR:   %.2f' % mr_)
      print('H@10: %.4f' % hat10)

      f.write('%s (%d)\n' % (rel, count))
      f.write('mrr\t%f\n' % mrr_)
      f.write('mr\t%f\n' % mr_)
      f.write('h@10\t%f\n\n' % hat10)


def split_ranking_evals():
  outdir = os.path.join(config.eval_dir, config.run_name)

  ppa = float(str(next(open(os.path.join(outdir, 'ppa.txt')))).strip())
  print('PPA:  %.4f' % ppa)

  s_ranks = []
  o_ranks = []
  for fname in os.listdir(outdir):
    full_path = os.path.join(outdir, fname)
    if os.path.isfile(full_path) and fname.startswith('ranks_'):
      for fields in json.load(open(full_path)):
        s_ranks += [fields[0]]
        o_ranks += [fields[1]]

  def report(ranks):
    ranks_np = np.asarray(ranks, dtype=np.float)
    mrr_ = mrr(ranks_np)
    mr_ = mr(ranks_np)
    hat10 = hits_at_10(ranks_np)
    print('MRR:  %.4f' % mrr_)
    print('MR:   %.2f' % mr_)
    print('H@10: %.4f' % hat10)
  report(s_ranks)
  report(o_ranks)


def fix_json():
  config = Config.flags
  outdir = os.path.join(config.eval_dir, config.run_name)

  ppa = float(str(next(open(os.path.join(outdir, 'ppa.txt')))).strip())
  print('PPA:  %.4f' % ppa)

  ranks = []
  for fname in os.listdir(outdir):
    full_path = os.path.join(outdir, fname)
    if os.path.isfile(full_path) and fname.startswith('ranks_'):
      json_string = '[' + str(next(open(full_path))).strip()
      with open(full_path, 'w+') as f:
        f.write(json_string)
      for fields in json.load(open(full_path)):
        ranks += [fields[0], fields[2]]
  ranks_np = np.asarray(ranks, dtype=np.float)
  mrr_ = mrr(ranks_np)
  mr_ = mr(ranks_np)
  hat10 = hits_at_10(ranks_np)
  print('MRR:  %.4f' % mrr_)
  print('MR:   %.2f' % mr_)
  print('H@10: %.4f' % hat10)

  with open(os.path.join(outdir, 'ranking_evals.tsv'), 'w+') as f:
    f.write('ppa\t%f\n' % ppa)
    f.write('mrr\t%f\n' % mrr_)
    f.write('mr\t%f\n' % mr_)
    f.write('h@10\t%f' % hat10)


if __name__ == "__main__":
  if config.eval_mode == "save":
    save_ranks()
  elif config.eval_mode == "calc":
    calculate_ranking_evals()
  elif config.eval_mode == "calc-rel":
    calculate_ranking_evals_per_rel()
  else:
    raise Exception('Unrecognized eval_mode: %s' % config.eval_mode)
