import numpy as np
import os
import csv
import json
from tqdm import tqdm
import argparse
from collections import defaultdict
import random
import tensorflow as tf
import zlib
import hedgedog.nlp.wordpiece_tokenization as hgt

from .create_test_set import split
from . import umls_reader
from . import umls

import scispacy
import spacy
from ..emb import LanguageModel
from ..tf_util import checkpoint_utils


def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def is_preferred(x):
  return x.ts == 'P' and x.stt == 'PF' and x.ispref == 'Y'


class ConceptExample:
  def __init__(self, cid, cui):
    self.cid = cid
    self.cui = cui
    self.p_atom_idx = None
    self.atom_tokens = []
    self.atom_lookup = {}

  def to_dict(self):
    d = {
      'cid': self.cid,
      'cui': self.cui,
      'p_atom_idx': self.p_atom_idx,
      'atom_tokens': self.atom_tokens,
      'atom_lookup': self.atom_lookup
    }
    return d

  def add_atom_string(self, atom):
    a_str = atom.string.strip()
    added_atom = False
    if a_str not in self.atom_lookup:
      self.atom_lookup[a_str] = len(self.atom_lookup)
      added_atom = True
    if is_preferred(atom) and self.p_atom_idx is None:
      self.p_atom_idx = self.atom_lookup[a_str]
    return added_atom

  def to_json(self):
    return json.dumps(self.to_dict())


class RelationTypeExample:
  def __init__(self, rid, rel_cui):
    self.rid = rid
    self.rel_cui = rel_cui
    self.rel_tokens = None

  def to_dict(self):
    dict = {
      'rid': self.rid,
      'rel_cui': self.rel_cui,
      'rel_tokens': self.rel_tokens
    }
    return dict

  def to_json(self):
    return json.dumps(self.to_dict())


def load_rel_merge_mapping(filepath):
  rel_merge_mapping = {}
  with open(filepath, 'r') as f:
    for line in f:
      s, t = line.strip().split(',')
      rel_merge_mapping[s] = t
  return rel_merge_mapping


def load_rela_mapping(filepath):
  rela_mapping = {}
  with open(filepath, 'r') as f:
    for line in f:
      rela, rela_text = line.strip().split('\t')
      rela = rela.strip()
      rela_text = rela_text.strip()
      rela_mapping[rela] = rela_text
  return rela_mapping


def load_rel_mapping(filepath):
  rel_mapping = {}
  with open(filepath, 'r') as f:
    for line in f:
      rela, rel_text = line.strip().split('\t')
      rela = rela.strip()
      rel_text = rel_text.strip()
      rel_mapping[rela] = rel_text
  return rel_mapping


def metathesaurus_triples(umls_dir, output_dir, data_folder, vocab_file):
  # TODO completely restructure way this data is generated.
  # TODO first generate triples, then
  # TODO create file structure with kept atoms, definitions, contexts, etc.
  # TODO and create dataset loader, can be queued.
  triples = set()
  conc2id = {}
  rrf_file = os.path.join(umls_dir, 'META', 'MRREL.RRF')
  conso_file = os.path.join(umls_dir, 'META', 'MRCONSO.RRF')

  rel_merge_mapping = load_rel_merge_mapping(os.path.join(data_folder, 'rel_merge_mapping.txt'))
  rel_mapping = load_rel_mapping(os.path.join(data_folder, 'rel_desc.txt'))
  rela_mapping = load_rela_mapping(os.path.join(data_folder, 'rela_desc.txt'))

  vocab = hgt.load_vocab(vocab_file)
  tokenizer = hgt.WordpieceTokenizer(vocab)

  valid_rel_cuis = set(rel_merge_mapping.keys())
  languages = {'ENG'}
  print(f'Reading umls concepts...')
  def umls_concept_filter(x):
    # filter out non-english atoms
    if x.lat not in languages:
      return False
    # Ignore non-preferred atoms
    if is_preferred(x):
      return True
    return False
  concept_iter = umls_reader.read_umls(
    conso_file,
    umls.UmlsAtom,
    umls_filter=umls_concept_filter
  )
  seen_cuis = set()
  total_matching_concept_count = 3285966
  # First pass through to get all possible cuis which we have atoms for.
  for atom in tqdm(concept_iter, desc="reading", total=total_matching_concept_count):
    seen_cuis.add(atom.cui)
  print(f'Matching cui count: {len(seen_cuis)}')

  def umls_rel_filter(x):
    # remove recursive relations
    if x.cui2 == x.cui1:
      return False
    # ignore siblings, CHD is enough to infer
    if x.rel == 'SIB':
      return False
    # ignore PAR, CHD is reflexive
    if x.rel == 'PAR':
      return False
    # ignore RO with no relA, not descriptive
    if x.rel == 'RO' and x.rela == '':
      return False
    # reflexive with AQ
    if x.rel == 'QB':
      return False
    # too vague
    if x.rel == 'RB':
      return False
    # removes rels which have too few instances to keep around
    if f'{x.rel}:{x.rela}' not in valid_rel_cuis:
      return False
    # removes rels which do not have matching atoms/cuis
    if x.cui1 not in seen_cuis or x.cui2 not in seen_cuis:
      return False
    return True

  def add_concept(conc):
    if conc in conc2id:
      cid = conc2id[conc]
    else:
      cid = len(conc2id)
      conc2id[conc] = cid
    return cid

  nlp = spacy.load('en_core_sci_sm', disable=['tagger', 'parser', 'ner', 'textcat'])

  def tokenize(text):
    tokens = []
    token_ids = []
    doc = nlp(text.strip())
    tokens.append('[CLS]')
    token_ids.append(vocab['[CLS]'])
    for w_t in doc:
      wpt_tokens = tokenizer.tokenize(w_t.string)
      for wpt_t in wpt_tokens:
        tokens.append(wpt_t)
        token_ids.append(vocab[wpt_t])

    tokens.append('[SEP]')
    token_ids.append(vocab['[SEP]'])
    return tokens, token_ids

  rel_iter = umls_reader.read_umls(
    rrf_file,
    umls.UmlsRelation,
    umls_filter=umls_rel_filter
  )
  rel_count = 0

  total_matching_rel_count = 12833112
  relation_types = {}
  # now get all rels which match our requirements and also have atoms.
  for rel in tqdm(rel_iter, desc="reading", total=total_matching_rel_count):
    sid = add_concept(rel.cui1)
    rel_cui = rel_merge_mapping[f'{rel.rel}:{rel.rela}']
    rid = add_concept(rel_cui)
    if rel_cui not in relation_types:
      relation_types[rel_cui] = RelationTypeExample(rid, rel_cui)
      cui_rel, cui_rela = rel_cui.split(':')
      # if there is no rela then we use rel text
      if cui_rela == '':
        rel_text = rel_mapping[cui_rel]
      else:
        if cui_rela not in rela_mapping:
          rela_mapping[cui_rela] = ' '.join(cui_rela.split('_'))
          print(f'rela {cui_rela} not found in text mapping, defaulting to {rela_mapping[cui_rela]}.')
        rel_text = rela_mapping[cui_rela]
      _, t_ids = tokenize(rel_text)
      rt = relation_types[rel_cui]
      rt.rel_tokens = t_ids

    oid = add_concept(rel.cui2)
    triples.add((sid, rid, oid))
    rel_count += 1
  print(f'Matching rel count: {rel_count}')
  print(f'{len(relation_types)} tokenized')

  def umls_atom_filter(x):
    # filter out non-english atoms
    # TODO allow other language atoms?
    if x.lat not in languages:
      return False
    # ignore atoms for concepts of which there are no relations.
    if x.cui not in conc2id:
      return False
    return True

  print(f'Reading umls atoms...')
  atom_iter = umls_reader.read_umls(
      conso_file,
      umls.UmlsAtom,
      umls_filter=umls_atom_filter
  )
  atom_count = 0
  # total_matching_atom_count = 6873557
  total_matching_atom_count = 7753235
  concepts = {}

  # finally, get atoms for only concepts which we have relations for.
  for atom in tqdm(atom_iter, desc="reading", total=total_matching_atom_count):
    cid = conc2id[atom.cui]
    if atom.cui not in concepts:
      concepts[atom.cui] = ConceptExample(cid, atom.cui)
    c = concepts[atom.cui]
    if c.add_atom_string(atom):
      _, t_ids = tokenize(atom.string)
      c.atom_tokens.append(t_ids)
    atom_count += 1

  print(f'Read {atom_count} atoms.')
  print(f'{len(concepts)} concepts')
  #
  # entity_dir = os.path.join(output_dir, 'entities')
  # if not os.path.exists(entity_dir):
  #   os.mkdir(entity_dir)

  lm_embeddings_dir = os.path.join(output_dir, 'lm_embeddings')
  if not os.path.exists(lm_embeddings_dir):
    os.mkdir(lm_embeddings_dir)

  lm_rt_embeddings_dir = os.path.join(lm_embeddings_dir, 'rt')
  if not os.path.exists(lm_rt_embeddings_dir):
    os.mkdir(lm_rt_embeddings_dir)

  lm_concept_embeddings_dir = os.path.join(lm_embeddings_dir, 'concept')
  if not os.path.exists(lm_concept_embeddings_dir):
    os.mkdir(lm_concept_embeddings_dir)

  sizes = []
  p_sizes = []
  s_sizes = []
  s_counts = []

  print('Calculating sizes of relation types...')
  for rui, rt in tqdm(relation_types.items(), desc="calculating", total=len(relation_types)):
    sizes.append(len(rt.rel_tokens))
    # rt_str = rt.to_json()
    # rt_file = os.path.join(entity_dir, f'{rt.rid}.json')
    # with open(rt_file, 'w') as f:
    #   f.write(rt_str)

  print('Calculating sizes of concepts...')
  for cui, c in tqdm(concepts.items(), desc="calculating", total=len(concepts)):

    assert c.p_atom_idx >= 0, f'AssertionError: p_atom_idx: {c.p_atom_idx}'
    assert c.p_atom_idx < len(c.atom_tokens), f'AssertionError: p_atom_idx: {c.p_atom_idx}'
    for idx, atom_t in enumerate(c.atom_tokens):
      atom_size = len(atom_t)
      if c.p_atom_idx == idx:
        p_sizes.append(atom_size)
      else:
        s_sizes.append(atom_size)
      sizes.append(atom_size)
    s_counts.append(len(c.atom_tokens) - 1)
    # c_str = c.to_json()
    # c_file = os.path.join(entity_dir, f'{c.cid}.json')
    # with open(c_file, 'w') as f:
    #   f.write(c_str)

  def stats(name, data):
    d_mean = np.mean(data)
    d_percentile = int(np.round(np.percentile(data, 95)))
    d_min = np.min(data)
    d_max = np.max(data)
    print(f'{name}: mean={d_mean:.2f}, 95p={d_percentile}, min={d_min}, max={d_max}')
    return d_percentile
  atom_token_pad = stats('sizes', sizes)
  stats('p_sizes', p_sizes)
  stats('s_sizes', s_sizes)
  np_atom_count_pad = stats('s_counts', s_counts)

  gpu_config = tf.ConfigProto()
  gpu_config.gpu_options.allow_growth = True
  bert_config = '/users/max/data/models/bert/biobert_v1.1_pubmed/bert_config.json'
  encoder_checkpoint = '/users/max/data/models/bert/biobert_v1.1_pubmed/bert_model.ckpt'
  with tf.Graph().as_default(), tf.Session(config=gpu_config) as session:
    print('Loading bert...')
    lm = LanguageModel.BertLanguageModel(
      bert_config_path=bert_config,
      train_bert=False
    )
    entity_token_ids = tf.placeholder(dtype=tf.int32, shape=[None, None])
    entity_token_lengths = tf.placeholder(dtype=tf.int32, shape=[None])
    entity_seq_embeddings = lm.encode(entity_token_ids, entity_token_lengths)

    checkpoint_utils.init_from_checkpoint(encoder_checkpoint)

    tf.global_variables_initializer().run()
    tf.local_variables_initializer().run()

    print('Creating relation type lm embeddings...')
    for rui, rt in tqdm(relation_types.items(), desc="embedding", total=len(relation_types)):
      t_l = len(rt.rel_tokens)
      t_ids = rt.rel_tokens
      token_ids = np.expand_dims(np.array(t_ids[:atom_token_pad], dtype=np.int32), axis=0)
      token_lengths = np.array([min(t_l, atom_token_pad)], dtype=np.int32)

      token_embeddings = session.run(
        entity_seq_embeddings,
        feed_dict={
          entity_token_ids: token_ids,
          entity_token_lengths: token_lengths
        }
      )[0]
      rt_file = os.path.join(lm_rt_embeddings_dir, f'{rt.rid}.tfexample')
      feature = {
        'lm_embedding': tf.train.Feature(
          float_list=tf.train.FloatList(
            value=np.reshape(
              token_embeddings,
              token_embeddings.shape[0] * token_embeddings.shape[1]
            )
          )
        ),
        'token_length': _int64_feature(token_lengths[0]),
        'token_ids': tf.train.Feature(
          int64_list=tf.train.Int64List(
            value=token_ids[0]
          )
        ),
        'entity_id': _int64_feature(rt.rid)
      }
      with open(rt_file, 'wb') as f:
        example_proto_str = tf.train.Example(
          features=tf.train.Features(feature=feature)
        ).SerializeToString()
        example_proto_str = zlib.compress(example_proto_str)
        f.write(example_proto_str)

    print('Creating concept lm embeddings...')
    for cui, c in tqdm(concepts.items(), desc="calculating", total=len(concepts)):
      c_file = os.path.join(lm_concept_embeddings_dir, f'{c.cid}.tfexample')
      # if os.path.exists(c_file):
      #   statinfo = os.stat(c_file)
      #   if statinfo.st_size != 0:
      #     continue
      nrof_atoms = len(c.atom_tokens)
      max_token_length = max([len(x) for x in c.atom_tokens])
      concept_token_pad = min(max_token_length, atom_token_pad)
      token_ids = np.zeros([nrof_atoms, concept_token_pad], dtype=np.int32)
      token_lengths = np.zeros([nrof_atoms], dtype=np.int32)
      for a_id, t_ids in enumerate(c.atom_tokens):
        t_l = len(t_ids)
        if t_l > concept_token_pad:
          token_ids[a_id] = t_ids[:concept_token_pad]
          token_lengths[a_id] = concept_token_pad
        elif t_l < concept_token_pad:
          token_ids[a_id] = t_ids + [0] * (concept_token_pad - t_l)
          token_lengths[a_id] = t_l
        else:
          token_ids[a_id] = t_ids
          token_lengths[a_id] = t_l

      if nrof_atoms > np_atom_count_pad:
        nrof_atoms = np_atom_count_pad
        token_ids = token_ids[:np_atom_count_pad]
        token_lengths = token_lengths[:np_atom_count_pad]

      token_embeddings = session.run(
        entity_seq_embeddings,
        feed_dict={
          entity_token_ids: token_ids,
          entity_token_lengths: token_lengths
        }
      )
      lm_emb_size = token_embeddings.shape[2]
      feature = {
        'lm_embeddings': tf.train.Feature(
          float_list=tf.train.FloatList(
            value=np.reshape(
              token_embeddings,
              nrof_atoms * concept_token_pad * lm_emb_size
            )
          )
        ),
        'token_lengths': tf.train.Feature(int64_list=tf.train.Int64List(value=token_lengths)),
        'token_ids': tf.train.Feature(
          int64_list=tf.train.Int64List(
            value=np.reshape(
              token_ids,
              nrof_atoms * concept_token_pad
            )
          )
        ),
        'entity_id': _int64_feature(c.cid),
        'nrof_atoms': _int64_feature(nrof_atoms),
        'concept_token_pad': _int64_feature(concept_token_pad),
        'lm_emb_size': _int64_feature(lm_emb_size),
        'p_atom_idx': _int64_feature(c.p_atom_idx)
      }
      with open(c_file, 'wb') as f:
        example_proto_str = tf.train.Example(
          features=tf.train.Features(feature=feature)
        ).SerializeToString()
        example_proto_str = zlib.compress(example_proto_str)
        f.write(example_proto_str)

  subjs, rels, objs = zip(*triples)
  snp = np.asarray(subjs, dtype=np.int32)
  rnp = np.asarray(rels, dtype=np.int32)
  onp = np.asarray(objs, dtype=np.int32)

  id2conc = {v: k for k, v in conc2id.items()}
  concepts = [id2conc[i] for i in np.unique(np.concatenate((subjs, objs)))]
  relations = [id2conc[i] for i in set(rels)]

  print(f"Saving {rnp.shape[0]} unique triples to {output_dir}.")
  print(f"{len(concepts)} concepts spanning {len(relations)} relations")
  split(snp, rnp, onp, output_dir)
  print('Saving dicts...')
  with open(os.path.join(output_dir, 'name2id.json'), 'w+') as f:
    json.dump(conc2id, f, indent=2)

  print('Done!')

  def stats(name, data):
    d_mean = np.mean(data)
    d_percentile = int(np.round(np.percentile(data, 95)))
    d_min = np.min(data)
    d_max = np.max(data)
    print(f'{name}: mean={d_mean:.2f}, 95p={d_percentile}, min={d_min}, max={d_max}')
    return d_percentile
  stats('sizes', sizes)
  stats('p_sizes', p_sizes)
  stats('s_sizes', s_sizes)
  stats('s_counts', s_counts)
  print(f'atom_count: {atom_count}')


def main():
  parser = argparse.ArgumentParser(description='Extract relation triples into a compressed numpy file from MRCONSO.RRF')
  parser.add_argument(
    '--umls_dir',
    default='data',
    help='UMLS MRCONSO.RRF file containing metathesaurus relations')
  parser.add_argument('--output', default='data', help='the compressed numpy file to be created')
  parser.add_argument('--data_folder', default='python/data',
                      help='Data folder.')
  parser.add_argument('--seed', default=1337,
                      help='Random seed.')

  args = parser.parse_args()
  seed = args.seed
  random.seed(seed)
  np.random.seed(seed)

  vocab_file = '/users/max/data/models/bert/biobert_v1.1_pubmed/vocab.txt'
  # Previously MRCONSO.RRF, changed to MRREL.RRF
  metathesaurus_triples(args.umls_dir, args.output, args.data_folder, vocab_file)


if __name__ == "__main__":
  main()
