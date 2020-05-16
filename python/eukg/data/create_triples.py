import numpy as np
import os
import csv
import json
from tqdm import tqdm
import argparse
from collections import defaultdict
import random

import hedgedog.nlp.wordpiece_tokenization as hgt
import tensorflow as tf

from .create_test_set import split
from . import umls_reader
from . import umls

import scispacy
import spacy


def _bytes_feature(value):
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
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
    a_str = atom.string.lower().strip()
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
    # w_tokens = text.strip().lower().split()
    tokens.append('[CLS]')
    token_ids.append(vocab['[CLS]'])
    for w_t in doc:
      wpt_tokens = tokenizer.tokenize(w_t.string.lower())
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
  total_matching_atom_count = 6873557
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

  triples_dir = os.path.join(output_dir, 'triples')
  if not os.path.exists(triples_dir):
    os.mkdir(triples_dir)

  subjs, rels, objs = zip(*triples)
  snp = np.asarray(subjs, dtype=np.int32)
  rnp = np.asarray(rels, dtype=np.int32)
  onp = np.asarray(objs, dtype=np.int32)

  id2conc = {v: k for k, v in conc2id.items()}

  print(f"Saving {rnp.shape[0]} unique triples to {output_dir}.")
  print(f"{len(concepts)} concepts spanning {len(relation_types)} relation types")
  train_idx, val_idx, test_idx = split(snp, rnp, onp, output_dir)

  def save_triples(idxs, name):
    print(f'Creating {name} tfrecords...')
    with tf.io.TFRecordWriter(os.path.join(triples_dir, f'{name}.tfrecords')) as writer:
      for r_idx, sid, rid, oid in tqdm(zip(idxs, snp[idxs], rnp[idxs], onp[idxs]), total=len(idxs)):
        subj = concepts[id2conc[sid]]
        rt = relation_types[id2conc[rid]]
        obj = concepts[id2conc[oid]]
        features = {
          'r_idx': _int64_feature(r_idx),
          'subj_id': _int64_feature(sid),
          'subj_token_ids': tf.train.Feature(int64_list=tf.train.Int64List(value=subj.atom_tokens[subj.p_atom_idx])),
          'subj_token_length': _int64_feature(len(subj.atom_tokens[subj.p_atom_idx])),
          'rt_id': _int64_feature(rid),
          'rt_token_ids': tf.train.Feature(int64_list=tf.train.Int64List(value=rt.rel_tokens)),
          'rt_token_length': _int64_feature(len(rt.rel_tokens)),
          'obj_id': _int64_feature(oid),
          'obj_token_ids': tf.train.Feature(int64_list=tf.train.Int64List(value=obj.atom_tokens[obj.p_atom_idx])),
          'obj_token_length': _int64_feature(len(obj.atom_tokens[obj.p_atom_idx])),
        }
        example_proto = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(example_proto.SerializeToString())

  print('Saving triples...')
  save_triples(train_idx, 'train')
  save_triples(val_idx, 'val')
  save_triples(test_idx, 'test')

  print('Saving dicts...')
  with open(os.path.join(output_dir, 'name2id.json'), 'w+') as f:
    json.dump(conc2id, f, indent=2)

  print('Done!')


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

  vocab_file = '/shared/hltdir4/disk1/team/data/models/bert/uncased_L-24_H-1024_A-16/vocab.txt'
  # Previously MRCONSO.RRF, changed to MRREL.RRF
  metathesaurus_triples(args.umls_dir, args.output, args.data_folder, vocab_file)


if __name__ == "__main__":
  main()
