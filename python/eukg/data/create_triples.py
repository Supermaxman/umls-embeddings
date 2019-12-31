import numpy as np
import os
import csv
import json
from tqdm import tqdm
import argparse
from collections import defaultdict
import random

import hedgedog.nlp.wordpiece_tokenization as hgt

from .create_test_set import split
from . import umls_reader
from . import umls


def pref_atom_filter(x):
  # filter out non-english atoms
  if x.lat not in {'ENG'}:
    return False
  # TODO determine best way to filter atoms out
  # Ignore atoms with supress flag
  if x.suppress in {'O'}:
    return False
  # Ignore non-ts preferred atoms
  if x.ts not in {'P'}:
    return False
  # Ignore non-stt preferred atoms
  if x.stt not in {'PF'}:
    return False
  # Ignore non-ispref atoms
  if x.ispref not in {'Y'}:
    return False
  return True


def tokenize_concept(cui, umls_atoms, umls_defs, umls_contexts, tokenize):
  # TODO do this in a different way.
  assert len(umls_atoms) > 0, f'No atom found for concept {cui}!'
  primary_atom = umls_atoms[0]
  tokens, token_ids = tokenize(primary_atom.string)
  return token_ids


# def tokenize_rel(umls_rela, umls_rel_defs, tokenize):
#   rela_text = ' '.join(umls_rela.strip().split('_'))
#   _, token_ids = tokenize(rela_text)
#   return token_ids


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

  # TODO read in atoms other than only preferred.
  languages = {'ENG'}
  suppresses = {'O'}
  tses = {'P'}
  pfes = {'PF'}
  isprefs = {'Y'}

  print(f'Reading umls concepts...')
  def umls_concept_filter(x):
    # filter out non-english atoms
    # TODO allow other language atoms?
    if x.lat not in languages:
      return False
    # TODO determine best way to filter atoms out
    # Ignore atoms with suppress flag
    # if x.suppress in suppresses:
    #   return False
    # Ignore non-ts preferred atoms
    if x.ts not in tses:
      return False
    # Ignore non-stt preferred atoms
    if x.stt not in pfes:
      return False
    # Ignore non-ispref atoms
    if x.ispref not in isprefs:
      return False
    return True
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

  def tokenize(text):
    tokens = []
    token_ids = []
    # TODO simple whitespace tokenize, can do something more complicated later
    w_tokens = text.strip().lower().split()
    tokens.append('[CLS]')
    token_ids.append(vocab['[CLS]'])
    for w_t in w_tokens:
      wpt_tokens = tokenizer.tokenize(w_t)
      for wpt_t in wpt_tokens:
        tokens.append(wpt_t)
        token_ids.append(vocab[wpt_t])

    tokens.append('[SEP]')
    token_ids.append(vocab['[SEP]'])
    return tokens, token_ids

  umls_defs = defaultdict(list)
  umls_contexts = defaultdict(list)
  rel_defs = defaultdict(list)
  token_data = {}

  rel_iter = umls_reader.read_umls(
    rrf_file,
    umls.UmlsRelation,
    umls_filter=umls_rel_filter
  )
  rel_count = 0
  # total_matching_rel_count = 11391463
  total_matching_rel_count = 12833112
  # now get all rels which match our requirements and also have atoms.
  for rel in tqdm(rel_iter, desc="reading", total=total_matching_rel_count):
    # TODO make sure this is the direction we want these relations to go
    # TODO saw cui2 rela cui1 in documentation
    # TODO make sure we only want named relations (rela is not empty)
    # TODO consider flipping this to cui2 rela cui1 as per documentation.
    sid = add_concept(rel.cui1)
    rel_cui = rel_merge_mapping[f'{rel.rel}:{rel.rela}']
    rid = add_concept(rel_cui)
    if rid not in token_data:
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
      token_data[rid] = (t_ids, len(t_ids))
    oid = add_concept(rel.cui2)
    triples.add((sid, rid, oid))
    rel_count += 1
  print(f'Matching rel count: {rel_count}')
  print(f'{len(token_data)} tokenized')
  print(f'{max(token_data.keys())} max id')

  def umls_atom_filter(x):
    # filter out non-english atoms
    # TODO allow other language atoms?
    if x.lat not in languages:
      return False
    # TODO determine best way to filter atoms out
    # Ignore atoms with suppress flag
    # if x.suppress in suppresses:
    #   return False
    # Ignore non-ts preferred atoms
    if x.ts not in tses:
      return False
    # Ignore non-stt preferred atoms
    if x.stt not in pfes:
      return False
    # Ignore non-ispref atoms
    if x.ispref not in isprefs:
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
  # total_matching_atom_count = 1563246
  total_matching_atom_count = 3210782
  # finally, get atoms for only concepts which we have relations for.
  for atom in tqdm(atom_iter, desc="reading", total=total_matching_atom_count):
    cid = conc2id[atom.cui]
    # TODO make sure priority atom is first.
    if cid not in token_data:
      _, t_ids = tokenize(atom.string)
      token_data[cid] = (t_ids, len(t_ids))
    atom_count += 1

  print(f'Read {atom_count} atoms.')
  print(f'{len(token_data)} tokenized')
  print(f'{max(token_data.keys())} max id')
  tokenized_count = len(token_data)
  max_token_idx = max(token_data.keys())
  assert tokenized_count == max_token_idx + 1, \
    f'token_data not densely packed! len={tokenized_count} vs max={max_token_idx}'

  token_lengths = np.array([t_l for cid, (t_ids, t_l) in token_data.items()], dtype=np.int32)
  min_token_count = np.min(token_lengths)
  max_token_count = np.max(token_lengths)
  avg_token_count = np.mean(token_lengths)
  percentile_token_count = np.percentile(token_lengths, 95)
  # 3
  print(f'Min token counts: {min_token_count}')
  # 1189
  print(f'Max token counts: {max_token_count}')
  # 13.7
  print(f'Avg token counts: {avg_token_count}')
  # 27
  print(f'95 Percentile token counts: {percentile_token_count}')
  pad_count = int(np.ceil(percentile_token_count))
  # p_tokens = {}
  print('Padding tokens...')
  token_ids = np.zeros([len(token_data), pad_count], dtype=np.int32)
  for cid, (t_ids, t_l) in token_data.items():
    if t_l > pad_count:
      token_ids[cid] = t_ids[:pad_count]
    elif t_l < pad_count:
      token_ids[cid] = t_ids + [0] * (pad_count - t_l)
    else:
      token_ids[cid] = t_ids
  del token_data
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
  with open(os.path.join(output_dir, 'concept_vocab.json'), 'w+') as f:
    json.dump(concepts, f, indent=2)
  with open(os.path.join(output_dir, 'relation_vocab.json'), 'w+') as f:
    json.dump(relations, f, indent=2)

  print('Saving tokens...')
  np.savez_compressed(
    os.path.join(output_dir, 'id2tokens.npz'),
    token_ids=token_ids,
    token_lengths=token_lengths
  )
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
