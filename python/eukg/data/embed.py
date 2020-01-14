import tensorflow as tf
import numpy as np
import os
import json
from tqdm import tqdm
import argparse
import hedgedog.nlp.wordpiece_tokenization as hgt


from ..ace import load_ace


def embed():
  parser = argparse.ArgumentParser()
  parser.add_argument('--ctxt2id_file', default='/home/max/data/artifacts/i2b2/2010/new_data/ctxt2id.json')
  parser.add_argument('--ace_path', default='/users/max/data/models/umls-embeddings/transd-distmult/transd-dm-gan-joint-ace-20')
  parser.add_argument('--out_file', default='/home/max/data/artifacts/i2b2/2010/new_data/transd-dm-gan-joint-ace-20-embeddings.npz')
  parser.add_argument('--embedding_size', default=100)
  config = parser.parse_args()

  vocab_file = '/shared/hltdir4/disk1/team/data/models/bert/uncased_L-24_H-1024_A-16/vocab.txt'
  vocab = hgt.load_vocab(vocab_file)
  tokenizer = hgt.WordpieceTokenizer(vocab)
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

  ctxt2id = json.load(open(config.ctxt2id_file))
  nrof_concepts = len(ctxt2id)
  print(f'{nrof_concepts} total unique concepts')

  ctokens = {}
  for c_txt, c_id in ctxt2id.items():
    _, c_tokens = tokenize(c_txt)
    ctokens[c_id] = (c_tokens, len(c_tokens))

  token_lengths = np.array([t_l for cid, (t_ids, t_l) in ctokens.items()], dtype=np.int32)
  pad_count = int(np.ceil(max(token_lengths)))
  token_ids = np.zeros([len(ctokens), pad_count], dtype=np.int32)
  for cid, (t_ids, t_l) in ctokens.items():
    if t_l > pad_count:
      token_ids[cid] = t_ids[:pad_count]
    elif t_l < pad_count:
      token_ids[cid] = t_ids + [0] * (pad_count - t_l)
    else:
      token_ids[cid] = t_ids

  concept_ids = np.arange(len(ctokens))

  gpu_config = tf.ConfigProto()
  gpu_config.gpu_options.allow_growth = True

  with tf.Graph().as_default(), tf.Session(config=gpu_config) as session:
    model = load_ace(config.ace_path)

    # TODO build dataset from ctxt2id
    data = tf.data.Dataset.from_tensor_slices((token_ids, concept_ids))
    data = data.batch(batch_size=32)
    data = data.prefetch(buffer_size=1)

    iterator = data.make_one_shot_iterator()
    b_token_ids, b_concept_ids = iterator.get_next()
    b_concept_embs = model(b_token_ids, 'concept')

    tf.global_variables_initializer().run()
    tf.local_variables_initializer().run()

    emb_size = config.embedding_size
    embeddings = np.zeros((nrof_concepts, emb_size // 2), dtype=np.float32)
    embeddings_proj = np.zeros((nrof_concepts, emb_size // 2), dtype=np.float32)
    seen_concepts = np.zeros(nrof_concepts, dtype=np.bool)
    pbar = tqdm(total=nrof_concepts)
    print('Creating concept embeddings...')
    try:
      while True:
        (c_embs, c_embs_proj), c_ids = session.run([b_concept_embs, b_concept_ids])
        embeddings[c_ids] = c_embs
        embeddings_proj[c_ids] = c_embs_proj
        pbar.update(len(c_ids))
        seen_concepts[c_ids] = True
    except tf.errors.OutOfRangeError:
      pass

    seen_count = int(seen_concepts.astype(np.int32).sum())
    total_count = len(seen_concepts)
    print(f'Seen concepts: {seen_count}/{total_count}')
    assert seen_count == total_count, 'Not all concepts have been seen!'
    print('Saving embeddings...')
    np.savez_compressed(
      config.out_file,
      embs=embeddings,
      p_embs=embeddings_proj
    )


if __name__ == "__main__":
  embed()
