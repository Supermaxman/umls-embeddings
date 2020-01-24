
import numpy as np
import json
import argparse
from sklearn.neighbors import NearestNeighbors


def top_k():
  parser = argparse.ArgumentParser()
  parser.add_argument('--emb_path', default='/users/rmm120030/working/kge_ner/info/max/embeddings.npz')
  parser.add_argument('--ment_path', default='/users/rmm120030/working/kge_ner/info/medmentions_mentions.npz')
  parser.add_argument('--out_file', default='/users/rmm120030/working/kge_ner/info/knn-emb.npz')
  parser.add_argument('--k', default=100, type=int)
  config = parser.parse_args()

  ctxt2id = json.load(open(config.ctxt2id_file))
  nrof_concepts = len(ctxt2id)
  print(f'{nrof_concepts} total unique concepts')
  k = config.k
  top_candidates = np.zeros(shape=[nrof_concepts, k])
  with np.load(config.emb_path) as npz:
    embs = npz['embs']
  with np.load(config.mention_path) as npz:
    ment = npz['embs']

  print(f'Total embs: {len(embs)}')
  print(f'Total ments: {len(ment)}')
  print(f'Running KNN...')
  knn = NearestNeighbors(n_neighbors=1, metric='l2')
  knn.fit(embs)
  print(f'Querying KNN...')
  distances, candidates = knn.kneighbors(ment, n_neighbors=k)

  np.savez_compressed(
    config.out_file,
    distances=distances,
    candidates=candidates
  )

  print(f'Done!')


if __name__ == "__main__":
  top_k()
