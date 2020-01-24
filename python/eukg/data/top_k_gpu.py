
import numpy as np
import json
import argparse
import tensorflow as tf
from tqdm import tqdm


def top_k():
  parser = argparse.ArgumentParser()
  parser.add_argument('--emb_path', default='/users/rmm120030/working/kge_ner/info/max/embeddings.npz')
  parser.add_argument('--mention_path', default='/users/rmm120030/working/kge_ner/info/medmentions_mentions.npz')
  parser.add_argument('--out_file', default='/users/rmm120030/working/kge_ner/info/knn-emb.npz')
  parser.add_argument('--k', default=100, type=int)
  parser.add_argument('--batch_size', default=8, type=int)
  config = parser.parse_args()
  k = config.k

  with np.load(config.emb_path) as npz:
    embs = npz['embs']
  with np.load(config.mention_path) as npz:
    ment = npz['embs']

  print(f'Total embs: {len(embs)}')
  print(f'Total ments: {len(ment)}')
  distances = np.zeros(
    shape=[len(ment), k],
    dtype=np.float32
  )
  candidates = np.zeros(
    shape=[len(ment), k],
    dtype=np.int32
  )
  mention_ids = np.arange(len(ment))

  gpu_config = tf.ConfigProto()
  gpu_config.gpu_options.allow_growth = True

  with tf.Graph().as_default(), tf.Session(config=gpu_config) as session:
    # [len(embs), 50]
    embeddings = tf.constant(embs)
    mentions = tf.constant(ment)

    data = tf.data.Dataset.from_tensor_slices((mention_ids,))
    data = data.batch(batch_size=config.batch_size)
    data = data.prefetch(buffer_size=1)
    iterator = data.make_one_shot_iterator()
    # bsize, [bsize, 50]
    b_mention_id = iterator.get_next()
    b_mention = tf.nn.embedding_lookup(mentions, b_mention_id)
    # [bsize, 1, 50] + [1, e, 50]
    diff = tf.expand_dims(b_mention, axis=1) - tf.expand_dims(embeddings, axis=0)
    # [bsize, e]
    dists = tf.reduce_sum(diff * diff, axis=-1)
    # [bsize, k]
    dist_top_k = tf.argsort(
      dists,
      axis=-1,
      direction='ASCENDING'
    )[:, :k]
    # [bsize, k]
    top_k_dists = tf.gather(
      tf.expand_dims(dists, axis=-1),
      dist_top_k,
      batch_dims=1
    )[:, :, 0]

    seen_concepts = np.zeros(len(ment), dtype=np.bool)
    pbar = tqdm(total=len(ment))
    print('Gathering top k...')
    try:
      while True:
        m_ids, m_top_ks, m_top_ds = session.run([b_mention_id, dist_top_k, top_k_dists])
        candidates[m_ids] = m_top_ks
        distances[m_ids] = m_top_ds
        pbar.update(len(m_ids))
        seen_concepts[m_ids] = True
    except tf.errors.OutOfRangeError:
      pass

  seen_count = int(seen_concepts.astype(np.int32).sum())
  total_count = len(seen_concepts)
  print(f'Seen mentions: {seen_count}/{total_count}')
  assert seen_count == total_count, 'Not all mentions have been seen!'
  print('Saving results...')
  np.savez_compressed(
    config.out_file,
    distances=distances,
    candidates=candidates
  )

  print(f'Done!')


if __name__ == "__main__":
  top_k()
