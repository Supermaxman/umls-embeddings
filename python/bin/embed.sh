#!/usr/bin/env bash

python -m python.eukg.data.embed

python -m python.eukg.data.embed_rels \
  --relid2txt='/home/max/data/artifacts/i2b2/2010/new_data/relid2txt_def.json' \
  --out_file='/home/max/data/artifacts/i2b2/2010/new_data/transd-dm-gan-joint-ace-20-rel-def-embeddings.npz'

python -m python.eukg.data.embed_rels \
  --relid2txt='/home/max/data/artifacts/i2b2/2010/new_data/relid2txt_def_inv.json' \
  --out_file='/home/max/data/artifacts/i2b2/2010/new_data/transd-dm-gan-joint-ace-20-rel-def-inv-embeddings.npz'


python -m python.eukg.data.embed_rels \
  --relid2txt='/home/max/data/artifacts/i2b2/2010/new_data/relid2txt_umls.json' \
  --out_file='/home/max/data/artifacts/i2b2/2010/new_data/transd-dm-gan-joint-ace-20-rel-umls-embeddings.npz'

python -m python.eukg.data.embed_rels \
  --relid2txt='/home/max/data/artifacts/i2b2/2010/new_data/relid2txt_umls_specific.json' \
  --out_file='/home/max/data/artifacts/i2b2/2010/new_data/transd-dm-gan-joint-ace-20-rel-umls-specific-embeddings.npz'

