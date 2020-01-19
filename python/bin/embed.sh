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



python -m python.eukg.data.embed_rels \
  --relid2txt='/users/max/data/corpora/ChemProt/original/relid2txt_umls.json' \
  --out_file='/home/max/data/artifacts/chemprot/transd-dm-gan-joint-ace-20-rel-umls-embeddings.npz'


python -m python.eukg.data.embed_rels \
  --relid2txt='/users/max/data/corpora/ChemProt/original/relid2txt_custom.json' \
  --out_file='/home/max/data/artifacts/chemprot/transd-dm-gan-joint-ace-20-rel-custom-embeddings.npz'

  python -m python.eukg.data.embed_rels \
  --relid2txt='/users/max/data/corpora/ChemProt/original/relid2txt_umls_2.json' \
  --out_file='/home/max/data/artifacts/chemprot/transd-dm-gan-joint-ace-20-rel-umls-2-embeddings.npz'

  python -m python.eukg.data.embed_rels \
  --relid2txt='/users/max/data/corpora/ChemProt/original/relid2txt_umls_3.json' \
  --out_file='/home/max/data/artifacts/chemprot/transd-dm-gan-joint-ace-20-rel-umls-3-embeddings.npz'


python -m python.eukg.data.embed \
  --ctxt2id_file='/home/max/data/corpora/ddi2013-type/original/ctxt2id.json' \
  --out_file='/home/max/data/artifacts/ddi2013-type/transd-dm-gan-joint-ace-20-embeddings.npz'

python -m python.eukg.data.embed_rels \
  --relid2txt='/users/max/data/corpora/ddi2013-type/original/relid2txt_umls.json' \
  --out_file='/home/max/data/artifacts/ddi2013-type/transd-dm-gan-joint-ace-20-rel-umls-embeddings.npz'


