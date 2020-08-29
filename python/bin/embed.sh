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

python -m python.eukg.data.embed_rels \
  --relid2txt='/users/max/data/corpora/ChemProt/original/relid2txt_umls_4.json' \
  --out_file='/home/max/data/artifacts/chemprot/transd-dm-gan-joint-ace-20-rel-umls-4-embeddings.npz'

python -m python.eukg.data.embed_rels \
  --relid2txt='/users/max/data/corpora/ChemProt/original/relid2txt_umls_5.json' \
  --out_file='/home/max/data/artifacts/chemprot/transd-dm-gan-joint-ace-20-rel-umls-5-embeddings.npz'


python -m python.eukg.data.embed \
  --ctxt2id_file='/home/max/data/corpora/ddi2013-type/original/ctxt2id.json' \
  --out_file='/home/max/data/artifacts/ddi2013-type/transd-dm-gan-joint-ace-20-embeddings.npz'

python -m python.eukg.data.embed_rels \
  --relid2txt='/users/max/data/corpora/ddi2013-type/original/relid2txt_umls.json' \
  --out_file='/home/max/data/artifacts/ddi2013-type/transd-dm-gan-joint-ace-20-rel-umls-embeddings.npz'

#
python -m python.eukg.data.embed \
  --ctxt2id_file='/users/rmm120030/working/kge_ner/info/clef_mentions.json' \
  --out_file='/users/rmm120030/working/kge_ner/info/clef_mentions.npz'

python -m python.eukg.data.embed \
  --ctxt2id_file='/users/rmm120030/working/kge_ner/info/medmentions_mentions.json' \
  --out_file='/users/rmm120030/working/kge_ner/info/medmentions_mentions.npz'

python -m python.eukg.data.embed \
  --ctxt2id_file='/users/rmm120030/working/kge_ner/info/semtype_text.json' \
  --out_file='/users/rmm120030/working/kge_ner/info/semtype_text.npz'


python -m python.eukg.data.embed_rels \
  --relid2txt='/users/max/data/corpora/ddi2013-type/original/relid2txt_umls_2.json' \
  --out_file='/home/max/data/artifacts/ddi2013-type/transd-dm-gan-joint-ace-20-rel-umls-2-embeddings.npz'

python -m python.eukg.data.embed_rels \
  --relid2txt='/users/max/data/corpora/ddi2013-type/original/relid2txt_umls_3.json' \
  --out_file='/home/max/data/artifacts/ddi2013-type/transd-dm-gan-joint-ace-20-rel-umls-3-embeddings.npz'


python -m python.eukg.data.embed \
  --ctxt2id_file='/users/rmm120030/working/kge_ner/model/medmentions/pm_bilstm/test/text2id.json' \
  --out_file='/users/rmm120030/working/kge_ner/model/medmentions/pm_bilstm/test/text_embeddings.npz' \
  --batch_size=32

python -m python.eukg.data.embed \
  --ctxt2id_file='/users/rmm120030/working/kge_ner/model/medmentions/pm_bilstm/dev/text2id.json' \
  --out_file='/users/rmm120030/working/kge_ner/model/medmentions/pm_bilstm/dev/text_embeddings2.npz' \
  --batch_size=32



python -m python.eukg.data.top_k_gpu \
  --batch_size=1


python -m python.eukg.data.embed \
  --ctxt2id_file='/users/max/data/corpora/nary-rel/tf-0/ctxt2id.json' \
  --out_file='/users/max/data/corpora/nary-rel/tf-0/transd-dm-gan-joint-ace-20-embeddings.npz' \
  --batch_size=16


# rotate

python -m python.eukg.data.embed \
  --ctxt2id_file='/home/max/data/artifacts/i2b2/2010/new_data/ctxt2id.json' \
  --ace_path='/users/max/data/models/umls-embeddings/rotate/rotate-dis-self-mod-8' \
  --out_file='/home/max/data/artifacts/i2b2/2010/new_data/rotate-dis-self-mod-8-embeddings.npz' \
  --embedding_size=256 \
  --batch_size=64

python -m python.eukg.data.embed_rels \
  --relid2txt='/home/max/data/artifacts/i2b2/2010/new_data/relid2txt_umls.json' \
  --ace_path='/users/max/data/models/umls-embeddings/rotate/rotate-dis-self-mod-8' \
  --out_file='/home/max/data/artifacts/i2b2/2010/new_data/rotate-dis-self-mod-8-rel-umls-embeddings.npz' \
  --embedding_size=256

python -m python.eukg.data.embed \
  --ctxt2id_file='/users/max/code/extract_COVID19_events_from_Twitter/data/can_not_test_ctxt2id.json' \
  --out_file='/users/max/code/extract_COVID19_events_from_Twitter/data/can_not_test_embs.npz'


# covid19 twitter
python -m python.eukg.data.embed \
  --ctxt2id_file='/users/max/code/extract_COVID19_events_from_Twitter/data/can_not_test_ctxt2id.json' \
  --out_file='/users/max/code/extract_COVID19_events_from_Twitter/data/can_not_test_embs.npz'
python -m python.eukg.data.embed \
  --ctxt2id_file='/users/max/code/extract_COVID19_events_from_Twitter/data/death_ctxt2id.json' \
  --out_file='/users/max/code/extract_COVID19_events_from_Twitter/data/death_embs.npz'
python -m python.eukg.data.embed \
  --ctxt2id_file='/users/max/code/extract_COVID19_events_from_Twitter/data/test_negative_ctxt2id.json' \
  --out_file='/users/max/code/extract_COVID19_events_from_Twitter/data/test_negative_embs.npz'
python -m python.eukg.data.embed \
  --ctxt2id_file='/users/max/code/extract_COVID19_events_from_Twitter/data/test_positive_ctxt2id.json' \
  --out_file='/users/max/code/extract_COVID19_events_from_Twitter/data/test_positive_embs.npz'
python -m python.eukg.data.embed \
  --ctxt2id_file='/users/max/code/extract_COVID19_events_from_Twitter/data/cure_and_prevention_ctxt2id.json' \
  --out_file='/users/max/code/extract_COVID19_events_from_Twitter/data/cure_and_prevention_embs.npz'

/users/max/code/extract_COVID19_events_from_Twitter/data/test_negative_ctxt2id.json
/users/max/code/extract_COVID19_events_from_Twitter/data/test_positive_ctxt2id.json
/users/max/code/extract_COVID19_events_from_Twitter/data/cure_and_prevention_ctxt2id.json

