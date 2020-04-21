
python -m python.eukg.data.create_triples \
--umls_dir=/shared/hltdir1/disk1/home/max/data/ontologies/umls_2019/2019AA-full/2019AA/ \
--output=/media/mdrive/umls-embeddings-ext


python -m python.eukg.data.create_lm_embeddings \
--batch_size=128 \
--data_dir=/users/max/data/artifacts/umls-embeddings \
--secondary_data_dir=/users/max/data/artifacts/umls-embeddings-compressed \
--encoder_checkpoint=/users/max/data/models/bert/uncased_L-12_H-768_A-12/bert_model.ckpt


python -m python.eukg.data.create_lm_embeddings \
--batch_size=256 \
--data_dir=/users/max/data/artifacts/umls-embeddings \
--secondary_data_dir=/media/mdrive/umls-embeddings-compressed \
--encoder_checkpoint=/users/max/data/models/bert/uncased_L-12_H-768_A-12/bert_model.ckpt

CUDA_VISIBLE_DEVICES=7 python -m python.eukg.data.create_triples \
--umls_dir=/home/maw150130/umls_2019/2019AA-full/2019AA/ \
--vocab_file=/home/maw150130/models/bert/biobert_v1.1_pubmed/vocab.txt \
--bert_config=/home/maw150130/models/bert/biobert_v1.1_pubmed/bert_config.json \
--encoder_checkpoint=/home/maw150130/models/bert/biobert_v1.1_pubmed/bert_model.ckpt \
--output=/home/maw150130/umls-embeddings
