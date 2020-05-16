
python -m python.eukg.data.create_triples \
--umls_dir=/shared/hltdir1/disk1/home/max/data/ontologies/umls_2019/2019AA-full/2019AA/ \
--output=/users/max/data/artifacts/umls-embeddings-ext


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


python -m python.eukg.data.create_triples \
--umls_dir=/shared/hltdir1/disk1/home/max/data/ontologies/umls_2019/2019AA-full/2019AA/ \
--output=/users/max/data/artifacts/umls-embeddings-lm