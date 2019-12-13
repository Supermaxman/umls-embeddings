
python -m python.eukg.data.create_lm_embeddings \
--batch_size=128 \
--data_dir=/users/max/data/artifacts/umls-embeddings \
--secondary_data_dir=/max/umls \
--encoder_checkpoint=/users/max/data/models/bert/uncased_L-12_H-768_A-12/bert_model.ckpt