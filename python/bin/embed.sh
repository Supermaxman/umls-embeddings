#!/usr/bin/env bash

python -m python.eukg.test.save_embeddings \
--mode=disgen \
--model=transd-distmult \
--run_name=transd-dm-disgen-ace-18 \
--ace_model \
--no_semantic_network \
--encoder_rnn_layers=1 \
--encoder_rnn_size=512 \
--encoder_rnn_type=gru \
--embedding_size=100 \
--gamma=0.5 \
--energy_norm_ord=1 \
--learning_rate=1e-5 \
--batch_size=16 \
--val_batch_size=16 \
--num_epochs=100 \
--data_dir=/users/max/data/artifacts/umls-embeddings \
--secondary_data_dir=/media/mdrive/umls-embeddings-compressed \
--model_dir=/users/max/data/models/umls-embeddings \
--summaries_dir=/shared/hltdir4/disk1/max/logs \
--num_workers=6 \
--buffer_size=1 \
--load=True