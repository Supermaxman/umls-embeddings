#!/usr/bin/env bash

python -m python.eukg.test.ranking_evals \
--mode=disgen \
--model=transd-distmult \
--run_name=transd-dm-disgen-ace-1 \
--ace_model \
--no_semantic_network \
--train_bert=False \
--learning_rate=1e-5 \
--batch_size=64 \
--num_generator_samples=1 \
--val_batch_size=64 \
--data_dir=/users/max/data/artifacts/umls-embeddings \
--model_dir=/users/max/data/models/umls-embeddings \
--summaries_dir=/shared/hltdir4/disk1/max/logs \
--eval_mode=save \
--eval_dir=/users/max/data/artifacts/umls-embeddings \
--load=False \
--pre_run_name=transd-dm-disgen-ace-1


python -m python.eukg.test.classification \
--mode=disgen \
--model=transd-distmult \
--run_name=transd-dm-disgen-ace-1 \
--ace_model \
--no_semantic_network \
--train_bert=False \
--learning_rate=1e-5 \
--batch_size=64 \
--num_generator_samples=1 \
--val_batch_size=64 \
--data_dir=/users/max/data/artifacts/umls-embeddings \
--model_dir=/users/max/data/models/umls-embeddings \
--summaries_dir=/shared/hltdir4/disk1/max/logs \
--eval_mode=save \
--eval_dir=/users/max/data/artifacts/umls-embeddings \
--load=False \
--pre_run_name=transd-dm-disgen-ace-1