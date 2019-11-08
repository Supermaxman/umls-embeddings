#!/bin/bash

# pre-train discriminator
python -m python.eukg.train \
--mode=disc \
--model=transd \
--run_name=transd-disc-ace \
--batch_size=1024 \
--no_semantic_network \
--data_dir=/users/max/data/artifacts/umls-embeddings \
--model_dir=/users/max/data/models/umls-embeddings \
--summaries_dir=/shared/hltdir4/disk1/max/logs \
--ace_model \
--encoder_checkpoint=/users/max/data/models/bert/uncased_L-12_H-768_A-12/bert_model.ckpt

# pre-train generator
python -m python.eukg.train \
--mode=gen \
--model=distmult \
--run_name=dm-gen-ace \
--no_semantic_network \
--learning_rate=1e-3 \
--batch_size=1024 \
--data_dir=/users/max/data/artifacts/umls-embeddings \
--model_dir=/users/max/data/models/umls-embeddings \
--summaries_dir=/shared/hltdir4/disk1/max/logs \
--ace_model \
--encoder_checkpoint=#TODO

# train full GAN
python -m python.eukg.train \
--mode=gan \
--model=transd \
--run_name=gan-ace \
--no_semantic_network \
--dis_run_name=transd-disc-ace \
--gen_run_name=dm-gen-ace \
--batch_size=1024 \
--data_dir=/users/max/data/artifacts/umls-embeddings \
--model_dir=/users/max/data/models/umls-embeddings \
--summaries_dir=/shared/hltdir4/disk1/max/logs \
--ace_model \
--encoder_checkpoint=#TODO