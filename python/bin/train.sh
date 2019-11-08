#!/bin/bash

# pre-train discriminator
python -m python.eukg.train \
--mode=disc \
--model=transd \
--run_name=transd-disc-original \
--batch_size=1024 \
--no_semantic_network \
--data_dir=/users/max/data/artifacts/umls-embeddings \
--model_dir=/users/max/data/models/umls-embeddings \
--summaries_dir=/shared/hltdir4/disk1/max/logs

# pre-train generator
python -m python.eukg.train \
--mode=gen \
--model=distmult \
--run_name=dm-gen-original \
--no_semantic_network \
--learning_rate=1e-3 \
--batch_size=1024 \
--data_dir=/users/max/data/artifacts/umls-embeddings \
--model_dir=/users/max/data/models/umls-embeddings \
--summaries_dir=/shared/hltdir4/disk1/max/logs

# train full GAN
python -m python.eukg.train \
--mode=gan \
--model=transd \
--run_name=gan-original \
--no_semantic_network \
--dis_run_name=transd-disc-original \
--gen_run_name=dm-gen-original \
--batch_size=1024 \
--data_dir=/users/max/data/artifacts/umls-embeddings \
--model_dir=/users/max/data/models/umls-embeddings \
--summaries_dir=/shared/hltdir4/disk1/max/logs