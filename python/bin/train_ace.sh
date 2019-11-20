#!/bin/bash

# pre-train dis and gen jointly
python -m python.eukg.train \
--mode=disgen \
--model=transd-distmult \
--run_name=transd-dm-disgen-ace-1 \
--ace_model \
--no_semantic_network \
--train_bert=False \
--learning_rate=1e-5 \
--batch_size=8 \
--num_generator_samples=4 \
--val_batch_size=32 \
--max_batches_per_epoch=200000 \
--num_epochs=100 \
--nrof_queued_batches=20 \
--nrof_queued_workers=1 \
--encoder_checkpoint=/users/max/data/models/bert/uncased_L-12_H-768_A-12/bert_model.ckpt \
--data_dir=/users/max/data/artifacts/umls-embeddings \
--model_dir=/users/max/data/models/umls-embeddings \
--summaries_dir=/shared/hltdir4/disk1/max/logs \
--load=True

# train full GAN
python -m python.eukg.train \
--mode=gan \
--model=transd-distmult \
--run_name=gan-ace-23 \
--pre_run_name=transd-dm-disgen-ace-1 \
--ace_model \
--no_semantic_network \
--train_bert=False \
--dis_learning_rate=1e-5 \
--gen_learning_rate=1e-4 \
--batch_size=8 \
--num_generator_samples=16 \
--val_batch_size=16 \
--max_batches_per_epoch=100000 \
--num_epochs=100 \
--nrof_queued_batches=20 \
--nrof_queued_workers=1 \
--data_dir=/users/max/data/artifacts/umls-embeddings \
--model_dir=/users/max/data/models/umls-embeddings \
--summaries_dir=/shared/hltdir4/disk1/max/logs


python -m python.eukg.train \
--mode=gan \
--model=transd-distmult \
--run_name=gan-ace-24 \
--pre_run_name=transd-dm-disgen-ace-1 \
--ace_model \
--no_semantic_network \
--train_bert=False \
--dis_learning_rate=1e-5 \
--gen_learning_rate=1e-4 \
--batch_size=4 \
--num_generator_samples=32 \
--val_batch_size=8 \
--max_batches_per_epoch=100000 \
--num_epochs=100 \
--nrof_queued_batches=20 \
--nrof_queued_workers=1 \
--data_dir=/users/max/data/artifacts/umls-embeddings \
--model_dir=/users/max/data/models/umls-embeddings \
--summaries_dir=/shared/hltdir4/disk1/max/logs


python -m python.eukg.train \
--mode=gan-joint \
--model=transd-distmult \
--run_name=gan-ace-26 \
--pre_run_name=transd-dm-disgen-ace-1 \
--ace_model \
--no_semantic_network \
--train_bert=False \
--dis_learning_rate=1e-5 \
--gen_learning_rate=1e-4 \
--batch_size=4 \
--num_generator_samples=32 \
--val_batch_size=8 \
--max_batches_per_epoch=100000 \
--num_epochs=100 \
--nrof_queued_batches=20 \
--nrof_queued_workers=1 \
--data_dir=/users/max/data/artifacts/umls-embeddings \
--model_dir=/users/max/data/models/umls-embeddings \
--summaries_dir=/shared/hltdir4/disk1/max/logs


python -m python.eukg.train \
--mode=gan-joint \
--model=transd-distmult \
--run_name=gan-ace-27 \
--pre_run_name=transd-dm-disgen-ace-1 \
--ace_model \
--no_semantic_network \
--train_bert=False \
--dis_learning_rate=1e-5 \
--gen_learning_rate=1e-4 \
--batch_size=8 \
--num_generator_samples=16 \
--val_batch_size=16 \
--max_batches_per_epoch=100000 \
--num_epochs=100 \
--nrof_queued_batches=20 \
--nrof_queued_workers=1 \
--data_dir=/users/max/data/artifacts/umls-embeddings \
--model_dir=/users/max/data/models/umls-embeddings \
--summaries_dir=/shared/hltdir4/disk1/max/logs

