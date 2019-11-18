#!/bin/bash

# pre-train discriminator
# epochs take much longer, so only one full epoch of pre-training
# one epoch of pre-training will be 100 epochs of 10000 batches
#batch_size=16
#learning_rate=1e-5
#max_batches_per_epoch=5000
#num_epochs=100
#val_batch_size=512
python -m python.eukg.train \
--mode=disc \
--model=transd \
--run_name=transd-disc-ace-10 \
--batch_size=16 \
--learning_rate=1e-5 \
--no_semantic_network \
--data_dir=/users/max/data/artifacts/umls-embeddings \
--model_dir=/users/max/data/models/umls-embeddings \
--summaries_dir=/shared/hltdir4/disk1/max/logs \
--ace_model \
--train_bert=False \
--max_batches_per_epoch=150000 \
--num_epochs=100 \
--val_batch_size=32 \
--nrof_queued_batches=20 \
--nrof_queued_workers=1 \
--encoder_checkpoint=/users/max/data/models/bert/uncased_L-12_H-768_A-12/bert_model.ckpt \
--load=True

# pre-train generator
python -m python.eukg.train \
--mode=gen \
--model=distmult \
--run_name=dm-gen-ace-14 \
--no_semantic_network \
--learning_rate=1e-5 \
--batch_size=16 \
--data_dir=/users/max/data/artifacts/umls-embeddings \
--model_dir=/users/max/data/models/umls-embeddings \
--summaries_dir=/shared/hltdir4/disk1/max/logs \
--ace_model \
--train_bert=False \
--max_batches_per_epoch=150000 \
--num_epochs=100 \
--val_batch_size=32 \
--num_generator_samples=4 \
--nrof_queued_batches=20 \
--nrof_queued_workers=1 \
--encoder_checkpoint=/users/max/data/models/umls-embeddings/transd/transd-disc-ace-10/transd-250000 \
--load=False


# pre-train dis and gen jointly
python -m python.eukg.train \
--mode=disgen \
--model=transd-distmult \
--run_name=transd-dm-disgen-ace-1 \
--no_semantic_network \
--learning_rate=1e-5 \
--batch_size=8 \
--data_dir=/users/max/data/artifacts/umls-embeddings \
--model_dir=/users/max/data/models/umls-embeddings \
--summaries_dir=/shared/hltdir4/disk1/max/logs \
--ace_model \
--train_bert=False \
--max_batches_per_epoch=100000 \
--num_epochs=100 \
--val_batch_size=32 \
--num_generator_samples=4 \
--nrof_queued_batches=20 \
--nrof_queued_workers=1 \
--encoder_checkpoint=/users/max/data/models/bert/uncased_L-12_H-768_A-12/bert_model.ckpt \
--load=False

# train full GAN
python -m python.eukg.train \
--mode=gan \
--model=transd-distmult \
--run_name=gan-ace-1 \
--no_semantic_network \
--pre_run_name=transd-dm-disgen-ace-1 \
--learning_rate=1e-4 \
--batch_size=8 \
--data_dir=/users/max/data/artifacts/umls-embeddings \
--model_dir=/users/max/data/models/umls-embeddings \
--summaries_dir=/shared/hltdir4/disk1/max/logs \
--ace_model \
--train_bert=False \
--max_batches_per_epoch=100000 \
--num_epochs=100 \
--val_batch_size=32 \
--num_generator_samples=8 \
--nrof_queued_batches=10
