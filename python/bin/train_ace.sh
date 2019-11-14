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
--run_name=transd-disc-ace-7 \
--batch_size=16 \
--learning_rate=1e-5 \
--no_semantic_network \
--data_dir=/users/max/data/artifacts/umls-embeddings \
--model_dir=/users/max/data/models/umls-embeddings \
--summaries_dir=/shared/hltdir4/disk1/max/logs \
--ace_model \
--train_bert=False \
--max_batches_per_epoch=50000 \
--num_epochs=100 \
--val_batch_size=32 \
--encoder_checkpoint=/users/max/data/models/bert/uncased_L-12_H-768_A-12/bert_model.ckpt \
--load=False

# pre-train generator
python -m python.eukg.train \
--mode=gen \
--model=distmult \
--run_name=dm-gen-ace-3 \
--no_semantic_network \
--learning_rate=1e-5 \
--batch_size=4 \
--data_dir=/users/max/data/artifacts/umls-embeddings \
--model_dir=/users/max/data/models/umls-embeddings \
--summaries_dir=/shared/hltdir4/disk1/max/logs \
--ace_model \
--train_bert=False \
--max_batches_per_epoch=50000 \
--num_epochs=100 \
--val_batch_size=4 \
--num_generator_samples=10 \
--nrof_queued_batches=10 \
--encoder_checkpoint=/users/max/data/models/umls-embeddings/transd/transd-disc-ace-7/transd-200000 \
--load=False

# pre-train generator (larger bsize test)
#python -m python.eukg.train \
#--mode=gen \
#--model=distmult \
#--run_name=dm-gen-ace-4 \
#--no_semantic_network \
#--learning_rate=1e-5 \
#--batch_size=8 \
#--data_dir=/users/max/data/artifacts/umls-embeddings \
#--model_dir=/users/max/data/models/umls-embeddings \
#--summaries_dir=/shared/hltdir4/disk1/max/logs \
#--ace_model \
#--train_bert=False \
#--max_batches_per_epoch=50000 \
#--num_epochs=100 \
#--val_batch_size=8 \
#--num_generator_samples=10 \
#--nrof_queued_batches=10 \
#--encoder_checkpoint=/users/max/data/models/umls-embeddings/transd/transd-disc-ace-7/transd-200000 \
#--load=False


# train full GAN
python -m python.eukg.train \
--mode=gan \
--model=transd \
--run_name=gan-ace \
--no_semantic_network \
--dis_run_name=transd-disc-ace \
--gen_run_name=dm-gen-ace \
--learning_rate=1e-4 \
--batch_size=32 \
--data_dir=/users/max/data/artifacts/umls-embeddings \
--model_dir=/users/max/data/models/umls-embeddings \
--summaries_dir=/shared/hltdir4/disk1/max/logs \
--ace_model \
--encoder_checkpoint=#TODO