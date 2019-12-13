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

python -m python.eukg.train \
--mode=gan-joint \
--model=transd-distmult \
--run_name=gan-ace-50 \
--pre_run_name=transd-dm-disgen-ace-1 \
--ace_model \
--no_semantic_network \
--train_bert=False \
--learning_rate=1e-5 \
--batch_size=4 \
--num_generator_samples=32 \
--val_batch_size=8 \
--baseline_type=avg_prev_batch_momentum \
--baseline_momentum=0.99 \
--max_batches_per_epoch=300000 \
--num_epochs=100 \
--data_dir=/users/max/data/artifacts/umls-embeddings \
--model_dir=/users/max/data/models/umls-embeddings \
--summaries_dir=/shared/hltdir4/disk1/max/logs

python -m python.eukg.train \
--mode=gan-joint \
--model=transd-distmult \
--run_name=gan-ace-1 \
--pre_run_name=transd-dm-disgen-ace-1 \
--ace_model \
--no_semantic_network \
--train_bert=False \
--learning_rate=1e-4 \
--batch_size=8 \
--num_generator_samples=32 \
--val_batch_size=8 \
--baseline_type=avg_prev_batch_momentum \
--baseline_momentum=0.99 \
--max_batches_per_epoch=100000 \
--num_epochs=100 \
--data_dir=/users/max/data/artifacts/umls-embeddings \
--model_dir=/users/max/data/models/umls-embeddings \
--summaries_dir=/shared/hltdir4/disk1/max/logs



# pre-train dis and gen jointly with new data setup
python -m python.eukg.train \
--mode=disgen \
--model=transd-distmult \
--run_name=transd-dm-disgen-ace-5 \
--ace_model \
--no_semantic_network \
--learning_rate=1e-5 \
--batch_size=8 \
--num_generator_samples=4 \
--val_batch_size=32 \
--num_epochs=100 \
--data_dir=/users/max/data/artifacts/umls-embeddings \
--model_dir=/users/max/data/models/umls-embeddings \
--summaries_dir=/shared/hltdir4/disk1/max/logs \
--load=False


python -m python.eukg.train \
--mode=disgen \
--model=transd-distmult \
--run_name=transd-dm-disgen-ace-6 \
--ace_model \
--no_semantic_network \
--learning_rate=1e-5 \
--batch_size=8 \
--num_generator_samples=4 \
--val_batch_size=32 \
--num_epochs=100 \
--data_dir=/users/max/data/artifacts/umls-embeddings \
--secondary_data_dir=/users/max/data/artifacts/umls-embeddings-compressed \
--model_dir=/users/max/data/models/umls-embeddings \
--summaries_dir=/shared/hltdir4/disk1/max/logs \
--num_workers=6 \
--buffer_size=1 \
--load=False



python -m python.eukg.train \
--mode=gan-joint \
--model=transd-distmult \
--run_name=gan-ace-3 \
--pre_run_name=transd-dm-disgen-ace-5 \
--ace_model \
--no_semantic_network \
--train_bert=False \
--learning_rate=1e-5 \
--batch_size=8 \
--num_generator_samples=32 \
--val_batch_size=8 \
--baseline_type=avg_prev_batch_momentum \
--baseline_momentum=0.99 \
--num_epochs=100 \
--data_dir=/users/max/data/artifacts/umls-embeddings \
--secondary_data_dir=/users/max/data/artifacts/umls-embeddings-compressed \
--model_dir=/users/max/data/models/umls-embeddings \
--summaries_dir=/shared/hltdir4/disk1/max/logs \
--num_workers=2 \
--buffer_size=2

