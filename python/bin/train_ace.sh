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
--mode=gan-joint \
--model=transd-distmult \
--run_name=gan-ace-36 \
--pre_run_name=transd-dm-disgen-ace-1 \
--ace_model \
--no_semantic_network \
--train_bert=False \
--dis_learning_rate=1e-5 \
--gen_learning_rate=1e-4 \
--batch_size=4 \
--num_generator_samples=32 \
--val_batch_size=16 \
--max_batches_per_epoch=200000 \
--num_epochs=100 \
--data_dir=/users/max/data/artifacts/umls-embeddings \
--model_dir=/users/max/data/models/umls-embeddings \
--summaries_dir=/shared/hltdir4/disk1/max/logs


python -m python.eukg.train \
--mode=gan-joint \
--model=transd-distmult \
--run_name=gan-ace-39 \
--pre_run_name=transd-dm-disgen-ace-1 \
--ace_model \
--no_semantic_network \
--train_bert=False \
--dis_learning_rate=1e-6 \
--gen_learning_rate=1e-5 \
--batch_size=16 \
--num_generator_samples=8 \
--val_batch_size=32 \
--max_batches_per_epoch=200000 \
--num_epochs=100 \
--data_dir=/users/max/data/artifacts/umls-embeddings \
--model_dir=/users/max/data/models/umls-embeddings \
--summaries_dir=/shared/hltdir4/disk1/max/logs


python -m python.eukg.train \
--mode=gan-joint \
--model=transd-distmult \
--run_name=gan-ace-40 \
--pre_run_name=transd-dm-disgen-ace-1 \
--ace_model \
--no_semantic_network \
--train_bert=False \
--dis_learning_rate=1e-6 \
--gen_learning_rate=1e-5 \
--batch_size=4 \
--num_generator_samples=32 \
--val_batch_size=8 \
--baseline_type=avg_prev_batch_momentum \
--baseline_momentum=0.99 \
--max_batches_per_epoch=200000 \
--num_epochs=100 \
--data_dir=/users/max/data/artifacts/umls-embeddings \
--model_dir=/users/max/data/models/umls-embeddings \
--summaries_dir=/shared/hltdir4/disk1/max/logs


python -m python.eukg.train \
--mode=gan-joint \
--model=transd-distmult \
--run_name=gan-ace-41 \
--pre_run_name=transd-dm-disgen-ace-1 \
--ace_model \
--no_semantic_network \
--train_bert=False \
--dis_learning_rate=1e-5 \
--gen_learning_rate=1e-4 \
--batch_size=4 \
--num_generator_samples=32 \
--val_batch_size=8 \
--baseline_type=avg_prev_batch_momentum \
--baseline_momentum=0.99 \
--max_batches_per_epoch=200000 \
--num_epochs=100 \
--data_dir=/users/max/data/artifacts/umls-embeddings \
--model_dir=/users/max/data/models/umls-embeddings \
--summaries_dir=/shared/hltdir4/disk1/max/logs


python -m python.eukg.train \
--mode=gan-joint \
--model=transd-distmult \
--run_name=gan-ace-42 \
--pre_run_name=transd-dm-disgen-ace-1 \
--ace_model \
--no_semantic_network \
--train_bert=False \
--dis_learning_rate=1e-6 \
--gen_learning_rate=1e-5 \
--batch_size=1 \
--num_generator_samples=100 \
--val_batch_size=4 \
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
--run_name=gan-ace-45 \
--pre_run_name=transd-dm-disgen-ace-1 \
--ace_model \
--no_semantic_network \
--train_bert=False \
--dis_learning_rate=1e-5 \
--gen_learning_rate=1e-4 \
--batch_size=8 \
--num_generator_samples=32 \
--val_batch_size=16 \
--baseline_type=avg_prev_batch_momentum \
--baseline_momentum=0.9 \
--max_batches_per_epoch=300000 \
--num_epochs=100 \
--data_dir=/users/max/data/artifacts/umls-embeddings \
--model_dir=/users/max/data/models/umls-embeddings \
--summaries_dir=/shared/hltdir4/disk1/max/logs




python -m python.eukg.train \
--mode=gan-joint \
--model=transd-distmult \
--run_name=gan-ace-48 \
--pre_run_name=transd-dm-disgen-ace-1 \
--ace_model \
--no_semantic_network \
--train_bert=False \
--dis_learning_rate=1e-7 \
--gen_learning_rate=1e-6 \
--batch_size=1 \
--num_generator_samples=64 \
--val_batch_size=4 \
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
--run_name=gan-ace-47 \
--pre_run_name=transd-dm-disgen-ace-1 \
--ace_model \
--no_semantic_network \
--train_bert=False \
--dis_learning_rate=1e-6 \
--gen_learning_rate=1e-5 \
--batch_size=1 \
--num_generator_samples=64 \
--val_batch_size=2 \
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
--run_name=gan-ace-49 \
--pre_run_name=transd-dm-disgen-ace-1 \
--ace_model \
--no_semantic_network \
--train_bert=False \
--dis_learning_rate=1e-5 \
--gen_learning_rate=1e-4 \
--batch_size=8 \
--num_generator_samples=32 \
--val_batch_size=16 \
--baseline_type=avg_prev_batch_momentum \
--baseline_momentum=0.99 \
--max_batches_per_epoch=200000 \
--num_epochs=100 \
--data_dir=/users/max/data/artifacts/umls-embeddings \
--model_dir=/users/max/data/models/umls-embeddings \
--summaries_dir=/shared/hltdir4/disk1/max/logs