#!/usr/bin/env bash



python -m python.eukg.test.save_embeddings \
--mode=disgen \
--model=transd-distmult \
--run_name=transd-dm-disgen-ace-7 \
--ace_model \
--no_semantic_network \
--train_bert=False \
--batch_size=32 \
--data_dir=/users/max/data/artifacts/umls-embeddings \
--secondary_data_dir=/users/max/data/artifacts/umls-embeddings-compressed \
--model_dir=/users/max/data/models/umls-embeddings \
--summaries_dir=/shared/hltdir4/disk1/max/logs \
--eval_mode=save \
--eval_dir=/users/max/data/artifacts/umls-embeddings \
--load=True \
--num_workers=6 \
--buffer_size=1

python -m python.eukg.test.ranking_evals \
--mode=disgen \
--model=transd-distmult \
--run_name=transd-dm-disgen-ace-7 \
--ace_model \
--no_semantic_network \
--batch_size=32 \
--data_dir=/users/max/data/artifacts/umls-embeddings \
--secondary_data_dir=/users/max/data/artifacts/umls-embeddings-compressed \
--model_dir=/users/max/data/models/umls-embeddings \
--summaries_dir=/shared/hltdir4/disk1/max/logs \
--eval_mode=save \
--eval_dir=/users/max/data/artifacts/umls-embeddings \
--load=False \
--load_embeddings=True \
--embedding_file=/users/max/data/artifacts/umls-embeddings/transd-dm-disgen-ace-7/test_embeddings.npz \
--num_workers=6 \
--buffer_size=1


python -m python.eukg.test.ranking_evals \
--mode=disgen \
--model=transd-distmult \
--run_name=transd-dm-disgen-ace-1 \
--ace_model \
--no_semantic_network \
--train_bert=False \
--learning_rate=1e-5 \
--batch_size=1310720 \
--num_shards=2 \
--shard=2 \
--num_eval_threads=2 \
--num_generator_samples=1 \
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

python -m python.eukg.test.ppa \
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

python -m python.eukg.test.ranking_evals \
--mode=gan-joint \
--model=transd-distmult \
--run_name=gan-ace-50 \
--ace_model \
--no_semantic_network \
--train_bert=False \
--learning_rate=1e-5 \
--batch_size=1024 \
--num_generator_samples=1 \
--val_batch_size=1024 \
--baseline_type=avg_prev_batch_momentum \
--baseline_momentum=0.99 \
--data_dir=/users/max/data/artifacts/umls-embeddings \
--model_dir=/users/max/data/models/umls-embeddings \
--summaries_dir=/shared/hltdir4/disk1/max/logs \
--eval_mode=save \
--eval_dir=/users/max/data/artifacts/umls-embeddings \ \
--load=True

python -m python.eukg.test.classification \
--mode=gan-joint \
--model=transd-distmult \
--run_name=gan-ace-50 \
--ace_model \
--no_semantic_network \
--train_bert=False \
--learning_rate=1e-5 \
--batch_size=64 \
--num_generator_samples=1 \
--val_batch_size=64 \
--baseline_type=avg_prev_batch_momentum \
--baseline_momentum=0.99 \
--data_dir=/users/max/data/artifacts/umls-embeddings \
--model_dir=/users/max/data/models/umls-embeddings \
--summaries_dir=/shared/hltdir4/disk1/max/logs \
--eval_dir=/users/max/data/artifacts/umls-embeddings \
--load=True

python -m python.eukg.test.ppa \
--mode=gan-joint \
--model=transd-distmult \
--run_name=gan-ace-50 \
--ace_model \
--no_semantic_network \
--train_bert=False \
--learning_rate=1e-5 \
--batch_size=64 \
--num_generator_samples=1 \
--val_batch_size=64 \
--baseline_type=avg_prev_batch_momentum \
--baseline_momentum=0.99 \
--data_dir=/users/max/data/artifacts/umls-embeddings \
--model_dir=/users/max/data/models/umls-embeddings \
--summaries_dir=/shared/hltdir4/disk1/max/logs \
--eval_dir=/users/max/data/artifacts/umls-embeddings \
--load=True