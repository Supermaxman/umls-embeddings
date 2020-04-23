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
--run_name=transd-dm-disgen-ace-7 \
--ace_model \
--no_semantic_network \
--learning_rate=1e-5 \
--batch_size=16 \
--val_batch_size=16 \
--num_epochs=100 \
--data_dir=/users/max/data/artifacts/umls-embeddings \
--secondary_data_dir=/users/max/data/artifacts/umls-embeddings-compressed \
--model_dir=/users/max/data/models/umls-embeddings \
--summaries_dir=/shared/hltdir4/disk1/max/logs \
--num_workers=6 \
--buffer_size=1 \
--load=True



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
--secondary_data_dir=/run/media/max/ssd01/umls-embeddings-compressed \
--model_dir=/users/max/data/models/umls-embeddings \
--summaries_dir=/shared/hltdir4/disk1/max/logs \
--num_workers=2 \
--buffer_size=2


python -m python.eukg.train \
--mode=disgen \
--model=transd-distmult \
--run_name=transd-dm-disgen-ace-8 \
--ace_model \
--no_semantic_network \
--encoder_rnn_layers=2 \
--encoder_rnn_size=128 \
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
--load=False

python -m python.eukg.train \
--mode=disgen \
--model=transd-distmult \
--run_name=transd-dm-disgen-ace-9 \
--ace_model \
--no_semantic_network \
--encoder_rnn_layers=1 \
--encoder_rnn_size=512 \
--embedding_size=100 \
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
--load=False

python -m python.eukg.train \
--mode=disgen \
--model=transd-distmult \
--run_name=transd-dm-disgen-ace-10 \
--ace_model \
--no_semantic_network \
--encoder_rnn_layers=1 \
--encoder_rnn_size=512 \
--embedding_size=100 \
--gamma=0.4 \
--learning_rate=1e-5 \
--batch_size=16 \
--val_batch_size=16 \
--num_epochs=100 \
--data_dir=/users/max/data/artifacts/umls-embeddings \
--secondary_data_dir=/users/max/data/artifacts/umls-embeddings-compressed \
--model_dir=/users/max/data/models/umls-embeddings \
--summaries_dir=/shared/hltdir4/disk1/max/logs \
--num_workers=6 \
--buffer_size=1 \
--load=False

python -m python.eukg.train \
--mode=disgen \
--model=transd-distmult \
--run_name=transd-dm-disgen-ace-12 \
--ace_model \
--no_semantic_network \
--encoder_rnn_layers=1 \
--encoder_rnn_size=512 \
--embedding_size=100 \
--gamma=0.1 \
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
--load=False

python -m python.eukg.train \
--mode=disgen \
--model=transd-distmult \
--run_name=transd-dm-disgen-ace-17 \
--ace_model \
--no_semantic_network \
--encoder_rnn_layers=1 \
--encoder_rnn_size=512 \
--embedding_size=100 \
--gamma=0.5 \
--energy_norm_ord=2 \
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
--load=False

python -m python.eukg.train \
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
--load=False

python -m python.eukg.train \
--mode=disgen \
--model=transd-distmult \
--run_name=transd-dm-disgen-ace-19 \
--ace_model \
--no_semantic_network \
--encoder_rnn_layers=1 \
--encoder_rnn_size=512 \
--encoder_rnn_type=gru \
--embedding_size=100 \
--gamma=0.5 \
--energy_norm_ord=2 \
--learning_rate=1e-5 \
--batch_size=16 \
--val_batch_size=16 \
--num_epochs=100 \
--data_dir=/users/max/data/artifacts/umls-embeddings \
--secondary_data_dir=/users/max/data/artifacts/umls-embeddings-compressed \
--model_dir=/users/max/data/models/umls-embeddings \
--summaries_dir=/shared/hltdir4/disk1/max/logs \
--num_workers=6 \
--buffer_size=1 \
--load=False

python -m python.eukg.train \
--mode=disgen \
--model=transd-distmult \
--run_name=transd-dm-disgen-ace-20 \
--ace_model \
--no_semantic_network \
--encoder_rnn_layers=1 \
--encoder_rnn_size=512 \
--encoder_rnn_type=lstm \
--embedding_size=100 \
--gamma=0.5 \
--energy_norm_ord=2 \
--learning_rate=1e-5 \
--batch_size=16 \
--val_batch_size=16 \
--num_epochs=100 \
--data_dir=/users/max/data/artifacts/umls-embeddings \
--secondary_data_dir=/users/max/data/artifacts/umls-embeddings-compressed \
--model_dir=/users/max/data/models/umls-embeddings \
--summaries_dir=/shared/hltdir4/disk1/max/logs \
--num_workers=6 \
--buffer_size=1 \
--load=False

python -m python.eukg.train \
--mode=disgen \
--model=transd-distmult \
--run_name=transd-dm-disgen-ace-20 \
--ace_model \
--no_semantic_network \
--encoder_rnn_layers=1 \
--encoder_rnn_size=512 \
--encoder_rnn_type=lstm \
--embedding_size=100 \
--gamma=1.0 \
--energy_norm_ord=2 \
--learning_rate=1e-5 \
--batch_size=16 \
--val_batch_size=16 \
--num_epochs=100 \
--data_dir=/users/max/data/artifacts/umls-embeddings \
--secondary_data_dir=/users/max/data/artifacts/umls-embeddings-compressed \
--model_dir=/users/max/data/models/umls-embeddings \
--summaries_dir=/shared/hltdir4/disk1/max/logs \
--num_workers=6 \
--buffer_size=1 \
--load=True

python -m python.eukg.train \
--mode=gan-joint \
--model=transd-distmult \
--run_name=transd-dm-gan-joint-ace-20 \
--pre_run_name=transd-dm-disgen-ace-20 \
--ace_model \
--no_semantic_network \
--encoder_rnn_layers=1 \
--encoder_rnn_size=512 \
--encoder_rnn_type=lstm \
--embedding_size=100 \
--gamma=1.0 \
--energy_norm_ord=2 \
--learning_rate=1e-5 \
--batch_size=16 \
--val_batch_size=16 \
--num_epochs=100 \
--baseline_type=avg_prev_batch_momentum \
--baseline_momentum=0.99 \
--data_dir=/users/max/data/artifacts/umls-embeddings \
--secondary_data_dir=/users/max/data/artifacts/umls-embeddings-compressed \
--model_dir=/users/max/data/models/umls-embeddings \
--summaries_dir=/shared/hltdir4/disk1/max/logs \
--num_workers=6 \
--buffer_size=1 \
--load=True

python -m python.eukg.train \
--mode=gan-joint \
--model=transd-distmult \
--run_name=transd-dm-gan-joint-mod-1 \
--encoder_rnn_layers=1 \
--encoder_rnn_size=512 \
--encoder_rnn_type=lstm \
--embedding_size=100 \
--gamma=1.0 \
--learning_rate=1e-5 \
--batch_size=128 \
--val_batch_size=128 \
--num_epochs=100 \
--reward_type=neg_margin \
--baseline_type=avg_prev_batch_momentum \
--baseline_momentum=0.9 \
--data_dir=/users/max/data/artifacts/umls-embeddings \
--secondary_data_dir=/users/max/data/artifacts/umls-embeddings-compressed \
--model_dir=/users/max/data/models/umls-embeddings \
--summaries_dir=/shared/hltdir4/disk1/max/logs \
--load=False


python -m python.eukg.train \
--mode=gan-joint \
--model=transd-distmult \
--run_name=transd-dm-gan-joint-mod-2 \
--encoder_rnn_layers=1 \
--encoder_rnn_size=512 \
--encoder_rnn_type=lstm \
--embedding_size=100 \
--gamma=1.0 \
--learning_rate=1e-5 \
--batch_size=128 \
--val_batch_size=128 \
--num_epochs=100 \
--dis_loss_type=gen_and_uniform \
--reward_type=neg_margin \
--baseline_type=avg_prev_batch_momentum \
--baseline_momentum=0.9 \
--data_dir=/users/max/data/artifacts/umls-embeddings \
--secondary_data_dir=/users/max/data/artifacts/umls-embeddings-compressed \
--model_dir=/users/max/data/models/umls-embeddings \
--summaries_dir=/shared/hltdir4/disk1/max/logs \
--load=False


python -m python.eukg.train \
--mode=gan-joint \
--model=transd-distmult \
--run_name=transd-dm-gan-joint-mod-3 \
--encoder_rnn_layers=1 \
--encoder_rnn_size=512 \
--encoder_rnn_type=lstm \
--embedding_size=100 \
--gamma=1.0 \
--learning_rate=1e-5 \
--batch_size=16 \
--val_batch_size=16 \
--num_epochs=100 \
--dis_loss_type=gen_and_uniform \
--reward_type=neg_margin \
--baseline_type=avg_prev_batch_momentum \
--baseline_momentum=0.99 \
--data_dir=/users/max/data/artifacts/umls-embeddings \
--secondary_data_dir=/users/max/data/artifacts/umls-embeddings-compressed \
--model_dir=/users/max/data/models/umls-embeddings \
--summaries_dir=/shared/hltdir4/disk1/max/logs \
--load=False


python -m python.eukg.train \
--mode=gan-joint \
--model=rotate-distmult \
--run_name=rotate-dm-gan-joint-mod-5 \
--encoder_rnn_layers=1 \
--encoder_rnn_size=512 \
--encoder_rnn_type=lstm \
--embedding_size=100 \
--gamma=3.0 \
--learning_rate=1e-5 \
--batch_size=16 \
--val_batch_size=16 \
--num_epochs=100 \
--dis_loss_type=gen_and_uniform \
--reward_type=neg_margin \
--baseline_type=avg_prev_batch_momentum \
--baseline_momentum=0.99 \
--data_dir=/users/max/data/artifacts/umls-embeddings \
--secondary_data_dir=/users/max/data/artifacts/umls-embeddings-compressed \
--model_dir=/users/max/data/models/umls-embeddings \
--summaries_dir=/shared/hltdir4/disk1/max/logs \
--load=False

python -m python.eukg.train \
--mode=dis-self \
--model=rotate \
--run_name=rotate-dis-self-mod-6 \
--encoder_rnn_layers=1 \
--encoder_rnn_size=512 \
--encoder_rnn_type=lstm \
--embedding_size=100 \
--gamma=3.0 \
--learning_rate=1e-5 \
--batch_size=16 \
--val_batch_size=16 \
--num_epochs=100 \
--data_dir=/users/max/data/artifacts/umls-embeddings \
--secondary_data_dir=/users/max/data/artifacts/umls-embeddings-compressed \
--model_dir=/users/max/data/models/umls-embeddings \
--summaries_dir=/shared/hltdir4/disk1/max/logs \
--load=False
