#!/usr/bin/env bash



python -m python.eukg.train --mode=disgen --model=transd-distmult --run_name=transd-dm-disgen-ace-12 --ace_model --no_semantic_network --learning_rate=1e-5 --batch_size=32 --val_batch_size=32 --num_epochs=100 --data_dir=E:/umls-embeddings-compressed --secondary_data_dir=E:/umls-embeddings-compressed --model_dir=D:/Models/umls-embeddings/models --summaries_dir=D:/Logs/umls-embeddings/logs --num_workers=8 --buffer_size=4 --load=False
