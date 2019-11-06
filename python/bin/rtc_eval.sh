#!/bin/bash
# transe
python -m eukg.test.classification --mode=disc --model=transe --run_name=transe-disc
# transd
python -m eukg.test.classification --mode=disc --model=transd --run_name=transd-disc_e0init --embedding_size=100
# transe-sn
python -m eukg.test.classification --mode=disc --model=transe --run_name=transe-sn-disc
# transd-sn
python -m eukg.test.classification --mode=disc --model=transd --run_name=transd-sn-disc_e0init --embedding_size=100
# transe-sn gan
python -m eukg.test.classification --mode=gan --model=transe --run_name=sn-gan --dis_run_name=transe-sn-disc
# transd-sn gan
python -m eukg.test.classification --mode=gan --model=transd --run_name=gan --dis_run_name=transd-sn-disc_e0init --embedding_size=100
################ SN ##################
# transe-sn
python -m eukg.test.classification --mode=disc --model=transe --run_name=transe-sn-disc --sn_eval --eval_mode=sn
# transd-sn
python -m eukg.test.classification --mode=disc --model=transd --run_name=transd-sn-disc_e0init --embedding_size=100 --sn_eval --eval_mode=sn
# transe-sn gan
python -m eukg.test.classification --mode=gan --model=transe --run_name=sn-gan --dis_run_name=transe-sn-disc --sn_eval --eval_mode=sn
# transd-sn gan
python -m eukg.test.classification --mode=gan --model=transd --run_name=gan --dis_run_name=transd-sn-disc_e0init --embedding_size=100 --sn_eval --eval_mode=sn
