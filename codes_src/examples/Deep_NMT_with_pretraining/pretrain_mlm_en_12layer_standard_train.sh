#!/usr/bin/bash
export CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'
MAIN_DIR=../../fairseq-master-underDevelop
#DATA_DIR=/dockerdata/zieenyang/corpus/ende_wmt16/
DATA_DIR=/dockerdata/zieenyang/corpus/data_fairseq/ende_wmt16/
PROCESSED_DIR=$DATA_DIR/processed_only_para

python $MAIN_DIR/train_new.py \
  --task cross_lingual_lm $PROCESSED_DIR \
  --save-dir en-mlm-checkpoints-12layers-0.0007-12000-60000-only-para/ \
  --max-update 60000 \
  --save-interval 1 \
  --no-epoch-checkpoints \
  --arch mlm_base_yue \
  --dropout 0.1 \
  --criterion legacy_masked_lm_loss \
  --max-tokens 6000 \
  --tokens-per-sample 256 \
  --attention-dropout 0.1 \
  --seed 0 \
  --masked-lm-only \
  --monolingual-langs 'en' \
  --num-segment 0 \
  --keep-last-epochs 5 \
  --log-interval 1 \
  --keep-interval-updates 5 \
  --save-interval-updates 1000 \
  --ddp-backend=no_c10d \
  --update-freq 16 \
  --optimizer adam \
  --adam-betas '(0.9,0.98)' \
  --adam-eps 1e-6 \
  --lr 0.0007 \
  --encoder-layers 12 \
  --lr-scheduler polynomial_decay \
  --weight-decay 0.01 \
  --warmup-updates 12000 \
  --total-num-update 60000 \
  --clip-norm 0.0 \
  --shuffle
