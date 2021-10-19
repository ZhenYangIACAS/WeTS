#!/usr/bin/bash

# The second-phase pre-training for En2cn

workspace=path_to_WeTS
save_dir=$workspace/models
codes_dir=$workspace/codes_src
data_dir=$workspace/data-bin

seed=1111
max_tokens=8192
update_freq=1
dropout=0.1
attention_heads=8
embed_dim=512
ffn_embed_dim=2048
encoder_layers=6
decoder_layers=6

mkdir -p $save_dir

export CUDA_VISIBLE_DEVICES='0'

python $codes_dir/train.py $data_dir \
    --task input_suggestion \
    --arch istransformer \
    --source-lang 'en' \
    --target-lang 'cn' \
    --save-dir $save_dir \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr-scheduler inverse_sqrt --lr 0.0006 --min-lr 1e-09 --max-update 200200 \
    --warmup-updates 8000 --warmup-init-lr '1e-07' \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --seed ${seed} \
    --update-freq ${update_freq} --max-tokens ${max_tokens} \
    --dropout ${dropout} --relu-dropout 0.1 --attention-dropout 0.1 \
    --decoder-attention-heads ${attention_heads} --encoder-attention-heads ${attention_heads} \
    --decoder-embed-dim ${embed_dim} --encoder-embed-dim ${embed_dim} \
    --decoder-ffn-embed-dim ${ffn_embed_dim} --encoder-ffn-embed-dim ${ffn_embed_dim} \
    --encoder-layers ${encoder_layers} --decoder-layers ${decoder_layers} \
    --log-format simple  --log-interval 500 \
    --left-pad-source True --left-pad-target False \
    --keep-interval-updates 10 --save-interval-updates 3000 \
    --ddp-backend=no_c10d \
    --fp16 \
    >$save_dir/train_finetuning.log 2>&1
