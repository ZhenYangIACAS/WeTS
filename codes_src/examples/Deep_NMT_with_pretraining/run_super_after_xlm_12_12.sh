#!/usr/bin/bash
export CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'
data_dir=/dockerdata/zieenyang/corpus/data_fairseq/ende_wmt16/processed_only_para
# data_dir=/dockerdata/zieenyang/corpus/data_fairseq/zh-uy_pretrain/processed_only_para
save_dir=./en-mlm-checkpoints-12layers-0.0007-12000-60000-only-para/super-xlm-finetune-encoder-12-decoder-12-deep-init

main_dir=../../fairseq-master-underDevelop

seed=1234
max_tokens=4096
update_freq=1
dropout=0.1
attention_heads=8
embed_dim=512
ffn_embed_dim=2048
encoder_layers=12
decoder_layers=12
encoder_pretrained_checkpoints='en:./en-mlm-checkpoints-12layers-0.0007-12000-60000-only-para/checkpoint_best.pt'

mkdir -p $save_dir

python $main_dir/train.py $data_dir \
    --task xmasked_seq2seq \
	--source-langs en \
	--target-langs de \
    --langs en,de \
    --valid-lang-pairs en-de \
	--arch xtransformer \
    --mt_steps en-de \
    --encoder-pretrained-checkpoints ${encoder_pretrained_checkpoints} \
    --save-dir $save_dir \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr-scheduler inverse_sqrt --lr 0.0005 --min-lr 1e-09 \
    --warmup-updates 4000 \
    --warmup-init-lr '1e-07' \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --seed ${seed} \
    --max-tokens ${max_tokens} --update-freq ${update_freq} \
    --dropout ${dropout} --relu-dropout 0.1 --attention-dropout 0.1 \
    --decoder-attention-heads ${attention_heads} --encoder-attention-heads ${attention_heads} \
    --decoder-embed-dim ${embed_dim} --encoder-embed-dim ${embed_dim} \
    --decoder-ffn-embed-dim ${ffn_embed_dim} --encoder-ffn-embed-dim ${ffn_embed_dim} \
    --encoder-layers ${encoder_layers} --decoder-layers ${decoder_layers} \
    --max-update 100000000 --max-epoch 300 \
    --keep-interval-updates 5 --save-interval-updates 3000  --log-interval 1000 \
    --keep-last-epochs 5 \
    --share-decoder-input-output-embed \
	--ddp-backend=no_c10d \
    --deep-init
