#!/bin/bash -v

main_dir=../../fairseq-master-underDevelop
data_dir=/dockerdata/zieenyang/corpus/data_fairseq/ende_wmt16
mono_dir=$data_dir/mono_50M
processed=$data_dir/processed_mono_50M
sub_word_tool=../../../tools/subword-nmt-master

mkdir -p $processed

# generate monolingual data
for lg in en de
do 
    python $sub_word_tool/apply_bpe.py -c $data_dir/bpe.32000 <$mono_dir/${lg}/all.${lg}.tok.escape >$mono_dir/${lg}/all.tok.escape.32000.bpe.${lg}  

    python $main_dir/preprocess.py \
    --task cross_lingual_lm \
    --source-lang $lg \
    --only-source \
    --srcdict $data_dir/vocab.bpe.32000.fairseq \
    --trainpref $mono_dir/$lg/all.tok.escape.32000.bpe \
    --validpref $data_dir/newstest2013.tok.bpe.32000 \
    --destdir $processed \
    --workers 20

    for stage in train valid
    do 
        mv $processed/$stage.$lg-None.$lg.bin $processed/$stage.$lg.bin
        mv $processed/$stage.$lg-None.$lg.idx $processed/$stage.$lg.idx
    done
done
wait

# generate bilingual data
#python $main_dir/preprocess.py \
#  --task xmasked_seq2seq \
#  --source-lang en \
#  --target-lang de \
#  --trainpref ${data_dir}/train.tok.clean.bpe.32000 \
#  --validpref ${data_dir}/newstest2013.tok.bpe.32000 \
#  --destdir $processed \
#  --srcdict ${data_dir}/vocab.bpe.32000.fairseq \
#  --tgtdict ${data_dir}/vocab.bpe.32000.fairseq 
