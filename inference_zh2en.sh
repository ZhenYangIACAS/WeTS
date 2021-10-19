workspace=path_to_WeTS
tools_dir=path_to_tool
model_path=path_to_released_model

tokenizer=$tools_dir/mosesdecoder-master/scripts/tokenizer/tokenizer.perl
detokenizer=$tools_dir/mosesdecoder-master/scripts/tokenizer/detokenizer.perl
norm_punc=$tools_dir/mosesdecoder-master/scripts/tokenizer/normalize-punctuation.perl
rem_non_print=$tools_dir/mosesdecoder-master/scripts/tokenizer/remove-non-printing-char.perl

CODES=$workspace/codes_src
bpe_codes=$workspace/nmt_dicts_en_zh/zh-en.codes_src

cn_src_file=$1

# We use the open-source tool textsmart for Chinese segment. More details can be found at: https://ai.tencent.com/ailab/nlp/texsmart/zh/index.html
python $tools_dir/segment_qqseg.py $cn_src_file ${cn_src_file}.seg

python $tools_dir/subword-nmt/apply_bpe.py -c $bpe_codes \
    -i ${cn_src_file}.seg \
    -o ${cn_src_file}.seg.bpe \
    --num-workers 40 

cat ${cn_src_file}.seg.bpe | \
  python ${codes_src}/interactive.py $workspace/nmt_dicts_en_zh \
  --source-lang cn \
  --target-lang en \
  --path $model_path \
  --beam 4 \
  --batch-size 200 \
  --buffer-size 400 \
  --left-pad-source True \
  --left-pad-target False \
  > ${cn_src_file}.out.tmp

 grep ^H ${cn_src_file}.out.tmp | cut -f3 > ${cn_src_file}.out

 sed -r 's/(@@ )|(@@ ?$)//g' ${cn_src_file}.out > ${cn_src_file}.out.noBPE

perl detokenizer \
  < ${cn_src_file}.out.noBPE \
  > ${cn_src_file}.out.noBPE.detok
