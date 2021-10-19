workspace=path_to_WeTS
tools_dir=path_to_tools
model_path=path_to_the_released_model

tokenizer=$tools_dir/mosesdecoder-master/scripts/tokenizer/tokenizer.perl
norm_punc=$tools_dir/mosesdecoder-master/scripts/tokenizer/normalize-punctuation.perl
rem_non_print=$tools_dir/mosesdecoder-master/scripts/tokenizer/remove-non-printing-char.perl

CODES=$workspace/codes_src

bpe_codes=$workspace/nmt_dicts_en_zh/zh-en.codes
en_src_file=$1

cat $en_src_file | perl $norm_punc 'en' | perl $rem_non_print | perl $tokenizer -threads 20 -a -l 'en' >>${en_src_file}.tok

python $tools_dir/subword-nmt/apply_bpe.py -c $bpe_codes \
    -i ${en_src_file}.tok \
    -o ${en_src_file}.tok.bpe \
    --num-workers 40 

cat ${en_src_file}.tok.bpe | \
  python ${CODES}/interactive.py $curr_dir/nmt_dicts_en_zh \
  --source-lang en \
  --target-lang cn \
  --path $model_path \
  --beam 4 \
  --batch-size 400 \
  --buffer-size 800 \
  --left-pad-source True \
  --left-pad-target False \
  > ${en_src_file}.out.tmp

 grep ^H ${en_src_file}.out.tmp | cut -f3 > ${en_src_file}.out

 sed -r 's/(@@ )|(@@ ?$)//g' ${en_src_file}.out > ${en_src_file}.out.noBPE
