#!/usr/bin/bash
WORKSPACE=path_to_WeTS
TOOLS_DIR=path_to_tools
model_path=path_to_released_model

CODES_DIR=$WORKSPACE/codes_src
SCRIPTS=$TOOLS_DIR/mosesdecoder-master/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
DETOKENIZER=$SCRIPTS/tokenizer/detokenizer.perl
NORM_PUNC=$SCRIPTS/tokenizer/normalize-punctuation.perl
REM_NON_PRINT_CHAR=$SCRIPTS/tokenizer/remove-non-printing-char.perl
BPEROOT=$TOOLS_DIR/subword-nmt


source_lang='en'
target_lang='de'

bpe_codes=$WORKSPACE/nmt_dicts_en_de/en_de_fr.codes

en_src_file=$1

cat $en_src_file |  \
   perl $NORM_PUNC 'en' | \
   perl $REM_NON_PRINT_CHAR | \
   perl $TOKENIZER -threads 8 -a -l 'en' >${en_src_file}.tok

python $BPEROOT/apply_bpe.py -c $bpe_codes <${en_src_file}.tok \
    >${en_src_file}.tok.bpe

export CUDA_VISIBLE_DEVICES='0'
cat ${en_src_file}.tok.bpe | \
   python ${CODES_DIR}/interactive.py $WORKSPACE/nmt_dicts_en_de \
       --task multilingual_translation \
       --source-lang en \
       --target-lang de \
       --lang-pairs 'en-de,de-en' \
       --encoder-langtok 'src' \
       --decoder-langtok \
       --path $model_path \
       --beam 4 \
       --buffer-size 256 \
       --batch-size 128 \
       > ${en_src_file}.tok.bpe.out

grep ^H ${en_src_file}.tok.bpe.out | cut -f3 > ${en_src_file}.bpe.out.grep

sed -r 's/(@@ )|(@@ ?$)//g' ${en_src_file}.bpe.out.grep > ${en_src_file}.bpe.out.grep.noBPE

perl $DETOKENIZER -threads 8 -a -l 'de' <${en_src_file}.bpe.out.grep.noBPE \
        >${en_src_file}.bpe.out.grep.noBPE.detok
