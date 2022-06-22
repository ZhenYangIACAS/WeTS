# pre-process for en2cn

workspace=path_to_WeTS

codes_dir=$workspace/codes_src
data_dir=$workspace/corpus
output_dir=$workspac/data-bin

src_vocab=$workspace/src.vocab
tgt_vocab=$workspace/tgt.vocab

# bpe
python apply_bpe.py -c $bpe_codes <$data_dir/en2cn/en2cn.train.src > $data_dir/en2cn/en2cn.train.src.bpe.en 
python apply_bpe.py -c $bpe_codes <$data_dir/en2cn/en2cn.train.mask > $data_dir/en2cn/en2cn.train.src.bpe.cn
python apply_bpe.py -c $bpe_codes <$data_dir/en2cn/en2cn.train.tgt > $data_dir/en2cn/en2cn.train.tgt.cn

python apply_bpe.py -c $bpe_codes <$data_dir/en2cn/en2cn.valid.src > $data_dir/en2cn/en2cn.valid.src.bpe.en 
python apply_bpe.py -c $bpe_codes <$data_dir/en2cn/en2cn.valid.mask > $data_dir/en2cn/en2cn.valid.src.bpe.cn
python apply_bpe.py -c $bpe_codes <$data_dir/en2cn/en2cn.valid.tgt > $data_dir/en2cn/en2cn.valid.tgt.cn


# build vocab
touch $src_vocab
python $codes_dir/build_vocab.py $data_dir/en2cn/en2cn.train.src.bpe.en $data_dir/en2cn.train.src.bpe.en $src_vocab 5 

touch $tgt_vocab
python $codes_dir/build_vocab.py $data_dir/en2cn/en2cn.train.tgt.bpe $tgt_vocab 5 

# binarize
python $codes_dir/preprocess.py --source-lang en --target-lang cn \
  --task input_suggestion \
  --trainpref $data_dir/en2cn/en2cn.train.src.bpe \
  --validpref $data_dir/en2cn/en2cn.valid.src.bpe \
  --destdir $output_dir \
  --joined-dictionary \
  --srcdict ${src_vocab} \
  --workers 10

rm -r $output_dir/dict.cn.txt
mv $output_dir/train.en-cn.cn.bin $output_dir/train.en-cn.src.cn.bin
mv $output_dir/train.en-cn.cn.idx $output_dir/train.en-cn.src.cn.idx
mv $output_dir/train.en-cn.en.bin $output_dir/train.en-cn.src.en.bin
mv $output_dir/train.en-cn.en.idx $output_dir/train.en-cn.src.en.idx

mv $output_dir/valid.en-cn.cn.bin $output_dir/valid.en-cn.src.cn.bin
mv $output_dir/valid.en-cn.cn.idx $output_dir/valid.en-cn.src.cn.idx
mv $output_dir/valid.en-cn.en.bin $output_dir/valid.en-cn.src.en.bin
mv $output_dir/valid.en-cn.en.idx $output_dir/valid.en-cn.src.en.idx

python $codes_dir/preprocess.py --source-lang cn \
  --task input_suggestion \
  --trainpref $data_dir/en2cn/en2cn.train.tgt.bpe \
  --validpref $data_dir/en2cn/en2cn.valid.tgt.bpe \
  --destdir $output_dir \
  --srcdict ${tgt_vocab} \
  --only-source \
  --workers 10

mv $output_dir/train.cn-None.cn.bin $output_dir/train.en-cn.tgt.cn.bin
mv $output_dir/train.cn-None.cn.idx $output_dir/train.en-cn.tgt.cn.idx

mv $output_dir/valid.cn-None.cn.bin $output_dir/valid.en-cn.tgt.cn.bin
mv $output_dir/valid.cn-None.cn.idx $output_dir/valid.en-cn.tgt.cn.idx

