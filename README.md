# WeTS: A Benchmark for Translation Suggestion

`Translation Suggestion (TS)`, which provides alternatives for specific words or phrases given the entire documents translated by machine translation (MT) has been proven to play a significant role in post editing (PE). `WeTS` is a benchmark data set for TS, which is annotated by expert translators. WeTS contains corpus(train/dev/test) for four different translation directions, i.e., `English2German`, `German2English`, `Chinese2English` and `English2Chinese`.

***
## Contents
* [Data](#data)
* [Models](#models)
* [Get Started](#started)
* [Citation](#citation)
* [Licence](#licence)

### Data
----------

WeTS is a benchmark dataset for TS, where all the examples are annotated by expert translators. As far as we know, this is the first golden corpus for TS. The statistics about WeTS are listed in the following table:

|**Translation Direction**|**Train**| **Valid**| **Test**|
|---------------------|------|------|-----|
|English2German       |14,957|1000  |1000 |
|German2English       |11,777|1000  |1000 |
|English2Chinese      |15,769|1000  |1000 |
|Chinese2English      |21,213|1000  |1000 | 

For corpus in each direction, the data is organized as:  
*direction*.*split*.src: the source-side sentences  
*direction*.*split*.mask: the masked translation sentences, the placeholder is "\<MASK\>"  
*direction*.*split*.tgt: the predicted suggestions, the test set for English2Chinese has three references for each example 

*direction*: En2De, De2En, Zh2En, En2Zh  
*split*: train, dev, test  

### Models
---------
We release the pre-trained NMT models which are used to generate the MT sentences. Additionally, the released NMT models can be used to generate synthetic corpus for TS, which can improve the final performance dramatically.Detailed description about the way of generating synthetic corpus can be found in our paper.  

### Get Started
#### data preprocessing
```Bash
sh process.sh 
```

#### pre-training
Codes for the first-phase pre-training are not included in this repo, as we directly utilized the codes of XLM (https://github.com/facebookresearch/XLM) with little modiafication. And we did not achieve much gains with the first-phase pretraining.

The second-phase pre-training:  
```Bash
sh preptraining.sh
```

#### fine-tuning
```Bash
sh finetuning.sh
```

Codes in this repo is mainly forked from fairseq (https://github.com/pytorch/fairseq.git)
### Citation
Please cite the following paper if you found the resources in this repository useful.
```bibtex
@article{yang2021wets,
  title={WeTS: A Benchmark for Translation Suggestion},
  author={Yang, Zhen and Zhang, Yingxue and Li, Ernan and Meng, Fandong and Zhou, Jie},
  journal={arXiv preprint arXiv:2110.05151},
  year={2021}
}
```

### Licence
See LICENCE
