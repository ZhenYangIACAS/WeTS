# WeTS: A Benchmark for Translation Suggestion

`Translation Suggestion (TS)`, which provides alternatives for specific words or phrases given the entire documents translated by machine translation (MT) has been proven to play a significant role in post editing (PE). `WeTS` is a benchmark data set for TS, which is annotated by expert translators. WeTS contains corpus(train/dev/test) for four different translation directions, i.e., `English2German`, `German2English`, `Chinese2English` and `English2Chinese`.

***
## Contents
* [Data](#data)
* [Models](#models)
* [Get Started](#started)
* [Citation](#citation)
* [Licences](#licence)

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
*direction*.*split*.mask: the masked translation sentences, the placeholder is "<MASK>"
*direction*.*split*.tgt: the predicted suggestions, the test set for English2Chinese has three references for each example

*direction*: En2De, De2En, Zh2En, En2Zh
*split*: train, dev, test
### Models
---------
We release the pre-trained NMT models which are used to generate the MT sentences.
