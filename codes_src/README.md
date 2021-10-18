# Introduction
 This page gives an introduction about the project "Deep NMT with pre-training".
 Main structure is forked from https://github.com/pytorch/fairseq, and we also incorporate some advantages of MASS(https://github.com/microsoft/MASS), XLM(https://github.com/facebookresearch/XLM).

# Features
 - We support Bert-like pre-training and finetuning in Deep NMT. To stabilize the training of Deep NMT, we utilize the "Depth-init" initialization method proposed by "https://arxiv.org/abs/1908.11365"

 - We support multi-lingual NMT system with weight-sharing.

 - We support Mass pre-training for NMT.

 - Other features provided by Fairseq.

# examples
  We provide examples for pre-train 12-layer encoders for English, and then fineting the pre-trained model on English-German translation tasks with 12-layer encoders and 12-layer decoders.

# Requirements
  * Pytorch version > 1.0.0
  * Python version >=3.6
  * For training new models, you'll also need an NVIDIA GPU and NCCL
  * Other requirements by Fairseq. 
