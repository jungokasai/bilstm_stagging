# A TensorFlow implementation of TAG Supertagging

##### Table of Contents  
[Requirements](#requirements)  
We provide a TensorFlow implementation of [bi-LSTM TAG Supertagging](https://jungokasai.github.io/papers/EMNLP2017.pdf).
The supertagger takes as input a file with the Conllu+Supertag format.

## Requirements

TensorFlow needs to be installed before running the training script.
TensorFlow 1.0.0 or higher is supported. 
## Downloading Embedding Vectors

Our architecture utilizes pre-trained word embedding vectors, [GloveVectors](http://nlp.stanford.edu/projects/glove/)., run
```bash
wget http://nlp.stanford.edu/data/glove.6B.zip 
```
and save it to a sub-directory glovevector/. 


## Train a Supertagger
All you need to do is to create a directory. 
```bash
python train_bilstm_stagger.py config.json
```

## Jackknife POS tagging

To Be Added.

## Pre-trained TAG Supertagger

To Be Added.

## Notes

If you use this tool for your research, please consider citing:
```
@InProceedings{Kasai&al.17,
  author =  {Jungo Kasai and Robert Frank and R. Thomas McCoy and Owen Rambow and Alexis Nasr},
  title =   {TAG Parsing with Neural Networks and Vector Representations of Supertags},
  year =    {2017},  
  booktitle =   {Proceedings of EMNLP},  
  publisher =   {Association for Computational Linguistics},
}
```
