# A TensorFlow implementation of TAG Supertagging

We provide a TensorFlow implementation of [bi-LSTM TAG Supertagging](https://jungokasai.github.io/papers/EMNLP2017.pdf).
The supertagger takes as input a file with the Conllu+Supertag format.

## Requirements

TensorFlow needs to be installed before running the training script.
TensorFlow 1.0.0 or higher is supported. 
## Downloading Embedding Vectors

Our architecture utilizes pre-trained word embedding vectors, [GloveVectors](http://nlp.stanford.edu/projects/glove/) and [Word2Vec](https://code.google.com/archive/p/word2vec/). For [GloveVectors](http://nlp.stanford.edu/projects/glove/), run
```bash
wget http://nlp.stanford.edu/data/glove.6B.zip 
```
and save it to a sub-directory glovevector/. 


## Train a Supertagger


## Pre-trained TAG Supertagger

To Be Added.

## Notes

If you you this tool for your research, please consider citing:
```
@InProceedings{Kasai&al.17,
  author =  {Jungo Kasai and Robert Frank and R. Thomas McCoy and Owen Rambow and Alexis Nasr},
  title =   {TAG Parsing with Neural Networks and Vector Representations of Supertags},
  year =    {2017},  
  booktitle =   {Proceedings of EMNLP},  
  publisher =   {Association for Computational Linguistics},
}
```

 

<!--  We provide a pre-trained supertagger. First, download the [pre-trained supertagger](https://drive.google.com/drive/folders/0B2DPH65KzVP7aC1FRWNjRXFLT28?usp=sharing). Place all of the files into, for example, `pretrained/ptb_wsj/`. If you want to run the supertagger on your own corpus, place a tokenized file with one sentence per line, in say, `data/sents/your_corpus.txt`. Then run the following command.
You first need to perform POS tagging on the corpus, using for example the Stanford Core NLP. Place your one sentence per line POS tag data in `data/pos_data/your_pos.txt`. We also provide pre-traned POS tagger. Finally run the following command:

```bash
python tf_lstm_main.py test --model pretrained/ptb_wsj/epoch18_accuracy0.89657 --pos data/pos_data/your_pos.txt --save_tags
```
where the last option saves 1-best stags in a 1best_stags.txt.
If you have gold stag data, you can compute accuracy by adding the option of `--get_accuracy` and `--tag PATH_TO_GOLD_STAGS`.

## Training the Vanilla Network

In order to train the network, execute
```bash
python tf_pos_lstm_main.py train 
```

You can see documentation on each of the training configurations by running
```bash
python tf_pos_lstm_main.py --help
```

## TO-DO's

10 best/beta pruning outputs will be supported soon.
-->
