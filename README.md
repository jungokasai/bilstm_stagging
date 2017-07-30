# A TensorFlow implementation of TAG Supertagging

We provide a TensorFlow implementation of the [bi-LSTM TAG Supertagging](https://jungokasai.github.io/papers/EMNLP2017.pdf).

### Table of Contents  
* [Requirements](#requirements)  
* [GloVe](#glove)
* [Train a Supertagger](#Train a Supertagger)
* [Run a pre-trained TAG Supertagger](#Run a pre-trained TAG Supertagger)
* [Notes](#Notes)

## Requirements

TensorFlow needs to be installed before running the training script.
TensorFlow 1.0.0 or higher is supported. 

## GloVe

Our architecture utilizes pre-trained word embedding vectors, [GloveVectors](http://nlp.stanford.edu/projects/glove/)., run
```bash
wget http://nlp.stanford.edu/data/glove.6B.zip 
```
and save it to a sub-directory glovevector/. 

## Data Format
The supertagger takes as input a file in the Conllu+Supertag (conllustag) format, in which one column for supertags is added to the original conllu format at the end. See a [sample](sample_data/conllu/sample.conllustag).

## Train a Supertagger
All you need to do is to create a new directory for your data in the [conllustag format](#Data Format) and a json file for the model configuration and data information. We provide a [sample json file](config_demo.json) for the [sample](sample_data) data directory. You can train a supertagger on the sample data by the following command:
```bash
python train_bilstm_stagger.py sample_data/config_demo.json
```
After running this command, you should be getting the following files and directories in sample_data/:

| Dir/File | Description |
|------|--------|
|checkpoint.txt||
|sents|Contains the words in the one-sentence-per-line format|
|gold_pos|Contains the gold POS tags in the one-sentence-per-line format|
|gold_stag|Contains the gold supertags in the one-sentence-per-line format|
|predicted_stag|Contains the predicted supertags in the one-sentence-per-line format|
|Super_models|Stores the best model.|
|conllu/sample.conllustag_stag|Contains the predicted supertags in the conllustag format|


## Jackknife POS tagging

To Be Added.

## Run a pre-trained TAG Supertagger

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
