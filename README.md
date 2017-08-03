# A TensorFlow implementation of TAG Supertagging

We provide a TensorFlow implementation of the [bi-LSTM TAG Supertagging](https://jungokasai.github.io/papers/EMNLP2017.pdf).

<img src="/images/bilstm.png" width="300">

### Table of Contents  
* [Requirements](#requirements)  
* [GloVe](#glove)
* [Data Format](#data)
* [Train a Supertagger](#train)
* [Jackknife POS Tagging](#jackknife)
* [Run a pre-trained TAG Supertagger](#pretrained)
* [Structure of the Code](#structure)
* [Notes](#notes)

## Requirements

TensorFlow needs to be installed before running the training script.
TensorFlow 1.0.0 or higher is supported. 

## GloVe

Our architecture utilizes pre-trained word embedding vectors, [GloveVectors](http://nlp.stanford.edu/projects/glove/). Run the following:
```bash
wget http://nlp.stanford.edu/data/glove.6B.zip 
```
and save it to a sub-directory glovevector/. 

## <a name="data"></a>Data Format
The supertagger takes as input a file in the Conllu+Supertag (conllustag) format, in which one column for supertags is added to the original conllu format at the end. See a [sample](sample_data/conllu/sample.conllustag).

## <a name="train"></a>Train a Supertagger
All you need to do is to create a new directory for your data in the [conllustag format](#data) and a json file for the model configuration and data information. We provide a [sample json file](sample_data/config_demo.json) for the [sample](sample_data) data directory. You can train a supertagger on the sample data by the following command:
```bash
python train_bilstm_stagger.py sample_data/config_demo.json
```
After running this command, you should be getting the following files and directories in sample_data/:

| Directory/File | Description |
|------|--------|
|checkpoint.txt|Contains information about the best model.|
|sents/|Contains the words in the one-sentence-per-line format|
|gold_pos/|Contains the gold POS tags in the one-sentence-per-line format|
|gold_stag/|Contains the gold supertags in the one-sentence-per-line format|
|predicted_stag/|Contains the predicted supertags in the one-sentence-per-line format|
|Super_models/|Stores the best model.|
|conllu/sample.conllustag_stag|Contains the predicted supertags in the conllustag format|

## <a name="jackknife"></a>Jackknife POS tagging
To Be Added.


## <a name="pretrained"></a>Run a pre-trained TAG Supertagger

To Be Added.

## <a name="structure"></a>Structure of the Code
| File | Description |
|------|--------|
|``utils/preprocessing.py``|Contains tools for preprocessing. Mainly for tokenizing and indexing words/tags. Gets imported to ``utils/data_process_secsplit.py``|
|``utils/data_process_secsplit.py``|Reads training and test data and tokenize/index words, POS tags and stags. Extracts suffixes and number/capitalization features.|
|``utils/stagging_model.py``|Contains the ``Stagging_Model`` class that constructs our LSTM computation graph. The class has the necessary methods for training and testing. Gets imported to ``bilstm_stagger_model.py``|
|``utils/lstm.py``|Contains tensorflow LSTM equations. Gets imported to ``utils/stagging_model.py``.|
|``bilstm_stagger_model.py``|Contains functions that instantiate the ``Stagging_Model`` class and train/test a model. Gets imported to ``bilstm_stagger_main.py``|
|``bilstm_stagger_main.py``|Main file to run experiments. Reads model and data options.|
|``train_bilstm_stagger.py``|Runs ``bilstm_stagger_main.py`` in bash according to the json file that gets passed.|

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
