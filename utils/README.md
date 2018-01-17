# Stagging Model

## Stagger
The stagging model is based on the [Kasai et al. EMNLP 2017](http://www.aclweb.org/anthology/D17-1180).
The inputs to the supertagger (stagger) are:

* Word Identities
* 2 character suffixes
* Number Indicators
* Capitalization Indicators
* Predicted POS tags from the jackknife training
* Char-level CNNs (not in the paper, added later)

All the inputs listed above are sequences of length equal to the sentence length; our placeholders in the TensorFlow implementation are just those lists! Simple.

## Notes on the Implementation

### Data Loader
``data_process_secsplit.py`` serves as the universal data loader.
test_opts is ``None`` when you train the model. You pass a ``test_opts`` when you use the test option in ``bilstm_stagger_main.py``.

Unlike word identities, we do not reserve the zero index for the supertags; zero is used both for zero-padding and the most frequent supertag. This is because we can extract where those paddings are by looking at zero's in word sequences, and we do not need to include a zero pad in the probability distribution outputted through the final softmax layer.
