## Tools

### Data Format Converters
- micastags2sents.py: convert n-best sequences with scores
- goldconllu2predictedconllu.py: convert conllu with gold POS tags to conllu with predicted POS tags
- convert_parentheses.py: convert parentheses to -RRB- and -LRB- following John Chen's corpus

### Stagging Analysis
- feature_analyzer.py: create confusion matrices
- compute_prec_recall.py: compute macro and micro precision and recall based on the confusion matrices
- modif_analyzer.py: collapse the modif confusion matrix for modifier analysis
