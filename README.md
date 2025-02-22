# MLP_Word_Segementation
This a code written for training and evaluating Sandhi Splitter or Word segmentor in Dravidian languages. The dataset provided here is for Malayalam, however any langauge dataset can be used. This system tries to classify each and every character in a word to split point and non split point based on the character contexts.
# Concept Of Sandhi
Sandhi is the process of joining two words
or characters, where morphophonemic changes
occur at the point of joining. The presence of
Sandhi is abundant in Sanskrit and all Dravidian
languages. When compared to other Dravidian
languages, the presence of Sandhi is relatively
high in Malayalam. Even a full sentence may
exist as a single string due to the process of
Sandhi. For example, avanaaraaN)
is a sentence in Malayalam which means “Who
is he ?”. It is composed of 3 independent words,
namely (avan (he)), (aar(who)) and
(aaN(is)).

# Why Sandhi SPlitter?
The identification of words becomes
complex when words are joined to form a single
string with morpho-phonemic changes at the point
of joining. More over, the sandhi can happen between
any linguistic classes like, a noun and a
verb, or a verb and a connective etc. This leads to
misidentification of classes of words by POS tagger
which eventually affects parsing. Sandhi acts
as a bottle-neck for all term distribution based approaches
for any NLP and IR task.

# Customisable Parameters

The code has been written in such a way that one can give the percentage of training data to be used as a parameter. Every time it shuffles the training data and chooses the first x%. Along with that, one can choose datasets from gold/wiki/gold+wiki(full). Options to choose the batch size and the number of epochs also have been incorporated.

 - data wiki(default), gold, full.
 - percentage 100.00(default)
 - batch_size 100(default)
 - epochs 100(default)

# Prerequisites.

* Python3 Version: 3.5.2
* keras(Tensorflow Backend) version: 2.0.7
* gensim version: 2.3.0
* sklearn version: 0.19.0
* numpy version: 1.13.1

# How to run
```
$ cd MLP_SandhiSplitter 
$ python3 mlp_sandhi_splitter.py --data wiki --percentage 0.50 --batch_size 100 --epochs 100
```
