# TODO - Progress Bar
# TODO - Unigram Dont Remove Stop Words


import random
import pandas as pd
import matplotlib.pyplot as plt
from tokenization import split_sentences
import seaborn as sns
import nltk
from nltk.corpus import stopwords, brown
from itertools import chain

class NGramModel(object):
    """
    An n-gram language model trained on a given corpus.

    For a given n and given corpus, constructs an n-gram language model
    Includes
    Laplace Smoothing
    Perplexity Calculation
    Args:
        corpus (list of str): list of sentences comprising the training corpus.
        n (int): the order of language model to build (i.e. 1 for unigram, 2 for bigram, etc.).
        laplace (int): lambda multiplier to use for laplace smoothing (default 1 for add-1 smoothing).
    """

    def __init__(self, n, corpus, laplace_factor=1, train_ratio=0.7):
        self.n = n
        self.tokens = split_sentences(corpus)
        self.train_tokens, self.test_tokens = self.train_test_split(train_ratio)
        self.laplace_factor = laplace_factor
        self.n_grams = []
        self.vocab = None
        self.unique_words = set(chain.from_iterable(self.tokens))

    def train_test_split(self, train_ratio):
        random.shuffle(self.tokens)
        index = int(len(self.tokens) * train_ratio)
        return self.tokens[:index], self.tokens[index:]

    def train(self):
        stop_words = set(stopwords.words('english'))
        for sentence in self.train_tokens:
            n_grams = list(nltk.ngrams(sentence, self.n))
            for n_gram in n_grams:
                # Doesn't append ngram containing only stopwords
                if any(token not in stop_words for token in n_gram):
                    self.n_grams.append(n_gram)
        self.vocab = nltk.FreqDist(self.n_grams)


    def smoothing(self):
        pass

    # def visualize_freq(self):
    #     ngram_fd = self.vocab.most_common(20)
    #     ## Sort values by highest frequency
    #     ngram_sorted = {k: v for k, v in sorted(ngram_fd.items(), key=lambda item: -item[1])}
    #
    #     ## Join bigram tokens with '_' + maintain sorting
    #     ngram_joined = {'_'.join(k): v for k, v in
    #                     sorted(ngram_fd.items(), key=lambda item: -item[1])}
    #
    #     ## Convert to Pandas series for easy plotting
    #     ngram_freqdist = pd.Series(ngram_joined)
    #
    #     ## Setting figure & ax for plots
    #     fig, ax = plt.subplots(figsize=(10, 10))
    #
    #     ## Setting plot to horizontal for easy viewing + setting title + display
    #     bar_plot = sns.barplot(x=ngram_freqdist.values, y=ngram_freqdist.index, orient='h', ax=ax)
    #     plt.title('Frequency Distribution')
    #     plt.show();

if __name__ == '__main__':

    BiGram = NGramModel(2, brown.words(categories='news'))
    BiGram.train()
    # print(BiGram.vocab)
    print(BiGram.n_grams[:30])
    BiGram.vocab.plot(cumulative=False)
