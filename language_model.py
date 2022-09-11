# TODO - Progress Bar
from tqdm import tqdm
import random
from tokenization import split_sentences
import nltk
from itertools import chain
from nltk.tokenize import RegexpTokenizer


class NGramModel():
    """
    N-Gram Model with Laplace Smoothing
    Args:
        corpus (list of str): list of sentences comprising the training corpus.
        n (int): the order of language model to build (i.e. 1 for unigram, 2 for bigram, etc.).
        laplace (int): lambda multiplier to use for laplace smoothing (default 1 for add-1 smoothing).
    """

    def __init__(self, n, corpus, laplace_factor=1, train_ratio=0.7):
        self.n = n
        self.tokens = split_sentences(corpus)
        self.train_tokens, self.test_tokens = self.train_test_split(train_ratio)
        self.laplace = laplace_factor
        self.n_grams = []
        self.vocab = None
        self.unique_words = set(chain.from_iterable(self.tokens))
        self.model = None

    def train_test_split(self, train_ratio):
        random.shuffle(self.tokens)
        index = int(len(self.tokens) * train_ratio)
        return self.tokens[:index], self.tokens[index:]

    def _smooth(self):
        """
        Laplace smoothing
        Returns:
            dict: Mapping of each n-gram (tuple of str) to its Laplace-smoothed
            probability (float).
        """
        vocab_size = len(self.vocab)
        m_grams = []

        for sentence in self.train_tokens:
            m_grams += list(nltk.ngrams(sentence, self.n - 1))
        m_vocab = nltk.FreqDist(m_grams)

        def smoothed_count(n_gram, n_count):
            m_gram = n_gram[:-1]
            m_count = m_vocab[m_gram]
            return (n_count + self.laplace) / (m_count + self.laplace * vocab_size)

        return {n_gram: smoothed_count(n_gram, count) for n_gram, count in self.vocab.items()}

    def fit(self):
        """
        Create a probability distribution for the vocabulary of the training corpus with smoothing.

        Unigram model no laplace smoothing
        Returns:
            A dict mapping each n-gram (tuple of str) to its probability (float).
        """
        for sentence in tqdm(self.train_tokens, desc="Fitting the Model"):
            self.n_grams += list(nltk.ngrams(sentence, self.n))
        self.vocab = nltk.FreqDist(self.n_grams)

        if self.n == 1:
            num_tokens = len(self.vocab)
            self.model = {unigram: count / num_tokens for unigram, count in self.vocab.items()}
        else:
            self.model = self._smooth()

    def probability(self, sentence):
        tokenizer = RegexpTokenizer(r'\w+')
        tokens = tokenizer.tokenize(sentence)
        n_grams = list(nltk.ngrams(tokens, self.n))
        return sum([self.model.get(n_gram, 0) for n_gram in n_grams])
