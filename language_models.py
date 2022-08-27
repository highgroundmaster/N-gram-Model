from tokenization import vocab_builder, tokenize
import nltk


def get_ngrams(n: int, tokens: list) -> list:
    """
    :param n: n-gram size
    :param tokens: tokenized sentence
    :return: list of ngrams
    ngrams of tuple form: ((context), target word)
    """
    tokens = (n - 1) * ['<s>'] + tokens
    n_gram = [(tuple([tokens[i - p - 1] for p in reversed(range(n - 1))]), tokens[i]) for i in
              range(n - 1, len(tokens))]
    return n_gram


class LanguageModel(object):
    """
    An n-gram language model trained on a given corpus.

    For a given n and given training corpus, constructs an n-gram language
    model for the corpus by:
    1. preprocessing the corpus (adding SOS/EOS/UNK tokens)
    2. calculating (smoothed) probabilities for each n-gram
    Also contains methods for calculating the perplexity of the model
    against another corpus, and for generating sentences.
    Args:
        corpus (list of str): list of sentences comprising the training corpus.
        n (int): the order of language model to build (i.e. 1 for unigram, 2 for bigram, etc.).
        laplace (int): lambda multiplier to use for laplace smoothing (default 1 for add-1 smoothing).
    """

    def __init__(self, corpus, n, laplace=1):
        self.n = n
        self.laplace = laplace
        self.tokens = tokenize(corpus, n)
        self.vocab = nltk.FreqDist(self.tokens)
        self.model = self._create_model()
