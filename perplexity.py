from language_model import NGramModel
import nltk
from nltk.corpus import webtext
import math

nltk.download("webtext")


class Perplexity(NGramModel):
    def perplexity(self):
        test_n_grams = []
        for sentence in self.test_tokens:
            test_n_grams += [n_gram for n_gram in nltk.ngrams(sentence, self.n) if n_gram in self.vocab]

        probabilities = [self.model.get(n_gram) for n_gram in test_n_grams]

        return math.exp((-1 / len(self.test_tokens)) * sum(map(math.log, probabilities)))


if __name__ == '__main__':
    corpus = []
    for fileid in webtext.fileids():
        corpus += [list(i) for i in webtext.sents(fileid)]

    bi_gram = Perplexity(2, corpus, train_ratio=0.7)
    tri_gram = Perplexity(3, corpus, train_ratio=0.7)
    tri_gram.train_tokens, tri_gram.test_tokens = bi_gram.train_tokens, bi_gram.test_tokens
    quad_gram = Perplexity(4, corpus, train_ratio=0.7)
    quad_gram.train_tokens, quad_gram.test_tokens = bi_gram.train_tokens, bi_gram.test_tokens

    bi_gram.fit()
    tri_gram.fit()
    quad_gram.fit()
    print(f"Bi-Gram Perplexity - {bi_gram.perplexity()}")
    print(f"Tri-Gram Perplexity - {tri_gram.perplexity()}")
    print(f"Quad-Gram Perplexity - {quad_gram.perplexity()}")
