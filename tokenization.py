import re

import nltk
import string
import json
nltk.download("wordnet")
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download("words")
nltk.download('omw-1.4')

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.corpus import wordnet, brown


def pos_tagger(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None


def preprocess(tokens):
    # Remove Stop Words
    stop_words = set(stopwords.words('english'))
    stop = [token for token in tokens if token not in stop_words]
    # tokenize the sentence and find the POS tag for each token
    pos_tagged = nltk.pos_tag(stop)
    # we use our own pos_tagger function to make things simpler to understand.
    pos_tagged = list(map(lambda x: (x[0].lower(), pos_tagger(x[1])), pos_tagged))

    # Initialize Lemmatizer
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = []

    for word, tag in pos_tagged:
        if tag is None:
            # if there is no available tag, append the token as is
            lemmatized_tokens.append(word)
        else:
            # else use the tag to lemmatize the token
            lemmatized_tokens.append(lemmatizer.lemmatize(word, tag))
    return lemmatized_tokens


def tokenize(data: string) -> list:
    # Remove Punctuation
    string.punctuation = string.punctuation + '“' + '”' + '-' + '’' + '‘' + '—'
    string.punctuation = string.punctuation.replace('.', '')
    # pre_tokens =
    # if shape == 1:
    return [word_preprocess(token) for token in nltk.word_tokenize(data)
            if all(punct != token for punct in string.punctuation)]
    # tokens = []
    # start, end = 0, 0
    # for index, token in enumerate(pre_tokens):
    #
    #     if token == "." or token.endswith("."):
    #         end = index
    #         tokens.append(pre_tokens[start:end])
    #         start = index + 1
    # return tokens


def word_preprocess(word: string) -> string:
    return re.sub(r'[^\w\s]', "", word)


def split_sentences(data: list) -> list:
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = []
    sent = []
    for sentence in data:
        tags = nltk.pos_tag(sentence)
        for index, token in enumerate(sentence):
            new_tokens = tokenizer.tokenize(token)
            for new_token in new_tokens:
                if len(new_token) == 1:
                    sent.append(new_token)
            if token == ".":
                tokens.append(sent)
                sent = []
    return tokens

    # for index, token in enumerate(data):
    #     tag = nltk.pos_tag([token])


if __name__ == '__main__':
    # tokenizer = RegexpTokenizer(r'\w+')
    # print(tokenizer.tokenize("September-October"))
    print(split_sentences(brown.sents(categories='news')[:5]))
    # print(nltk.pos_tag(brown.words(categories='news')[:15]))
#     print(brown.words(categories='news')[:30])
#     with open("split.json", "w") as f:
#         json.dump(split_sentences(brown.words(categories='news')), f, indent= 4)
#     with open("brown.json", "w") as f:
#         json.dump(list(brown.words(categories='news')), f)