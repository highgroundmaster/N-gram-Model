import nltk

nltk.download("wordnet")
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download("words")

from nltk.metrics.distance import jaccard_distance
from nltk.util import ngrams
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.corpus import words


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


def tokenize(sentence):
    # Remove Stop Words
    stop_words = set(stopwords.words('english'))
    stop = " ".join([token for token in sentence.split() if token not in stop_words])

    # tokenize the sentence and find the POS tag for each token
    pos_tagged = nltk.pos_tag(nltk.word_tokenize(stop))
    # we use our own pos_tagger function to make things simpler to understand.
    pos_tagged = list(map(lambda x: (x[0], pos_tagger(x[1])), pos_tagged))

    # Initialize Lemmatizer
    lemmatizer = WordNetLemmatizer()
    lemmatized_sentence = []

    for word, tag in pos_tagged:
        if tag is None:
            # if there is no available tag, append the token as is
            lemmatized_sentence.append(word)
        else:
            # else use the tag to lemmatize the token
            lemmatized_sentence.append(lemmatizer.lemmatize(word, tag))
    return " ".join(lemmatized_sentence)


def vocab_builder():
    pass

