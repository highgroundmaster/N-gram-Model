from tqdm import tqdm

from language_model import NGramModel
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords, brown, gutenberg
import random
import time
import json

nltk.download('gutenberg')


class SpellCheck(NGramModel):
    def get_candidates(self, word):
        # Returns the words that are edit distance 0,1 or 2 away from the given word. Inspired from Peter Norvig
        candidates = [set(), set()]
        for candidate in self.unique_words:
            ed = nltk.edit_distance(word, candidate)
            # 80 percent errors in edit distance one, so prioritize this
            if ed == 1:
                candidates[0].update([candidate])
            elif ed == 2:
                candidates[1].update([candidate])
        return candidates

    def best_candidate(self, word, n_grams, is_real=False):
        """
        Replace a non-word with the closest word that has a Probability
        :param n_grams: n_grams the word is involved in
        :return: best candidate (str)
        """
        # Non - Word Errors
        if not is_real and (word in self.unique_words or len(word) == 1):
            return word

        score = 0
        best_candidate = None
        candidates = self.get_candidates(word)
        for e1_candidate in candidates[0]:
            prob = sum([self.model.get(tuple(map(lambda x: x.replace(word, e1_candidate), n_gram)), 0)
                        for n_gram in n_grams]) * 0.8
            if score < prob:
                score = prob
                best_candidate = e1_candidate

        for e2_candidate in candidates[1]:
            prob = sum([self.model.get(tuple(map(lambda x: x.replace(word, e2_candidate), n_gram)), 0)
                        for n_gram in n_grams]) * 0.2
            if score < prob:
                score = prob
                best_candidate = e2_candidate

        if best_candidate is None:
            best_candidate = word

        return best_candidate

    def non_word(self, sentence):
        tokenizer = RegexpTokenizer(r'\w+')
        tokens = tokenizer.tokenize(sentence)
        n_grams = list(nltk.ngrams(tokens, self.n))
        for index, token in enumerate(tokens):
            tokens[index] = self.best_candidate(token, [n_gram for n_gram in n_grams if token in n_gram])
            sentence = " ".join(tokens)
        return sentence

    def real_word(self, sentence):
        tokenizer = RegexpTokenizer(r'\w+')
        tokens = tokenizer.tokenize(sentence)
        n_grams = list(nltk.ngrams(tokens, self.n))
        max_prob = 0
        best_sentence = sentence
        for index, token in enumerate(tokens):
            temp_tokens = tokens
            if len(token) == 1:
                continue
            temp_tokens[index] = self.best_candidate(token, [n_gram for n_gram in n_grams if token in n_gram], True)
            temp = " ".join(temp_tokens)
            prob = self.probability(temp)
            if max_prob < prob:
                max_prob = prob
                best_sentence = temp
        return best_sentence

    def spell_check(self, sentence):
        return self.real_word(self.non_word(sentence))

    def non_word_error(self, word):
        error = random.choices([1, 2], weights=[0.8, 0.2])[0]
        if len(word) == 2:
            kind = random.choice(["i", "s"])
        elif len(word) == 1:
            kind = "i"
        else:
            kind = random.choice(["i", "s", "d"])
        index = random.choice(range(len(word)))
        for i in range(error):
            letter = random.choice(range(26))
            if kind == "d":
                word = word[: index] + word[index + 1:]
            elif kind == "i":
                word = word[: index] + chr(ord("a") + letter) + word[index:]
            else:
                word = word[: index] + chr(ord("a") + letter) + word[index + 1:]
        return word

    def real_word_error(self, word):
        error = random.choices([0, 1], weights=[0.8, 0.2])[0]
        candidates = list(self.get_candidates(word)[error])
        return random.choice(candidates)

    def evaluation(self, is_real=False):
        eval = []
        accuracy = 0
        name = "real-word.json" if is_real else "non-word.json"
        for sent in tqdm(self.test_tokens, desc="Spell Check Evaluation"):
            result = {}
            result["Original Sentence"] = " ".join(sent)
            if sent:
                token = random.choice([tok for tok in sent if len(tok) > 2])
                if not is_real:
                    error_token = self.non_word_error(token)
                    sent[sent.index(token)] = error_token
                    result["Error Sentence"] = " ".join(sent)
                    result["Non-Word Sentence"] = self.non_word(result["Error Sentence"])
                    result["Accuracy"] = self.accuracy(
                        nltk.word_tokenize(result["Original Sentence"]),
                        nltk.word_tokenize(result["Non-Word Sentence"])
                    )
                    accuracy += result["Accuracy"]
                    eval.append(result)
                else:
                    error_token = self.real_word_error(token)
                    sent[sent.index(token)] = error_token
                    result["Error Sentence"] = " ".join(sent)
                    result["Real-Word Sentence"] = self.real_word(result["Error Sentence"])
                    result["Accuracy"] = self.accuracy(
                        nltk.word_tokenize(result["Original Sentence"]),
                        nltk.word_tokenize(result["Real-Word Sentence"])
                    )
                    accuracy += result["Accuracy"]
                    print(result)
                    eval.append(result)

        with open(name, "w") as f:
            json.dump(eval, f, indent=4)

        return accuracy / len(self.test_tokens)

    def accuracy(self, original, predicted):
        accuracy = 0
        for i in range(len(original)):
            if original[i].lower() == predicted[i].lower() :
                accuracy += 1
        return accuracy/len(original)




def sample_corpus(sample_size=0.1):
    corpus = []
    for category in nltk.corpus.brown.categories():
        para = [list(i) for i in nltk.corpus.brown.sents(categories=category)]
        corpus += random.sample(para, int(sample_size * len(para)))
    return corpus


def tester(inputs, function, expected_outputs=None):
    passed = True
    for i, ele in enumerate(inputs):
        print(f"[TESTING] {i} Input = {ele}")
        print(f"[TESTING] Running...")
        start_time = time.time_ns()
        out = function(ele)
        end_time = time.time_ns()
        print(f"[TESTING] finished in {(start_time - end_time) / 1000}")
        print(f"[TESTING] Output is {out}")
        if expected_outputs is not None:
            exp_ele = expected_outputs[i]
            passed = passed and exp_ele != out
            if exp_ele != out:
                print(f"[TESTING] #### FAILED ####")
            else:
                print(f"[TESTING] ** PASSED **")

    if expected_outputs is not None:
        print(f"[TESTING] passed {passed}")


if __name__ == '__main__':
    checker = SpellCheck(2, gutenberg.sents('austen-emma.txt'), train_ratio=0.8)
    checker.fit()
    # print(checker.spell_check("He raw his son every year."))
    print(checker.evaluation(True))