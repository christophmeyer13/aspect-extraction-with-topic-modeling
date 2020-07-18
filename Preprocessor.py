# -*- coding: utf-8 -*-

import nltk
import string
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords


class Preprocessor():

    def __init__(self):
        nltk.download()

    def __tokenize(self, sentence_list):
        tokenized_sentences = [nltk.word_tokenize(sentence, language='english') for sentence in sentence_list]
        return tokenized_sentences

    def __remove_stop_words(self, tokenized_sentences):
        stopped_tokens = []
        for tokens in tokenized_sentences:
            stopped_tokens.append([i for i in tokens if i not in stopwords.words('english')])
        return stopped_tokens

    def __part_of_speech_tagging(self, stopped_tokens):
        tagged_tokens = [nltk.pos_tag(sentence) for sentence in stopped_tokens]
        return tagged_tokens

    def __wntag(self, pttag):
        if pttag in ['JJ', 'JJR', 'JJS']:
            return wn.ADJ
        elif pttag in ['NN', 'NNS', 'NNP', 'NNPS']:
            return wn.NOUN
        elif pttag in ['RB', 'RBR', 'RBS']:
            return wn.ADV
        elif pttag in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']:
            return wn.VERB
        return None

    def __remove_non_nouns(self, tagged_tokens):
        nouns = []
        for sentence in range(len(tagged_tokens)):
            nouns.append([])
            for token in tagged_tokens[sentence]:
                if self.__wntag(token[1]) == wn.NOUN:
                    nouns[sentence].append(token)
        return nouns

    def __lemmatize(self, lemmatizer, word, pos):
        if pos is None:
            return word
        else:
            return lemmatizer.lemmatize(word, pos)

    def __lemmatize_all(self, tagged_tokens):
        lemmatizer = nltk.WordNetLemmatizer()
        lemmatized_tokens = []
        for sentence in tagged_tokens:
            lemmatized_tokens.append([self.__lemmatize(lemmatizer, word, self.__wntag(pos)) for (word, pos) in sentence])
        return lemmatized_tokens

    def __stem(self, lemmatized_tokens):
        stemmer = nltk.PorterStemmer()
        stemmed_tokens = []
        for sentence in lemmatized_tokens:
            stemmed_tokens.append([stemmer.stem(t) for t in sentence])
        return stemmed_tokens

    def __remove_invalid_tokens(self, stemmed_tokens):
        without_interpunctation = [[token for token in sentence if token not in string.punctuation] for sentence in stemmed_tokens]
        without_short_tokens = [[token for token in sentence if len(token) > 3] for sentence in without_interpunctation]
        without_spaced_tokens = [[token for token in sentence if " " not in token] for sentence in without_short_tokens]
        return without_spaced_tokens

    def preprocess(self, document_list, tokenize=True, only_nouns=True):
        tokenized_sentences = document_list

        if tokenize is True:
            tokenized_sentences = self.__tokenize(document_list)

        stopped_tokens = self.__remove_stop_words(tokenized_sentences)
        tagged_tokens = self.__part_of_speech_tagging(stopped_tokens)

        if only_nouns is True:
            tagged_tokens = self.__remove_non_nouns(tagged_tokens)

        lemmatized_tokens = self.__lemmatize_all(tagged_tokens)
        stemmed_tokens = self.__stem(lemmatized_tokens)
        result = self.__remove_invalid_tokens(stemmed_tokens)

        return result
