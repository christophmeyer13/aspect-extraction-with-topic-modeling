# -*- coding: utf-8 -*-

import json
import numpy as np
from gensim import matutils


class Utils():

    @staticmethod
    def get_tokens_from_json(filename):
        sentence_list = []
        labledData = {}
        with open(filename) as f:
            labledData = json.load(f)
        for review_id in labledData:
            review = labledData[review_id]
            sentence_list.append(review["token"])
        return sentence_list

    @staticmethod
    def get_seeds_from_json(filename, vocab):
        seed_topic_list = [[],  # ambience (a)
                   [],  # service (s)
                   [],  # food (f)
                   [],  # price (p)
                   []   # general (g)
                  ]

        seeds = {}

        # Datei laden
        with open(filename) as f:
            seeds = json.load(f)

        seed_topic_list[0] += [i for i in seeds["ambiance"] if i in vocab]
        seed_topic_list[1] += [i for i in seeds["service"] if i in vocab]
        seed_topic_list[2] += [i for i in seeds["food"] if i in vocab]
        seed_topic_list[3] += [i for i in seeds["price"] if i in vocab]
        seed_topic_list[4] += [i for i in seeds["general"] if i in vocab]

        return seed_topic_list

    @staticmethod
    def calculate_recall(topic_word, topic_number, vocab, aspects_of_topic, n_top_words=50):
        topic_words = np.array(vocab)[np.argsort(topic_word[topic_number])][:-(n_top_words+1):-1]

        topic_relevant_count = len(aspects_of_topic)
        topic_false_positives = 0
        for word in topic_words:
            if word not in aspects_of_topic:
                topic_false_positives += 1
        topic_true_positives = n_top_words - topic_false_positives
        return topic_true_positives/topic_relevant_count

    @staticmethod
    def calculate_precision(topic_word, topic_number, vocab, aspects_of_topic, n_top_words=50):
        topic_words = np.array(vocab)[np.argsort(topic_word[topic_number])][:-(n_top_words+1):-1]

        topic_false_positives = 0
        for word in topic_words:
            if word not in aspects_of_topic:
                topic_false_positives += 1
        topic_true_positives = n_top_words - topic_false_positives
        return topic_true_positives/n_top_words
    
    @staticmethod
    def calculate_new_true_positives(topic_word, topic_number, vocab, aspects_of_topic, seeds_of_topic, n_top_words=50):
        topic_words = np.array(vocab)[np.argsort(topic_word[topic_number])][:-(n_top_words+1):-1]
        count = 0
        
        for word in topic_words:
            if word in aspects_of_topic:
                if word not in seeds_of_topic:
                    count += 1
        return count

    @staticmethod
    def calculate_average(list_of_nums):
        return sum(list_of_nums)/len(list_of_nums)

    def __bow_iterator(self, docs, dictionary):
        for doc in docs:
            yield dictionary.doc2bow(doc)

    def get_term_matrix(self, msgs, dictionary):
          bow = self.__bow_iterator(msgs, dictionary)
          X = np.transpose(matutils.corpus2csc(bow).astype(np.int64))
          return X
