# -*- coding: utf-8 -*-

import json


class AspectExtractor():

    def __init__(self, preprocessor, filename, tokenize, only_nouns):
        self.__tokenize = tokenize
        self.__only_nouns = only_nouns
        self.__preprocessor = preprocessor
        self.__filename = filename

    def __extract_aspects_from_json(self):
        labledData = {}
        tokens = []
        categories = []
        labels = []
        category_words = [[], [], [], [], []]

        # Datei laden
        with open(self.__filename) as f:
            labledData = json.load(f)

        # JSON auslesen
        for review_id in labledData:
            review = labledData[review_id]
            token = review["token"]
            category = review["category"]
            label = review["label"]

            tokens += token
            categories += category
            labels += label

        # Aspects nach Topics ordnen
        for i in range(0, len(tokens)):
            tokens[i] = tokens[i].lower()
            temp = ""
            if categories[i] != "":
                if categories[i] == "a" and labels[i] == "BA":
                    j = i
                    while labels[j] == "BA" or labels[j] == "IA":
                        temp += tokens[j].lower() + " "
                        j += 1
                    temp = temp[:-1]
                    category_words[0].append(temp)
                    temp = ""
                if categories[i] == "s" and labels[i] == "BA":
                    j = i
                    while labels[j] == "BA" or labels[j] == "IA":
                        temp += tokens[j].lower() + " "
                        j += 1
                    temp = temp[:-1]
                    category_words[1].append(temp)
                    temp = ""
                if categories[i] == "f" and labels[i] == "BA":
                    j = i
                    while labels[j] == "BA" or labels[j] == "IA":
                        temp += tokens[j].lower() + " "
                        j += 1
                    temp = temp[:-1]
                    category_words[2].append(temp)
                    temp = ""
                if categories[i] == "p" and labels[i] == "BA":
                    j = i
                    while labels[j] == "BA" or labels[j] == "IA":
                        temp += tokens[j].lower() + " "
                        j += 1
                    temp = temp[:-1]
                    category_words[3].append(temp)
                    temp = ""
                if categories[i] == "g" and labels[i] == "BA":
                    j = i
                    while labels[j] == "BA" or labels[j] == "IA":
                        temp += tokens[j].lower() + " "
                        j += 1
                    temp = temp[:-1]
                    category_words[4].append(temp)
                    temp = ""
        return category_words

    def __preprocess_aspects(self ,category_words):
        preprocessed_aspects = self.__preprocessor.preprocess(category_words, tokenize=self.__tokenize, only_nouns=self.__only_nouns)
        return preprocessed_aspects

    def __write_aspects_json(self, preprocessed_aspects):
        data = {}
        data['ambiance'] = preprocessed_aspects[0]
        data['service'] = preprocessed_aspects[1]
        data['food'] = preprocessed_aspects[2]
        data['price'] = preprocessed_aspects[3]
        data['general'] = preprocessed_aspects[4]
        with open('aspects.json', 'w') as outfile:
            json.dump(data, outfile)

    def extract(self):
        category_words = self.__extract_aspects_from_json()
        preprocessed_aspects = self.__preprocess_aspects(category_words)
        aspects_vocab = [list(dict.fromkeys(aspects)) for aspects in preprocessed_aspects]
        self.__write_aspects_json(aspects_vocab)
