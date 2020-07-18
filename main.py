# -*- coding: utf-8 -*-

from gensim import corpora
from Utils import Utils
from Preprocessor import Preprocessor
from AspectExtractor import AspectExtractor
from SeededLDA import SeededLDA

'''
CONFIG _____________________________________________________
'''

# Parameters for GuidedLDA
n_topics = 5
n_iter = 1
alpha = 0.025929007560952073
eta = 0.24817736265848747
random_state = 1
seed_confidence = 1

# Parameters for Preprocessing
filename = 'train_data_part1_with_categories.json' # Filename of data to be analysed
only_nouns = True # Preprocessing will only keep nouns if turned on
tokenize = False # If Tokenization is required turn it on

# Paramters for analysing results:
score_at = 16   # Score only works for labled data.
score_at2 = 50 # Score only works for labled data.

'''
____________________________________________________________
'''

utils = Utils()
sentence_list = utils.get_tokens_from_json(filename)

preprocessing = Preprocessor()
result = preprocessing.preprocess(sentence_list, tokenize=tokenize, only_nouns=only_nouns)

aspect_extractor = AspectExtractor(preprocessing, filename, tokenize=tokenize, only_nouns=only_nouns)
aspect_extractor.extract()

##########################################################################################################
#################################   SETUP LDA  ###########################################################
##########################################################################################################

dictionary = corpora.Dictionary(result)
X = utils.get_term_matrix(result, dictionary)
vocab = [key for key in dictionary.token2id]
word2id = dictionary.token2id
seed_topic_list = utils.get_seeds_from_json('seeds.json', vocab)
aspect_topic_list = utils.get_seeds_from_json('aspects.json', vocab)

print(X.shape)
print(X.sum())

##########################################################################################################
#################################   LDA WITH SEEDING  ####################################################
##########################################################################################################

seeded_lda_ = SeededLDA()
topic_model = seeded_lda_.do_seeded_lda(X, vocab, word2id, n_topics, n_iter, alpha=alpha, eta=eta, random_state=random_state, seed_topic_list=seed_topic_list, seed_confidence=seed_confidence)

print('\n\n')

##########################################################################################################
#################################   Precision & Recall @score_at  ########################################
##########################################################################################################

recalls = []
precisions = []
for i in range(5):
    recalls.append(utils.calculate_recall(topic_model, i, vocab, aspect_topic_list[i], n_top_words=score_at))
    precisions.append(utils.calculate_precision(topic_model, i, vocab, aspect_topic_list[i], n_top_words=score_at))

print(f'RECALL @{score_at}: {utils.calculate_average(recalls)}')
print(f'PRECISION @{score_at}: {utils.calculate_average(precisions)}')
print(f'F-SCORE @{score_at}: {(utils.calculate_average(recalls)*utils.calculate_average(precisions)*2)/(utils.calculate_average(recalls)+utils.calculate_average(precisions))}')

##########################################################################################################
#################################   Unknown Word Recognition @score_at  ##################################
##########################################################################################################

count = 0
for i in range(5):
    count += utils.calculate_new_true_positives(topic_model, i, vocab, aspect_topic_list[i], seed_topic_list[i],n_top_words=score_at)
print(f'NEW TRUE POSITIVES @{score_at}: {count}')

print('\n\n')

##########################################################################################################
#################################   Precision & Recall @score_at2  #######################################
##########################################################################################################

recalls = []
precisions = []
for i in range(5):
    recalls.append(utils.calculate_recall(topic_model, i, vocab, aspect_topic_list[i], n_top_words=score_at2))
    precisions.append(utils.calculate_precision(topic_model, i, vocab, aspect_topic_list[i], n_top_words=score_at2))

print(f'RECALL @{score_at2}: {utils.calculate_average(recalls)}')
print(f'PRECISION @{score_at2}: {utils.calculate_average(precisions)}')
print(f'F-SCORE @{score_at2}: {(utils.calculate_average(recalls)*utils.calculate_average(precisions)*2)/(utils.calculate_average(recalls)+utils.calculate_average(precisions))}')

##########################################################################################################
#################################   Unknown Word Recognition @score_at2  #################################
##########################################################################################################

count = 0
for i in range(5):
    count += utils.calculate_new_true_positives(topic_model, i, vocab, aspect_topic_list[i], seed_topic_list[i], n_top_words=score_at2)
print(f'NEW TRUE POSITIVES @{score_at2}: {count}')
