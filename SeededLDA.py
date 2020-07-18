# -*- coding: utf-8 -*-

import numpy as np
import guidedlda


class SeededLDA():

    def do_seeded_lda(self, X, vocab, word2id, n_topics, n_iter, alpha, eta, random_state, seed_topic_list, seed_confidence):
        model = guidedlda.GuidedLDA(n_topics=n_topics, n_iter=n_iter, alpha=alpha, eta=eta, random_state=random_state, refresh=1)

        seed_topics = {}
        for t_id, st in enumerate(seed_topic_list):
            for word in st:
                seed_topics[word2id[word]] = t_id

        model.fit(X, seed_topics=seed_topics, seed_confidence=seed_confidence)

        n_top_words = 10
        topic_word = model.topic_word_
        for i, topic_dist in enumerate(topic_word):
            topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
            print('Topic {}: {}'.format(i, ' '.join(topic_words)))
        return model.topic_word_
