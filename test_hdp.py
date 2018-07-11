from gensim.test.utils import common_corpus, common_dictionary
from gensim.models import HdpModel

__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'


hdp = HdpModel(corpus=common_corpus,
               id2word=common_dictionary)
unseen_document = [(1, 3.), (2, 4)]
doc_hdp = hdp[unseen_document]

topic_info = hdp.print_topics(num_topics=20,
                              num_words=10)

print(common_corpus)
print(common_dictionary)
print(hdp)
for topic in topic_info:
    print(topic)
