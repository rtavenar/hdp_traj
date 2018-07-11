from gensim.models import HdpModel
import os

from utils import read_traj_synthetic, ObsQuantizer

__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'


corpus_raw = []
path = "data/toy_pierre"
max_iter = 100

for fname in os.listdir(path):
    if not fname.endswith(".txt"):
        continue
    fullname = os.path.join(path, fname)
    traj = read_traj_synthetic(fullname)
    corpus_raw.append(traj)

oq = ObsQuantizer(min_x=-25, max_x=5, min_y=-10, max_y=10)
corpus_gensim = oq.fit(corpus_raw)


hdp = HdpModel(corpus=corpus_gensim,
               id2word=oq.dictionary)
for iter in range(max_iter):
    hdp.update(corpus=corpus_gensim)

topic_info = hdp.print_topics(num_topics=20,
                              num_words=10)

print(hdp)
for topic in topic_info:
    print(topic)
