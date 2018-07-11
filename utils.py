import json
import csv
from gensim.corpora import Dictionary

__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'


def read_traj_ais(fname):
    traj_json = json.load(open(fname, "r"))
    print(traj_json)


def read_traj_synthetic(fname):
    traj_csv = []
    for obs in csv.DictReader(open(fname, "r", encoding="utf-8"), delimiter=";"):
        traj_csv.append(obs)
    return traj_csv


class ObsQuantizer:
    def __init__(self, min_x=-25, max_x=25, min_y=-25, max_y=25, n_cells_x=10, n_cells_y=10):
        self.min_x = min_x
        self.max_x = max_x
        self.min_y = min_y
        self.max_y = max_y
        self.n_cells_x = n_cells_x
        self.n_cells_y = n_cells_y

        self.dictionary = Dictionary([["x%d_y%d" % (x, y) for y in range(self.n_cells_y)]
                                      for x in range(self.n_cells_x)])

    @property
    def delta_x(self):
        return (self.max_x - self.min_x) / self.n_cells_x

    @property
    def delta_y(self):
        return (self.max_y - self.min_y) / self.n_cells_y

    def fit_one_doc(self, doc):
        doc_list = []
        for obs in doc:
            quantized_x = (float(obs["Vx"]) - self.min_x) // self.delta_x
            quantized_y = (float(obs["Vy"]) - self.min_y) // self.delta_y
            if quantized_x < 0:
                quantized_x = 0
            if quantized_x > self.n_cells_x - 1:
                quantized_x = self.n_cells_x - 1
            if quantized_y < 0:
                quantized_y = 0
            if quantized_y > self.n_cells_y - 1:
                quantized_y = self.n_cells_y - 1
            word = "x%d_y%d" % (quantized_x, quantized_y)
            doc_list.append(word)
        return self.dictionary.doc2bow(document=doc_list)


    def fit(self, corpus):
        corpus_gensim = []
        for doc in corpus:
            corpus_gensim.append(self.fit_one_doc(doc))
        return corpus_gensim



if __name__ == "__main__":
    traj1 = read_traj_synthetic("data/toy_pierre/T1.txt")
    traj2 = read_traj_synthetic("data/toy_pierre/T2.txt")
    corpus_raw = [traj1, traj2]
    oq = ObsQuantizer(min_x=-25, max_x=5, min_y=-10, max_y=10)
    corpus_gensim = oq.fit(corpus_raw)

    for doc in corpus_gensim:
        print("New doc")
        for term, freq in doc:
            print(oq.dictionary[term], freq)