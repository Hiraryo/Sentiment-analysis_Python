# coding: utf-8
from gensim.models import word2vec
import logging
import sys
import numpy as np

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

class W2v():
    MODEL = None

    def __init__(self):
        if W2v.MODEL is None:
            #学習モデル読み込み
            W2v.MODEL = word2vec.Word2Vec.load('./wiki.model')



    def similarity(self, word1, word2):
        print("近似度計算: "+word1+" - "+word2)

        try:
            return W2v.MODEL.wv.similarity(word1, word2)
        except KeyError as e:
            print(e)
            return -1.0

    def getVec(self, word):
        try:
            return W2v.MODEL.wv[word]

        except KeyError:
            return np.zeros(200)

    #再学習、上書きを行うメソッド
    def updateTrain(self,corpus):
        sentences = word2vec.Text8Corpus(corpus)
        W2v.MODEL.build_vocab(sentences, update=True)
        W2v.MODEL.train(sentences, total_examples=W2v.MODEL.corpus_count, epochs=W2v.MODEL.iter)
        W2v.MODEL.save("wiki.model")

if __name__ == "__main__":
    w2v = W2v()

    word = "面白"

    w2v.updateTrain("corpus_omosiro.txt")

    print(word + "=")
    print(w2v.getVec(word))

    #単語数の確認
    print(len(W2v.MODEL.wv.vocab))
