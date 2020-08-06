# coding: utf-8
from gensim.models import word2vec
import logging
import sys
import numpy as np
import MeCab
import sys
import re
import gensim
import pprint
from collections import Counter
import cython
from gensim.models import KeyedVectors

# ファイル読み込み
cmd, infile = sys.argv
with open(infile) as f:
    data = f.read()

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

class W2v():
    MODEL = None

    def __init__(self):
        if W2v.MODEL is None:
            #学習モデル読み込み
            W2v.MODEL = KeyedVectors.load('./wiki.model')



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

# -------------------------------------------------
# 学習済みモデルの読み込み
word2vec_model = W2v.MODEL

# パース
mecab = MeCab.Tagger()
parse = mecab.parse(data)
lines = parse.split('\n')
items = (re.split('[\t,]', line) for line in lines)

# 名詞をリストに格納
words = [item[0]
         for item in items
         if (item[0] not in ('EOS', '', 't', 'ー') and
             item[1] == '名詞')]

# 頻度順に出力
counter = Counter(words)
kanjou = ''
for i in range(4):
    sum = 0.0
    ave = 0.0
    num = 0
    if (i == 0):
        kanjou = '感動'
    elif (i == 1):
        kanjou = '驚愕'
    elif (i == 2):
        kanjou = '面白'
    elif (i == 3):
        kanjou = '萌え'
    for word, count in counter.most_common():
        num += 1
        if (num <= 10):
            print(f"{word}")
            print(kanjou,"の時の類似度は、")
            pprint.pprint(word2vec_model.similarity(kanjou, f"{word}"))
            sum += word2vec_model.similarity(kanjou, f"{word}")
        ave = sum / 10
    print(kanjou,'の平均は',ave,'です。')
    i += 1
