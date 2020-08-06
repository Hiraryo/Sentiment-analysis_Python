import MeCab
import sys
import re
import gensim
import pprint
from collections import Counter


# ファイル読み込み
cmd, infile = sys.argv
with open(infile) as f:
    data = f.read()

# 学習済みモデルの読み込み
word2vec_model = gensim.models.KeyedVectors.load_word2vec_format('model.vec', binary=False)

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
