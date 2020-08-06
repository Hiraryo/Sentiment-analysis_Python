import gensim
import pprint

word2vec_model = gensim.models.KeyedVectors.load_word2vec_format('model.vec', binary=False)

pprint.pprint(word2vec_model.similarity('感動', '鳥肌'))
pprint.pprint(word2vec_model.similarity('感動', 'ざわ'))
pprint.pprint(word2vec_model.similarity('感動', '涙'))
pprint.pprint(word2vec_model.similarity('感動', '涙腺'))
pprint.pprint(word2vec_model.similarity('感動', 'ゲリラ'))
