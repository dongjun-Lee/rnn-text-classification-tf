from gensim.models.keyedvectors import KeyedVectors
from gensim.test.utils import get_tmpfile
from gensim.scripts.glove2word2vec import glove2word2vec
import numpy as np


def get_init_embedding(reversed_dict, embedding_size):
    print("Loading Glove vectors...")
    glove_file = "glove/glove.6B.%dd.txt" % embedding_size
    word2vec_file = get_tmpfile("word2vec_format.vec")
    glove2word2vec(glove_file, word2vec_file)
    word_vectors = KeyedVectors.load_word2vec_format(word2vec_file)

    word_vec_list = list()
    for _, word in sorted(reversed_dict.items()):
        try:
            word_vec = word_vectors.word_vec(word)
        except KeyError:
            word_vec = np.zeros([embedding_size], dtype=np.float32)

        word_vec_list.append(word_vec)

    return np.array(word_vec_list)
