import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import pickle


def read_glove_vecs(glove_file):
    with open(glove_file, 'r') as f:
        words = set()
        word_to_vec_map = {}
        for line in f:
            line = line.strip().split()
            curr_word = line[0]
            words.add(curr_word)
            word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)
        
        i = 1
        words_to_index = {}
        index_to_words = {}
        for w in sorted(words):
            words_to_index[w] = i
            index_to_words[i] = w
            i = i + 1
    return words_to_index, index_to_words, word_to_vec_map


word_to_index, index_to_word, word_to_vec_map = read_glove_vecs('glove.6B.300d.txt')

with open("word_to_index.pkl","wb") as f:
    pickle.dump(word_to_index,f)

with open("index_to_word.pkl","wb") as f:
    pickle.dump(index_to_word,f)

with open("word_to_vec_map.pkl","wb") as f:
    pickle.dump(word_to_vec_map,f)
