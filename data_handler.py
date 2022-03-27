import torch
import pandas as pd
import os
from collections import Counter
import numpy as np

batch_size = 16
seq_size = 32

class DataHandler():
    def __init__(self, filename, batch_size=16, seq_size=32):
        self.batch_size = batch_size
        self.seq_size = seq_size
        with open (filename, 'r') as f:
            self.doc = f.read()

        self.words = self.word_parser(self.doc)
        self.embeddings, self.words_to_embeddings = self.dataset_embedder(self.words)

    def word_parser(self, doc):
        lines = doc.split('\n')
        lines = [line.replace(".", "").replace("?", "").replace(",", "").replace("!", "").replace(";", "").strip(r'\"') for line in lines]
        words = ' '.join(lines).split()
        return words

    def dataset_embedder(self, words):
        freqs = Counter(words)
        sorted_freqs = sorted(freqs, key=freqs.get) #get the value for each item
        embeddings = {w:k for w,k in enumerate(sorted_freqs)}
        words_to_embeddings = {k:w for w,k in enumerate(sorted_freqs)}
        return embeddings, words_to_embeddings

    def word_decoder(self, num):
        return self.embeddings[num]

    def word_embedder(self, word):
        return self.words_to_embeddings[word]

    def batcher(self):
        '''
        every X -> input[:-1]
        every Y -> input[1:]
        '''
        words = [self.word_embedder(word) for word in self.words]
        num_of_words = len(words)
        num_of_batches = num_of_words//(self.batch_size*self.seq_size)
        num_of_sequences = num_of_batches*self.batch_size
        X = words[:num_of_sequences*self.seq_size] #to make every batch at the same length
        Y = words[1:num_of_sequences*self.seq_size+1]
        X = np.reshape(X, (num_of_sequences, self.seq_size))
        Y = np.reshape(Y, (num_of_sequences, self.seq_size))

        batches = []
        
        for i in range(0, batch_size*num_of_batches, batch_size):
            batches.append([X[i:i+batch_size, :], Y[i:i+batch_size, :]])
        #batches' shape = (459, 2, 16, 32) -> 459: num_of_batches, 2: X and Y, 16: batch_size, 32: seq_size
        return np.asarray(batches)