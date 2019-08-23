
import os, sys
import re
import random
import numpy as np
from collections import defaultdict
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

from utils import pickle_save, pickle_load

class DataPreparer:
    
    def __init__(self, category2idx, label2idx, vocab_path='./vocab', length=32):
        self.length = length
        self.category2idx = category2idx
        self.label2idx = label2idx
        self.idx2category = {a:b for b,a in self.category2idx.items()}
        self.idx2label = {a:b for b,a in self.label2idx.items()}
        self.corpus = []
        
    def load_all_corpus(self, corpus_dir, maintain_vocab=True):
        file = open(corpus_dir, 'r')

        flag = 0
        for line in file:
            if flag == 0:
                flag = 1
                continue

            inputs = line.strip().split(',')

            if len(inputs) < 3:
                print('inputs error !!!')
                print(inputs)
                continue

            category = self.category2idx[inputs[0]]
            # category = inputs[0]
            label = self.label2idx[int(inputs[1])]

            sents = ''
            for i in range(2, len(inputs)):
                sents += inputs[2]

            self.corpus.append((category, label, sents))

    def gen_batch(self, batch_size=128, shuffle=True):
        
        if shuffle:
            random.shuffle(self.corpus)
        
        begin = 0
        end = begin + batch_size
        while end < len(self.corpus):
            batch_category, batch_labels, batch_sentence = zip(*self.corpus[begin:end])
            batch_idxs = [[self.vocab['[CLS]']] + self.sentence2idx(s) + [self.vocab['[SEP]']] for s in batch_sentence]
            batch_idxs_with_pads = self.pad_batch(batch_idxs, length=self.length)
            batch_category_pro = [item for item in batch_category]
            batch_labels_pro = [item for item in batch_labels]
            yield batch_idxs_with_pads, batch_category_pro, batch_labels_pro
            begin = end
            end += batch_size
            
            
            
            
            
            
            
            
            