
import os, sys
import re
import random
import numpy as np
from collections import defaultdict
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

from utils import pickle_save, pickle_load

class DataPreparer:
    
    def __init__(self, cls2idx1, cls2idx2, vocab_path='./vocab', length=32):
        self.length = length
        self.cls2idx1 = cls2idx1
        self.cls2idx2 = cls2idx2

        self.idx2cls1 = {a:b for b,a in self.cls2idx1.items()}
        self.idx2cls2 = {a:b for b,a in self.cls2idx2.items()}

        self.corpus = []
        
    def load_all_corpus(self, corpus_dir, maintain_vocab=True):
        



    
    def gen_batch(self, batch_size=128, shuffle=True):
        
        if shuffle:
            random.shuffle(self.corpus)
        
        begin = 0
        end = begin + batch_size
        while end < len(self.corpus):
            batch_sentence, batch_tags, batch_labels = zip(*self.corpus[begin:end])
            batch_idxs = [[self.vocab['[CLS]']] + self.sentence2idx(s) + [self.vocab['[SEP]']] for s in batch_sentence]
            batch_idxs_with_pads = self.pad_batch(batch_idxs, length=self.length)
            batch_tags = [self.get_tag2idx(tags) for tags in batch_tags]
            batch_tags_with_pads = self.pad_batch(batch_tags, length=self.length)
            batch_labels_pro = [item for item in batch_labels]
            yield batch_idxs_with_pads, batch_tags_with_pads, batch_labels_pro
            begin = end
            end += batch_size
            
            
            
            
            
            
            
            
            