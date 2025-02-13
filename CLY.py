import sys, os
# 0 ~ 9
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from IPython.display import clear_output
clear_output(wait=False)

import re
import random
import tensorflow as tf
import numpy as np
from time import time
from collections import defaultdict

from models import TransformerTagger
from data_preparer import DataPreparer
from utils import get_available_gpus, pickle_load, pickle_save
import modeling

DIR_BERT = './std_ckpt/'
PATH_VOCAB = os.path.join(DIR_BERT, 'vocab.txt')
PATH_MODEL = os.path.join(DIR_BERT, 'bert_model.ckpt')
PATH_CONFIG = os.path.join(DIR_BERT, 'bert_config.json')
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

class DataPreparer2(DataPreparer):
    def __init__(self, tag2idx, vocab_path=PATH_VOCAB, length=32):
        self.length = length
        self.vocab = {}
        with open(vocab_path, 'r', encoding='utf-8') as f:
            for token in f:
                self.vocab[token.strip()] = len(self.vocab)
        self.idx2char = {i:c for c,i in self.vocab.items()}
        self.tag2idx = tag2idx
        self.idx2tag = {i:t for t,i in self.tag2idx.items()}
        self.corpus = []


class Model(TransformerTagger):

    def build(self):
        # inputs and labels
        x, y = self.create_placeholders()

        # construct bert model
        bert_config = modeling.BertConfig.from_json_file(PATH_CONFIG)
        bert_config.hidden_dropout_prob = self.dropout
        bert_config.attention_probs_dropout_prob = self.dropout
        input_ids = x
        input_mask = tf.to_float(tf.not_equal(x, self.data_preparer.vocab.get('[PAD]', 0)))
        model = modeling.BertModel(
            config=bert_config,
            is_training=True,
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_ids=None,
            use_one_hot_embeddings=True, )

        # construct output with a dense layer
        logits = self.build_linear_projection_layer(model.get_sequence_output())

        # compute loss
        loss = self.compute_loss(logits, y)
        g_step, train_op = self.set_optimizer(loss)

        # add acc and loss to summary
        merged = self.summary({
            'acc': self.acc,
            'mean_loss': self.mean_loss,
        })

        sess = self.new_session()
        self.train_writer = tf.summary.FileWriter('./logs/train', sess.graph)

        return g_step, logits, train_op


def tag2group(tags):
    '''
    ['O', 'O', 'B-XXX', 'M-XXX', 'E-XXX', 'O', 'S-YYY']
        ==>
    {
        'XXX': [(2, 5)],
        'YYY': [(6, 7)],
    }
    '''
    groups = defaultdict(set)
    last = None
    for i, tag in enumerate(tags):
        if tag == 'O':
            last = None
            continue
        tag_head, tag_type = tag.split('-')

        if tag_head == 'S':
            groups[tag_type].add((i, i + 1))
            last = None

        elif tag_head == 'B':
            last = [tag_type, i, None]

        elif tag_head == 'I':
            if last and last[0] == tag_type:
                pass
            else:
                last = None

        elif tag_head == 'E':
            if last and last[0] == tag_type:
                last[-1] = i + 1
                groups[tag_type].add((last[1], last[2]))
            last = None

    return groups
'''
tag2idx = {'O': 0,
           'I-机构名': 1,
           'I-日期': 2,
           'I-货币': 3,
           'B-机构名': 4,
           'E-机构名': 5,
           'I-百分比': 6,
           'B-日期': 7,
           'E-日期': 8,
           'B-百分比': 9,
           'E-百分比': 10,
           'B-货币': 11,
           'E-货币': 12,
           'B-地名': 13,
           'E-地名': 14,
           'B-人名': 15,
           'E-人名': 16,
           'I-地名': 17,
           'I-人名': 18,
           'S-地名': 19,
           'I-时间': 20,
           'B-时间': 21,
           'E-时间': 22,
           'S-人名': 23,}
'''
tag2idx = {'B-案发时间': 0, 'I-案发时间': 1, 'E-案发时间': 2, 'O': 3, 'B-姓名': 4, 'I-姓名': 5, 'E-姓名': 6, 'B-车型': 7, 'I-车型': 8, 'E-车型': 9, 'B-案发地': 10, 'I-案发地': 11, 'E-案发地': 12, 'B-酒精含量': 13, 'I-酒精含量': 14, 'E-酒精含量': 15}

dp = DataPreparer2(length=512, tag2idx=tag2idx, vocab_path=PATH_VOCAB)
dp.load_all_corpus('./train_corpus/', maintain_vocab=False)
dp2 = DataPreparer2(length=512, tag2idx=tag2idx, vocab_path=PATH_VOCAB)
dp2.load_all_corpus('./test_corpus/', maintain_vocab=False)
print('------------------------------------------------')
print(len(dp.corpus))
# split testset and trainset
random.shuffle(dp.corpus)
testset = dp2.corpus

# build model
tf.reset_default_graph()
lr = tf.Variable(0.5e-4, trainable=False, name='lr')
model = Model(data_preparer=dp, num_blocks=12, hidden_units=768, num_heads=12, lr=lr)

model.build()


# load pre-train parameters
load_variables = [v for v in tf.trainable_variables() if 'bert' in v.name]
loader = tf.train.Saver(load_variables)
loader.restore(model.sess, PATH_MODEL)

# feeze pre-train layer
model.train_op = model.optimizer.minimize(model.mean_loss,
                                          global_step=model.global_step,
                                          var_list=[v for v in tf.trainable_variables() if v not in load_variables])
model.sess.run(tf.assign(lr, 1e-4))

'''
# feeze part of pre-train layer
model.train_op = model.optimizer.minimize(model.mean_loss,
                                          global_step=model.global_step,
                                          var_list=[v for v in tf.trainable_variables() if v not in load_variables \
                                                       or 'layer_11' in v.name \
                                                       or 'layer_10' in v.name])
model.sess.run(tf.assign(lr, 5e-5))

# train all variables jointly
model.train_op = model.optimizer.minimize(model.mean_loss,
                                          global_step=model.global_step,
                                          var_list=tf.trainable_variables())
model.sess.run(tf.assign(lr, 1e-5))
'''
# START TRAIN !
with open('./log.txt', 'w', encoding='utf-8') as fout:
    period = 20
    save_period = 500
    losses = []
    accs = []
    tic = time()
    count_batch = 0
    batch_size = 128
    for _ in range(10):
        for x, y in dp.gen_batch(batch_size=batch_size, shuffle=True):
            g_step, loss, acc = model.train_batch(x, y, 0.1)
            losses.append(loss)
            accs.append(acc)

            count_batch += 1
#             if g_step % save_period == 0:
#                 acc = run_test()
#                 print('ACC: ', acc)
#                 if acc > max_acc:
#                     max_acc = acc
#                     model_prefix = 'fine_'10.214.4.211
#                     save_path = f"./ckpts/{model_prefix}model_acc{max_acc:.4f}"
#                     model.save(save_path)
#                     print(f'New max_acc, save to \'{save_path}\'')
            if g_step % period == 0:
                toc = time()
                fout.write(f"progress:{count_batch*batch_size/(len(dp.corpus))*100:.2f}%, global_step:{g_step}, "
                      f"avg_time:{(toc-tic)/period:.4f}, avg_loss:{np.mean(losses):.4f}, avg_acc:{np.mean(accs):.4f}\n")
                print(f"progress:{count_batch*batch_size/(len(dp.corpus))*100:.2f}%, global_step:{g_step}, "
                      f"avg_time:{(toc-tic)/period:.4f}, avg_loss:{np.mean(losses):.4f}, avg_acc:{np.mean(accs):.4f}\n")
                tic = time()
                losses = []
                accs = []

# save model
model_prefix = '80trainset_'
save_path = "./ckpts/{model_prefix}model"
model.save(save_path)

# save trainset and testset
pickle_save(testset, path='ckpts/20testset.pkl')
pickle_save(dp.corpus, path='ckpts/80trainset.pkl')

# test
types = ['姓名', '车型', '案发地', '案发时间', '酒精含量']
# ['人名', '地名', '日期', '时间', '机构名', '百分比', '货币']
type2idx = {t: i for i, t in enumerate(types)}

counts = np.zeros([len(types), 3], dtype=np.int32)
n = 0
for x, y in testset:
    n += 1
    x = ''.join(x[:512])
    y = y[:512]
    y_group = tag2group(y)
    pred_group = tag2group(model.predict(x))

    #     if y_group != pred_group:
    #         print(x)
    #         print(y_group)
    #         print(pred_group)
    #         print()
    #     diff_group = y_group.intersection(pred_group)

    for i, t in enumerate(types):
        correct = y_group[t].intersection(pred_group[t])
        counts[i, 0] += len(correct)
        counts[i, 1] += len(y_group[t])
        counts[i, 2] += len(pred_group[t])

    if n % 50 == 0:
        print(f'{n}/{len(testset)}')
        for i, t in enumerate(['姓名', '车型', '案发地', '案发时间', '酒精含量']):
            recall = counts[i, 0] / counts[i, 1]
            precision = counts[i, 0] / counts[i, 2]
            f1 = 2 / (1 / recall + 1 / precision)
            print(f'| {t} | {f1*100:.2f}% | {precision*100:.2f}% | {recall*100:.2f}% |')
        clear_output(wait=True)

