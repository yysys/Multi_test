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
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

class Model(TransformerTagger):

    def build(self):
        # inputs and labels
        x, y, z = self.create_placeholders()

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

        embedding = model.get_sequence_output()
        # construct output with a dense layer
        logits = self.build_linear_projection_layer(embedding)

        # compute loss
        loss = self.compute_loss(logits, y)
        #g_step, train_op = self.set_optimizer(loss)

        ## lable_logits
        ## out = self.attention(embedding)
        label_logits = self.build_dense_layer(embedding, filter_sizes=[3,4,5], num_filters=128)

        # compute loss
        label_loss = self.compute_label_loss(label_logits, z)

        all_loss = self.merge_loss(loss, label_loss)

        g_step, train_op = self.set_optimizer(all_loss)

        # add acc and loss to summary
        merged = self.summary({
            'acc': self.cly_acc,
            'mean_loss': self.label_mean_loss,
        })

        sess = self.new_session()
        self.train_writer = tf.summary.FileWriter('./logs/train', sess.graph)

        #return g_step, logits, train_op


# cls2idx = {'书籍': 0, '平板': 1, '手机': 2, '水果': 3, '洗发水': 4, '热水器': 5, '蒙牛': 6, '衣服': 7, '计算机': 8, '酒店': 9}
category2idx = {'书籍': [0,0,0,0,0,0,0,0,0,1],
           '平板': [0,0,0,0,0,0,0,0,1,0],
           '手机': [0,0,0,0,0,0,0,1,0,0],
           '水果': [0,0,0,0,0,0,1,0,0,0],
         '洗发水': [0,0,0,0,0,1,0,0,0,0],
         '热水器': [0,0,0,0,1,0,0,0,0,0],
           '蒙牛': [0,0,0,1,0,0,0,0,0,0],
           '衣服': [0,0,1,0,0,0,0,0,0,0],
         '计算机': [0,1,0,0,0,0,0,0,0,0],
           '酒店': [1,0,0,0,0,0,0,0,0,0]}

label2idx = {0:[0,1], 1:[1,0]}


dp = DataPreparer(category2idx=category2idx, label2idx=label2idx, vocab_path=PATH_VOCAB)
dp.load_all_corpus('./data/data.csv', maintain_vocab=False)
print('------------------------------------------------')
print(len(dp.corpus))
# split testset and trainset
split_id = len(dp.corpus) * 0.8
random.shuffle(dp.corpus)
testset, dp.corpus = dp.corpus[:split_id], dp.corpus[split_id:]

# build model
tf.reset_default_graph()
lr = tf.Variable(0.5e-4, trainable=False, name='lr')
model = Model(data_preparer=dp, num_blocks=12, hidden_units=768, num_heads=12, lr=lr)

model.build()


# load pre-train parameters
load_variables = [v for v in tf.trainable_variables() if 'bert' in v.name]
loader = tf.train.Saver(load_variables)
loader.restore(model.sess, PATH_MODEL)

# # feeze pre-train layer
# model.train_op = model.optimizer.minimize(model.all_loss,
#                                           global_step=model.global_step,
#                                           var_list=[v for v in tf.trainable_variables() if v not in load_variables])
# model.sess.run(tf.assign(lr, 1e-3))


# feeze part of pre-train layer
model.train_op = model.optimizer.minimize(model.all_loss,
                                          global_step=model.global_step,
                                          var_list=[v for v in tf.trainable_variables() if v not in load_variables \
                                                       or 'layer_11' in v.name \
                                                       or 'layer_10' in v.name])
model.sess.run(tf.assign(lr, 5e-4))

'''
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
    crf_accs = []
    cly_accs = []
    tic = time()
    count_batch = 0
    batch_size = 32
    for _ in range(100):
        for x, y, z in dp.gen_batch(batch_size=batch_size, shuffle=True):
            g_step, loss, crf_acc, cly_acc = model.train_batch(x, y, z, 0.1)
            losses.append(loss)
            crf_accs.append(crf_acc)
            cly_accs.append(cly_acc)

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
                      f"avg_time:{(toc-tic)/period:.4f}, avg_loss:{np.mean(losses):.4f}, avg_crf_acc:{np.mean(crf_accs):.4f}, avg_cly_acc:{np.mean(cly_accs):.4f}\n")
                print(f"progress:{count_batch*batch_size/(len(dp.corpus))*100:.2f}%, global_step:{g_step}, "
                      f"avg_time:{(toc-tic)/period:.4f}, avg_loss:{np.mean(losses):.4f}, avg_crf_acc:{np.mean(crf_accs):.4f}, avg_cly_acc:{np.mean(cly_accs):.4f}\n")
                tic = time()
                losses = []
                cly_accs = []
                crf_accs = []

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
# cls_confusion_matrix = {'tp':0, 'tn':0, 'fp':0, 'fn':0}
cls_correct = 0
cls_sum = 0

for x, y, z in testset:
    n += 1
    x = ''.join(x[:512])
    y = y[:512]
    y_group = tag2group(y)
    crf_pred, cls_pred = model.predict(x)
    pred_group = tag2group(crf_pred)

    cls_true = np.argmax(z)
    cls_pred = np.argmax(cls_pred)

    if cls_true == cls_pred:
        cls_correct += 1
    cls_sum += 1

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

    if n % 20 == 0:
        print(f'{n}/{len(testset)}')
        for i, t in enumerate(['姓名', '车型', '案发地', '案发时间', '酒精含量']):
            recall = counts[i, 0] / counts[i, 1]
            precision = counts[i, 0] / counts[i, 2]
            f1 = 2 / (1 / recall + 1 / precision)
            print(f'| {t} | {f1*100:.2f}% | {precision*100:.2f}% | {recall*100:.2f}% |')
        print(f'| 查获肇事准确率 | {1.0*cls_correct/cls_sum:.2f}%')
        cls_correct = 0
        cls_sum = 0

        clear_output(wait=True)

