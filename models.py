import os, sys
import numpy as np
import tensorflow as tf
from tensorflow.contrib.crf import viterbi_decode
from tensorflow.contrib.crf import crf_log_likelihood
from modules import embedding, positional_encoding, \
                    multihead_attention, feedforward, \
                    label_smoothing, gelu_fast

from keras import backend as K
from keras.layers import Dense
from keras.objectives import categorical_crossentropy
from keras.metrics import categorical_accuracy as accuracy

class TransformerTagger:

    def __init__(self,
                 data_preparer,
                 num_blocks=6,
                 num_heads=8,
                 hidden_units=128,
                 vocab_size=9000,
                 emb_pos_type='sin',
                 lr=1e-2):
        self.data_preparer = data_preparer
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.hidden_units = hidden_units
        self.maxlen = data_preparer.length
        self.vocab_size = vocab_size
        self.emb_pos_type = emb_pos_type
        self.lr = lr

        assert hidden_units % num_heads == 0

    def build(self):
        x, y = self.create_placeholders()

        emb = self.build_embedding_layer(x)
        outs = self.build_blocks(emb, tf.to_float(tf.not_equal(x, self.data_preparer.vocab.get('[PAD]', 0))))
        logits = self.build_linear_projection_layer(outs)
        loss = self.compute_loss(logits, y)
        g_step, train_op = self.set_optimizer(loss)

        # tf.summary.scalar('acc', self.acc)
        # tf.summary.scalar('mean_loss', self.mean_loss)
        # self.merged = tf.summary.merge_all()
        merged = self.summary({
                'acc': self.crf_acc,
                'mean_loss': self.mean_loss,
            })

        sess = self.new_session()
        self.train_writer = tf.summary.FileWriter('./logs/train', sess.graph)

        return g_step, logits, train_op

    def summary(self, var_dict={}):
        for name, var in var_dict.items():
            tf.summary.scalar(name, var)
        self.merged = tf.summary.merge_all()
        return self.merged

    def new_session(self, sess=None):
        if hasattr(self, 'sess'):
            self.sess.close()

        if sess is None:
            config = tf.ConfigProto()
            config.allow_soft_placement = True
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)
        else:
            self.sess = sess

        self.sess.run(tf.global_variables_initializer())

        return self.sess

    def train_batch(self, x_batch, y_batch, z_batch, dropout=0.1):

        feed_dict = {
            self.x : x_batch,
            self.y : y_batch,
            self.z : z_batch,
            self.dropout : dropout,
        }

        if not hasattr(self, 'sess'):
            self.new_session()

        g_step, loss, crf_acc, cly_acc, _, summary, label_loss = self.sess.run(
            [self.global_step, self.mean_loss, self.crf_acc, self.cly_acc, self.train_op, self.merged, self.label_logits],
            feed_dict=feed_dict,
        )

        #print(label_loss)

        self.train_writer.add_summary(summary, g_step)

        return g_step, loss, crf_acc, cly_acc

    def predict(self, x, pretoken=False):
        
        matrix = self.sess.run(self.trans)
        
        if isinstance(x, str):
            x = self.data_preparer.sentence2idx(x, pretoken=pretoken)
            x = [[self.data_preparer.vocab['[CLS]']] + x + [self.data_preparer.vocab['[SEP]']]]
            
            x = self.data_preparer.pad_batch(x, self.maxlen)
            feed_dict = {
                self.x : x,
                self.dropout : 0.0,
            }

            if not hasattr(self, 'sess'):
                self.new_session()

            logits, lengths, label_logits = self.sess.run(
                [self.logits_no_cls_sep, self.lengths, self.label_logits],
                feed_dict=feed_dict,
            )
            lengths = lengths.astype(np.int32)
            paths = self.decode(logits, lengths, matrix)
            tags = [self.data_preparer.idx2tag[idx] for idx in paths[0]]
            
            return tags, label_logits
            
            #preds = np.argmax(logits, axis=-1)
            #return [self.data_preparer.idx2tag[i] for i in preds.flatten()]
        
        # x = [[self.data_preparer.vocab['[CLS]']] + self.data_preparer.sentence2idx(line, pretoken=pretoken) + [self.data_preparer.vocab['[SEP]']] for line in x]
        # x = self.data_preparer.pad_batch(x, self.maxlen)
        #
        # rets = []
        #
        # begin = 0
        # while begin < x.shape[0]:
        #
        #     feed_dict = {
        #         self.x : x[begin:begin+16],
        #         self.dropout : 0.0,
        #     }
        #
        #     if not hasattr(self, 'sess'):
        #         self.new_session()
        #
        #     logits, lengths = self.sess.run(
        #         [self.logits_no_cls_sep, self.lengths,],
        #         feed_dict=feed_dict,
        #     )
        #     lengths = lengths.astype(np.int32)
        #     #rets += [[self.data_preparer.idx2tag[i] for i in preds[j]] for j in range(preds.shape[0])]
        #     paths = self.decode(logits, lengths, matrix)
        #     tags = [[self.data_preparer.idx2tag[idx] for idx in path] for path in paths]
        #
        #     rets += tags
        #
        #     begin += 16
        #
        # print('PPPPPPPPPPPPPPPP')
        # print(rets)
        # print(label_logits)
        # return rets, label_logits
    
    def decode(self, logits, lengths, matrix):
        """
        :param logits: [batch_size, num_steps, num_tags]float32, logits
        :param lengths: [batch_size]int32, real length of each sequence
        :param matrix: transaction matrix for inference
        :return:
        """
        # inference final labels usa viterbi Algorithm
        paths = []
        #small = -1000.0
        #start = np.asarray([[small]*self.num_tags +[0]])
        for score, length in zip(logits, lengths):
            score = score[:length]
            logits = score
            #pad = small * np.ones([length, 1])
            #logits = np.concatenate([score, pad], axis=1)
            #logits = np.concatenate([start, logits], axis=0)
            path, _ = viterbi_decode(logits, matrix)
    
            #paths.append(path[1:])
            paths.append(path)
        return paths

    def save(self, path):
        if not hasattr(self, 'saver'):
            self.saver = tf.train.Saver()
        self.saver.save(self.sess, path)

    def load(self, path):
        if not hasattr(self, 'saver'):
            self.saver = tf.train.Saver()
        if not hasattr(self, 'sess'):
            self.new_session()
        self.saver.restore(self.sess, path)

    def create_placeholders(self):
        # input and target
        self.x = tf.placeholder(tf.int32, shape=(None, self.maxlen))
        self.y = tf.placeholder(tf.int32, shape=(None, self.maxlen))
        self.z = tf.placeholder(tf.int32, shape=(None, 4))
        # dropout
        self.dropout = tf.placeholder(tf.float32,)

        return self.x, self.y, self.z

    def build_embedding_layer(self, inputs, reuse=None):
        self.emb_char = embedding(inputs,
                                  vocab_size=self.vocab_size,
                                  num_units=self.hidden_units,
                                  scale=True,
                                  scope="emb_char",
                                  reuse=reuse)
        self.emb_char_pos = self.emb_char
        if self.emb_pos_type == 'sin':
            self.emb_char_pos += positional_encoding(inputs,
                                                     num_units=self.hidden_units,
                                                     zero_pad=False,
                                                     scale=False,
                                                     scope="emb_pos",
                                                     reuse=reuse)
        else:
            self.emb_char_pos += embedding(tf.tile(tf.expand_dims(tf.range(tf.shape(inputs)[1]), 0), [tf.shape(inputs)[0], 1]),
                                           vocab_size=self.maxlen,
                                           num_units=self.hidden_units,
                                           zero_pad=False,
                                           scale=False,
                                           scope="emb_pos",
                                           reuse=reuse)

        self.emb = tf.layers.dropout(self.emb_char_pos, rate=self.dropout,)

        return self.emb

    def build_blocks(self, inputs, masks, reuse=None):
        self.blk = inputs
        for i in range(self.num_blocks):
            with tf.variable_scope("blocks_{}".format(i), reuse=reuse):
                ## Multihead Attention ( self-attention)
                self.blk = multihead_attention(queries=self.blk,
                                               keys=self.blk,
                                               qkv_masks=masks,
                                               num_units=self.hidden_units,
                                               num_heads=self.num_heads,
                                               dropout_rate=self.dropout,
                                               # is_training=is_training,
                                               causality=False,
                                               scope="self_attention",
                                               reuse=reuse)
                self.blk = feedforward(self.blk, num_units=[4*self.hidden_units, self.hidden_units], reuse=reuse)

        return self.blk

    def build_linear_projection_layer(self, inputs, reuse=None):
        self.logits = tf.layers.dense(inputs, len(self.data_preparer.tag2idx), name='logits', reuse=reuse)
        return self.logits

    def attention(self, inputs, attention_size=768, time_major=False):
        if isinstance(inputs, tuple):
            inputs = tf.concat(inputs, 2)
        if time_major:  # (T,B,D) => (B,T,D)
            inputs = tf.transpose(inputs, [1, 0, 2])
        hidden_size = inputs.shape[2].value
        # Trainable parameters
        w_omega = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))
        b_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
        u_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
        v = tf.tanh(tf.tensordot(inputs, w_omega, axes=1) + b_omega)

        vu = tf.tensordot(v, u_omega, axes=1, name='vu')  # (B,T) shape
        alphas = tf.nn.softmax(vu, name='alphas')  # (B,T) shape
        # the result has (B,D) shape
        output = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 1)

        return output

    def build_dense_layer(self, inputs, filter_sizes, num_filters, reuse=None):
        # self.out1 = tf.layers.dense(inputs, 100, name='out1', reuse=reuse)
        # self.out1 = tf.nn.relu(self.out1)
        # self.label_logits = tf.layers.dense(self.out1, len(self.data_preparer.label2id), name='label_logits', reuse=reuse)
        # print('YYYYYYYYYYYYYYYYYYYYYYYYY')
        # print(np.shape(self.label_logits))

        # Create a convolution + maxpool layer for each filter size

        inputs = tf.expand_dims(inputs, -1)
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, 768, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.02), name="W")
                b = tf.Variable(tf.constant(0.01, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    inputs,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, 512 - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, keep_prob=0.8)

        return self.h_drop

    def build_dense_layer(self, inputs, filter_sizes, num_filters, reuse=None):
        num_filters_total = num_filters * len(filter_sizes)

        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, 4],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.01, shape=[4]), name="b")
            self.label_logits = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")

        return self.label_logits

    def compute_label_loss(self, logits, labels):
        # skip [CLS] at the beginning and [SEP] at the end
        # logits = logits[:, :]
        # labels = labels[:, :]
        self.cly_acc = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(logits, axis=1, output_type=tf.int32), tf.argmax(labels, axis=1, output_type=tf.int32)), "float")) / tf.reduce_sum(tf.cast(tf.equal(tf.argmax(labels, axis=1, output_type=tf.int32), tf.argmax(labels, axis=1, output_type=tf.int32)), "float"))

        # self.debug_var = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(logits, axis=1, output_type=tf.int32), tf.reshape(labels, [-1])), "float"))

        self.label_loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels)
        self.label_mean_loss = tf.reduce_mean(self.label_loss)

        # self.y_smoothed = tf.one_hot(labels, depth=len(self.data_preparer.tag2idx)) #label_smoothing(tf.one_hot(labels, depth=len(self.data_preparer.tag2idx)))
        # self.loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=self.y_smoothed)
        # self.mean_loss = tf.reduce_mean(self.loss*self.istarget)

        return self.label_mean_loss

    def compute_loss(self, logits, labels):
        # skip [CLS] at the beginning and [SEP] at the end
        logits = logits[:, 1:-1,:]
        labels = labels[:, :-2]
        self.logits_no_cls_sep = logits
        self.istarget = tf.to_float(tf.not_equal(self.x, self.data_preparer.vocab['[PAD]'])[:, 1:-1])
        self.lengths = tf.reduce_sum(self.istarget, axis=-1)
        self.preds = tf.to_int32(tf.argmax(logits, axis=-1))
        self.crf_acc = tf.reduce_sum(tf.to_float(tf.equal(self.preds, labels))*self.istarget) / tf.reduce_sum(self.istarget)

        self.trans = tf.get_variable(
                    "transitions",
                    shape=[len(self.data_preparer.tag2idx), len(self.data_preparer.tag2idx)],)
        log_likelihood, self.trans = crf_log_likelihood(
                    inputs=logits,
                    tag_indices=labels,
                    transition_params=self.trans,
                    sequence_lengths=self.lengths)
        self.loss = -log_likelihood
        self.mean_loss = tf.reduce_mean(self.loss)
                                        
        # self.y_smoothed = tf.one_hot(labels, depth=len(self.data_preparer.tag2idx)) #label_smoothing(tf.one_hot(labels, depth=len(self.data_preparer.tag2idx)))
        # self.loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=self.y_smoothed)
        # self.mean_loss = tf.reduce_mean(self.loss*self.istarget)

        return self.mean_loss

    def merge_loss(self, loss1, loss2):

        self.all_loss = loss1 + loss2

        return self.all_loss

    def set_optimizer(self, loss):
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.9, beta2=0.98, epsilon=1e-8)
        # grads = self.optimizer.compute_gradients(loss)
        # for i, (g, v) in enumerate(grads):
        #     if g is not None:
        #         grads[i] = (tf.clip_by_norm(g, 5), v)  # 阈值这里设为5
        # self.train_op = self.optimizer.apply_gradients(grads)

        self.train_op = self.optimizer.minimize(loss, global_step=self.global_step)
        return self.global_step, self.train_op
    
    
    
    
