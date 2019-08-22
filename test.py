from tensorflow.contrib.crf import viterbi_decode
from torch.optim.adam import Adam
from torch.optim.sgd import SGD
from flair.data import Corpus, MultiCorpus
from flair.datasets import ColumnCorpus, ClassificationCorpus
from flair.embeddings import BertEmbeddings, DocumentRNNEmbeddings
from flair.data import Sentence
from flair.models import SequenceTagger, TextClassifier
from flair.trainers import ModelTrainer
from flair.optim import SGDW, AdamW
from flair.training_utils import clear_embeddings
from copy import deepcopy
import random


# this is the folder in which train, test and dev files reside
data_folder = './'

# init a corpus using column format, data folder and the names of the train, dev and test files
corpus: Corpus = ClassificationCorpus(
    data_folder,
    train_file='cls.train.long.txt',
    test_file='cls.test.long.txt',
    dev_file='cls.test.long.txt')

label_dict = corpus.make_label_dictionary()

embedding = BertEmbeddings('bert-base-chinese', layers='-1')
document_embeddings = DocumentRNNEmbeddings([embedding])

classifier = TextClassifier(document_embeddings, label_dictionary=label_dict)

expanded_corpus = Corpus(train=list(corpus.train)*10, test=corpus.test, dev=corpus.dev)

trainer: ModelTrainer = ModelTrainer(classifier, expanded_corpus, optimizer=torch.optim.Adam)

trainer.train('./cls_ckpt_long/',
              learning_rate=1e-3,
              mini_batch_size=8,
              max_epochs=80)

classifier = TextClassifier.load('./cls_ckpt_long/best-model.pt')

ret = classifier.evaluate(corpus.test)
result = ret[0]
print(result.detailed_results)