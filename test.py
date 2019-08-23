
from data_preparer import *


t = DataPreparer({})
t.load_all_corpus('./data/data.csv')

t_dict = {}

cnt = 0
for item in t.corpus:
    if item[0] not in t_dict:
        t_dict[item[0]] = cnt
        cnt += 1

print(t_dict)
