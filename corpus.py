from algorithm.algorithm import *
from algorithm.prediction import Predictor
import xlwt

# file = open('./data_corpus/train.long2.txt', 'r', encoding='utf-8')
# wt_file = open('./train_corpus/train.txt', 'w', encoding='utf-8')
#
# for line in file:
#     inputs = line.strip().split(' ')
#     if len(inputs) < 2:
#         wt_file.write('# O\n')
#     else:
#         wt_file.write(line)

file = open('./train_corpus/train.txt', 'r', encoding='utf-8')
data = []
s = ''
for line in file:
    inputs = line.strip().split()

    if len(inputs) != 2:
        continue

    if inputs[0] == '#':
        data.append(s)
        s = ''
    else:
        s += inputs[0]

if len(s) > 0:
    data.append(s)

# style = xlwt.XFStyle()
# font = xlwt.Font()
# font.name = "Times New Roman"
# font.bold = False
# font.colour_index = 4
# font.height = 220
# style.font = font
#
# wd = xlwt.Workbook()
# sheet = wd.add_sheet('activity',cell_overwrite_ok=True)

wt_file = open('./data_corpus/train_label.txt', 'w', encoding='utf-8')
for item in data:
    condition = exact_condition(item)
    wt_file.write(condition+'\n')

wt_file.close()