from nltk.translate.bleu_score import *
from ulti import *

# read a generated line
hy_data = read_data2("generated")
hypothesis = []
line_length = 5
for i in range(0, line_length):
    hypothesis.append(hy_data[0][i])

# read 20 referenced lines
ref_data = read_data("ref")
ref = []
poem_length = 4 * line_length
for i in range(0, poem_length):
    ref.append([ref_data[i][0]])
    for j in range(0, line_length):
        ref[i].append(ref_data[i][j])

# set the biagram weight
weight = (0.5, 0.5)
a = sentence_bleu(ref, hypothesis, weight)
print a
