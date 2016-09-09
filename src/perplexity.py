from ulti import *
import math
import collections
import random


# the read function to read the data sets
def read():
    qvalid_data = read_data("valid5")
    test_data = read_data("test5")
    train_data = read_data("train5")
    text = []
    for i in range(0, len(qvalid_data)):
        for j in range(0, poem_length):
            text.append(qvalid_data[i][j])
    for i in range(0, len(test_data)):
        for j in range(0, poem_length):
            text.append(test_data[i][j])
    for i in range(0, len(train_data)):
        for j in range(0, poem_length):
            text.append(train_data[i][j])
    return text


# the sort function to rank the frequency of the characters
def sort(dataset):
    result = {}

    # loop in the data list
    # count the frequency for each unique character
    for word in dataset:
        if word not in result:
            result[word] = 0
        result[word] += 1
    # sort the list with the frequency
    result = collections.OrderedDict(sorted(result.items(), key=lambda t: -t[1]))
    return result

global line_length, poem_length

line_length = 5
poem_length = line_length * 4

# call the read function
dataset = read()
# call the sort function
distribution = sort(dataset)

distribution1 = {}
count = 0
key_word = []

# calculate the probability distribution
for key, value in distribution.items():
    if count < 1000:
        # print key,value
        distribution1[key] = value
        key_word.append(key)
    count += 1

x = 0.0
x1 = 0.0

# read the generated poem
generate = read_data2("../result/attention_2")
word_number = poem_length * len(generate)
# the perplexity score of the random model
for i in range(0, word_number):
    j = random.choice(distribution1.values())
    p = math.log(float(j) / len(dataset)) / word_number
    x1 += p
y1 = math.exp(-x1)
print y1

# the perplexity score of the generated model
for i in range(0, len(generate)):
    for j in range(0, poem_length):
        if generate[i][j] not in key_word:
            p = math.log(0.05) / word_number
        else:
            p = math.log(float(distribution1[generate[i][j]]) / len(dataset)) / word_number
        x += p
y = math.exp(-x)
print y
