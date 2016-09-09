from ulti import *
import collections

# read the data
qvalid_data = read_data("valid5")
test_data = read_data("test5")
train_data = read_data("train5")

maxlen = 5
poetry_line = maxlen * 4

# put the character to the list
text = []
for i in range(0,len(qvalid_data)):
    for j in range(0, poetry_line):
        text.append(qvalid_data[i][j])
for i in range(0,len(test_data)):
    for j in range(0, poetry_line):
        text.append(test_data[i][j])
for i in range(0,len(train_data)):
    for j in range(0, poetry_line):
        text.append(train_data[i][j])

# write out the results
result = {}
out = codecs.open("../data/vocabulary5", 'w', 'utf-8')

# count the frequency
for word in text:
    if word not in result:
        result[word] = 0
    result[word] += 1
# sort
result = collections.OrderedDict(sorted(result.items(),key=lambda t: -t[1]))
count = 0
for key, value in result.items():

    # print top-10
    if count < 10:
        print key, value
    # write down top-2000
    if count < 2000:
        out.write(key)
        count += 1
