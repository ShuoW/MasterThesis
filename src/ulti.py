import gensim
import codecs
import numpy as np


########################
# Non-Linear Functions #
########################


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def tanh(x):
    return np.tanh(x)

#################
# I/O Functions #
#################


# read the original dataset
def read_data(set_name):

    f = codecs.open('../data/{set_name}'.format(set_name=set_name), 'r', encoding='utf-8')
    data_set = []
    for line in f:
        p = []
        # Each Chinese character occupies 4 spaces.
        l = len(line)
        for i in range(0, l, 2):
            p.append(line[i:i + 1])
        data_set.append(p)
    return data_set


# read the file, the two characters are continues
def read_data2(set_name):

    f = codecs.open('../data/{set_name}'.format(set_name=set_name), 'r', encoding='utf-8')
    data_set = []
    for line in f:
        p = []
        # Each Chinese character occupies 4 spaces.
        l = len(line)
        for i in range(0, l, 1):
            p.append(line[i:i + 1])
        data_set.append(p)
    return data_set

############################
# sample Functions #
############################

# sample function to predict the character
def sample(a, temperature = 1.0):
    # helper function to sample an index from a probability array
    a = np.log(a) / temperature
    a = np.exp(a) / np.sum(np.exp(a))
    return np.argmax(np.random.multinomial(1, a, 1))


############################
# Word-to-Vector Functions #
############################

# train the w2v model
def w2v_train(set_name, data):

    qvalid_model = gensim.models.Word2Vec(data, min_count=1, size=50)
    qvalid_model.save('../data/{set_name}.bin'.format(set_name=set_name))

# convert the word to the vector
def w2v_read(set_name):
    return gensim.models.Word2Vec.load('../data/{set_name}.bin'.format(set_name=set_name))


# convert the vector to the word
def v2w(vector, model):
    return model.most_similar(positive=[vector], topn=1)