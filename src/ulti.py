#coding=<utf-8>



# from csm import CSM
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


def read_data(set_name):
    """
    Read data set from rnnpg/data/ into a list
    :param set_name: the name of the data set; could be 'qtrain', 'qtest', 'qvalid'
    :return: the data set in a list, data_set, where data_set[i] contains the characters for the ith poetry
    """
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



def read_data2(set_name):
    """
    Read data set from rnnpg/data/ into a list
    :param set_name: the name of the data set; could be 'qtrain', 'qtest', 'qvalid'
    :return: the data set in a list, data_set, where data_set[i] contains the characters for the ith poetry
    """
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


def sample(a, temperature=1.0):
    # helper function to sample an index from a probability array
    a = np.log(a) / temperature
    a = np.exp(a) / np.sum(np.exp(a))
    return np.argmax(np.random.multinomial(1, a, 1))





############################
# Word-to-Vector Functions #
############################


def w2v_train(set_name, data):
    """
    Word-to-vector embedding and save training result
    :param set_name: the name of the data set; could be 'qtrain', 'qtest', 'qvalid'
    :param data: the sentence in list
    :return:
    """
    qvalid_model = gensim.models.Word2Vec(data, min_count=1, size=50)
    qvalid_model.save('../data/{set_name}.bin'.format(set_name=set_name))


def w2v_read(set_name):
    return gensim.models.Word2Vec.load('../data/{set_name}.bin'.format(set_name=set_name))


def v2w(vector, model):
    """
    Convert vector to the closest word
    :param vector: the vector to convert
    :param model: the w2v model
    :return: the word
    """
    return model.most_similar(positive=[vector], topn=1)


########
# Main #
########

#
# def main():
#     set_name = 'qvalid'
#     qvalid_data = read_data(set_name)
#     w2v_train(set_name, qvalid_data)
#     qvalid_model = w2v_read(set_name)
#
#     # print qvalid_model.vocab
#     # print qvalid_model[qvalid_data[0][qtest7]]
#
#     csm = CSM()
#     v = csm.sen2vec(qvalid_data[0][0:6], qvalid_model, tanh)
#     print v
#     # print qvalid_data[0][qtest7]
#
# if __name__ == "__main__":
#     main()