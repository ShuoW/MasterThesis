from keras.layers.core import Dense, Dropout, Activation
from keras.layers.wrappers import TimeDistributed
from keras.models import Sequential
import random
from ulti import *
from encoders import *
from decoders import *
from bidirectional import *

#################
# read function#
#################


def read_function(set_name, voc_name):

    # read the data set
    input_data1 = read_data(set_name)
    # read the vocabulary
    voc = read_data2(voc_name)

    # put the data into a list one by one
    text1 = []
    text2 = []
    for i in range(0, len(input_data1)):
        for j in range(0, poetry_len):
            text1.append(input_data1[i][j])

    for i in range(0,len(voc[0])):
        text2.append(voc[0][i])

    # the vocabulary size
    chars1 = set(text2)
    # print len(chars1)
    return input_data1, text1, chars1

#################
# dict function#
#################


def build_dict():
    # build the dictionary
    char_indices = dict((c, i) for i, c in enumerate(chars))
    indices_char = dict((i, c) for i, c in enumerate(chars))
    return char_indices, indices_char


#################
# model1 function#
#################


# copy the corresponding model from the training file
def model1():
    print 'Build model...'

    hidden_dim = 200
    depth1 = 2
    depth2 = 2
    drop = 0.1
    output_dim = len(chars)
    output_length = maxlen
    input_dim = len(chars)
    input_length = maxlen

    # build the DL model
    model = Sequential()
    # input layer
    model.add(Layer(batch_input_shape=(None, input_length, input_dim)))
    # encoder layer
    model.add(Bidirectional(LSTMEncoder(output_dim=int(hidden_dim / 2),state_input=False,
                                        return_sequences=True,input_length=maxlen,input_dim=len(chars))))
    for i in range(0, depth1 - 1):
        model.add(Dropout(drop))
        model.add(Bidirectional(LSTMEncoder(output_dim=int(hidden_dim / 2),state_input=False,
                                            return_sequences=True,input_length=maxlen,input_dim=len(chars))))
    model.add(Dropout(drop))
    model.add(TimeDistributed(Dense(hidden_dim if depth2 > 1 else output_dim)))
    # decoder layer
    decoder = AttentionDecoder(hidden_dim=hidden_dim, output_length=output_length,
                                   state_input=False, input_length=maxlen,input_dim=len(chars))
    model.add(Dropout(drop))
    model.add(decoder)
    # decoder layer
    for i in range(0, depth2 - 1):
        model.add(Dropout(drop))
        model.add(LSTMEncoder(output_dim=hidden_dim,state_input=False,
                              return_sequences=True,input_length=maxlen,input_dim=len(chars)))
    model.add(Dropout(drop))
    model.add(TimeDistributed(Dense(output_dim)))
    # active function
    model.add(Activation('softmax'))
    return model

#################
# writing function#
#################


def seed_function():

    # random select the first line
    start_index = random.randint(0, len(text) - maxlen - 1)
    start_index = start_index / poetry_len * poetry_len
    # selecting the first line
    sentence = text[start_index: start_index + maxlen]
    out.write("".join(sentence))
    print '----- Generating with seed: "' + ''.join(sentence).encode('utf-8') + '"'
    flag = []
    for s in sentence:
        # print judge(s)
        flag.append(judge(s))

    # check the tone pattern
    if flag[1] == 2 and flag[2] == 1 and flag[3] == 1 and flag[4] == 2:
        yunjiao = 1
    elif flag[1] == 2 and flag[2] == 2 and flag[3] == 1 and flag[4] == 1:
        yunjiao = 2
    elif flag[1] == 1 and flag[2] == 1 and flag[3] == 2 and flag[4] == 2:
        yunjiao = 3
    elif flag[0] == 1 and flag[1] == 1 and flag[3] == 2 and flag[4] == 1:
        yunjiao = 4
    else:
        yunjiao = 0
    # print yunjiao
    return sentence, yunjiao


# the fixed seed sentence
def seed_function1():

    first_sentence = read_data2("try")
    # put the data into a list obe by one
    text1 = []
    for j in range(0, maxlen):
        text1.append(first_sentence[0][j])


    out.write("".join(text1))
    print '----- Generating with seed: "' + ''.join(text1).encode('utf-8') + '"'
    flag = []
    for s in text1:
        # print judge(s)
        flag.append(judge(s))

    if flag[1] == 2 and flag[2] == 1 and flag[3] == 1 and flag[4] == 2:
        yunjiao = 1
    elif flag[1] == 2 and flag[2] == 2 and flag[3] == 1 and flag[4] == 1:
        yunjiao = 2
    elif flag[1] == 1 and flag[2] == 1 and flag[3] == 2 and flag[4] == 2:
        yunjiao = 3
    elif flag[0] == 1 and flag[1] == 1 and flag[3] == 2 and flag[4] == 1:
        yunjiao = 4
    else:
        yunjiao = 0
    print yunjiao
    return text1, yunjiao


##################
# check function #
##################


def judge(w):

    # read the ping vocabulary
    ping_data = []
    ping = read_data2("ping.txt")
    for i in range(0,len(ping[0])):
        ping_data.append(ping[0][i])

    # read the ze vocabulary
    ze_data = []
    ze = read_data2("ze.txt")
    for i in range(0,len(ze[0])):
        ze_data.append(ze[0][i])

    # provide the tone
    if w in ping_data:
        return 1
    elif w in ze_data:
        return 2
    else:
        return 0


#####################
# prediction function#
#####################

def pre(char_indices,indices_char,diversity,model,sentence,yun):
    # build the one hot model of the seed sentence
    x = np.zeros((1, maxlen, len(chars)))
    for t, char in enumerate(sentence):
        if char not in chars:
            char = "R"
        x[0, t, char_indices[char]] = 1.
    # predict with the next line
    preds = model.predict(x, verbose=0)[0]
    # initial for the new sentence
    new_sentence = []
    count_r = 0
    for j in range(0, maxlen):
        next_index = sample(preds[j], diversity)
        next_char = indices_char[next_index]
        new_sentence.append(next_char)
        if next_char == "R":
            count_r += 1
    if count_r == 0:
        sentence = new_sentence
        out.write("".join(sentence))
        print '----- Generate: "' + ''.join(new_sentence).encode('utf-8') + '"'
        return sentence
    else:
        return sentence

#################
# main function#
#################


def main():

    # set the data name and vocabulary name
    set_name = 'valid5'
    voc_name = 'vocabulary1'

    global input_data, text, chars, maxlen, poetry_len
    maxlen = 5
    poetry_len = maxlen * 4

    input_data, text, chars = read_function(set_name,voc_name)
    char_indices, indices_char = build_dict()

    # call the model
    model = model1()
    # load the weights
    model.load_weights("../weight/keras_wr_60")

    # write down the results
    global out
    # out = codecs.open("../result/second", 'w', 'utf-8')
    out = codecs.open("../result/attention_2", 'w', 'utf-8')
    # train the model, output generated text after each iteration
    for iteration in range(1, 1001):
        print '-' * 50
        print 'Iteration', iteration

        # put the data into the model
        diversity = 1.0
        print '----- diversity:', diversity
        sentence,yun = seed_function()


        c = 0
        # if the file is used in the Attention_3/Attention_4
        # change the loop to c < 1 to predict the second line only
        # while c < 1:
        while c < 3:
            s1 = sentence
            sentence = pre(char_indices, indices_char,diversity,model,sentence,yun)
            if sentence != s1:
                c += 1
        out.write("\n")
    out.flush()
    out.close()

if __name__ == "__main__":
    main()
