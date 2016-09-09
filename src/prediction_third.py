from keras.layers.core import Dense, Dropout, Activation, Merge
from keras.layers.wrappers import TimeDistributed
from keras.models import Sequential
from ulti import *
from encoders import *
from decoders import *
from bidirectional import *

#################
# read function#
#################


def read_function(set_name, voc_name):

    # the dataset name
    input_data1 = read_data2("../result/second")
    voc = read_data2("vocabulary1")
    # put the data into a list obe by one
    text1 = []
    text2 = []

    for i in range(0, len(input_data1)):
        for j in range(0, maxlen*2):
            text1.append(input_data1[i][j])
    # the total word number
    for i in range(0,len(voc[0])):
        text2.append(voc[0][i])
    return text1, text2

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

    left = Sequential()
    left.add(Layer(batch_input_shape=(None, input_length, input_dim)))
    left.add(Bidirectional(LSTMEncoder(output_dim=int(hidden_dim / 2),state_input=False,
                                       return_sequences=True,input_length=maxlen,input_dim=len(chars))))
    for i in range(0, depth1 - 1):
        left.add(Dropout(drop))
        left.add(Bidirectional(LSTMEncoder(output_dim=int(hidden_dim / 2), state_input=False,
                                            return_sequences=True, input_length=maxlen, input_dim=len(chars))))

    right = Sequential()
    right.add(Layer(batch_input_shape=(None, input_length, input_dim)))
    right.add(Bidirectional(LSTMEncoder(output_dim=int(hidden_dim / 2),state_input=False,
                                        return_sequences=True,input_length=maxlen,input_dim=len(chars))))
    for i in range(0, depth1 - 1):
        right.add(Dropout(drop))
        right.add(Bidirectional(LSTMEncoder(output_dim=int(hidden_dim / 2), state_input=False,
                                            return_sequences=True, input_length=maxlen, input_dim=len(chars))))

    merged = Merge([left, right], mode='concat')
    # build the DL model
    model = Sequential()
    # input layer
    model.add(merged)
    model.add(Dropout(drop))

    model.add(TimeDistributed(Dense(hidden_dim if depth2 > 1 else output_dim)))
    # decoder layer
    decoder = AttentionDecoder(hidden_dim=hidden_dim, output_length=output_length,
                               state_input=False, input_length=maxlen, input_dim=len(chars))
    model.add(Dropout(drop))
    model.add(decoder)
    # decoder layer
    for i in range(0, depth2 - 1):
        model.add(Dropout(drop))
        model.add(LSTMEncoder(output_dim=hidden_dim, state_input=False,
                              return_sequences=True, input_length=maxlen, input_dim=len(chars)))
    model.add(Dropout(drop))
    model.add(TimeDistributed(Dense(output_dim)))
    # active function
    model.add(Activation('softmax'))
    return model

#################
# writing function#
#################


def seed_function(iteration):

    first = text[(iteration-1)*maxlen*2:(iteration-1)*maxlen*2+maxlen]
    second = text[(iteration-1)*maxlen*2+maxlen:iteration*maxlen*2]
    # print (iteration-1)*10,(iteration-1)*10+5,(iteration-1)*10+5,iteration*10
    return first, second


##################
# check function #
##################


def judge(w):

    ping_data = []
    ping = read_data2("ping.txt")
    for i in range(0,len(ping[0])):
        ping_data.append(ping[0][i])
    ze_data = []
    ze = read_data2("ze.txt")
    for i in range(0,len(ze[0])):
        ze_data.append(ze[0][i])

    if w in ping_data:
        return 1
    elif w in ze_data:
        return 2
    else:
        return 0


#################
# prediction function#
#################

def pre(char_indices,indices_char,diversity,model,sentence,sentence1):
    # build the one hot model of the seed sentence
    x = np.zeros((1, maxlen, len(chars)))
    x1 = np.zeros((1, maxlen, len(chars)))

    for t, char in enumerate(sentence):
        if char not in chars:
            char = "R"
        x[0, t, char_indices[char]] = 1.
    for t, char in enumerate(sentence1):
        if char not in chars:
            char = "R"
        x1[0, t, char_indices[char]] = 1.

    # predict with the next line
    preds = model.predict([x,x1], verbose=0)[0]
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
        sentence = sentence1
        sentence1 = new_sentence
        out.write("".join(new_sentence))
        print '----- Generate: "' + ''.join(new_sentence).encode('utf-8') + '"'
        return sentence, sentence1
    else:
        return sentence,sentence1

#################
# main function#
#################


def main():

    # set the data name and vocabulary name
    set_name = "../result/second"
    voc_name = "vocabulary1"

    global text, chars, maxlen, poetry_len
    maxlen = 5
    poetry_len = maxlen * 4

    text, chars = read_function(set_name,voc_name)
    char_indices, indices_char = build_dict()

    # call the model
    model = model1()
    # load the weights
    model.load_weights("../weight/keras_sr_20")

    # write down the results
    global out
    out = codecs.open("../result/attention_3", 'w', 'utf-8')
    # train the model, output generated text after each iteration
    for iteration in range(1, 1001):
        print '-' * 50
        print 'Iteration', iteration

        # put the data into the model
        diversity = 1.0
        print '----- diversity:', diversity

        sentence, sentence1 = seed_function(iteration)
        out.write("".join(sentence))
        out.write("".join(sentence1))

        print '----- Generating with seed: "' + ''.join(sentence).encode('utf-8') + '"'
        print '----- Generating with seed: "' + ''.join(sentence1).encode('utf-8') + '"'

        c = 0
        while c < 2:
            s1 = sentence
            s2 = sentence1
            sentence, sentence1 = pre(char_indices, indices_char,diversity,model,sentence,sentence1)
            if sentence != s1 and sentence1 != s2:
                c += 1
        out.write("\n")
    out.flush()
    out.close()

if __name__ == "__main__":
    main()
