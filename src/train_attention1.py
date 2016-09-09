from keras.layers.core import Dense, Dropout, Activation
from keras.layers.wrappers import TimeDistributed
from keras.models import Sequential
import random
from ulti import *
from encoders import *
from decoders import *
from bidirectional import *


# This model is the attention model (Attention-1),
# trained with the 5-char dataset.


#################
# read function #
#################

# read the training data, and put it to the list
def read_function():

    # the set name
    input_data1 = read_data("test5")
    input_data2 = read_data("valid5")
    input_data3 = read_data("train1w")

    # put the data into a list one by one
    text1 = []
    text2 = []
    for i in range(0, len(input_data1)):
        for j in range(0, [poetry_len]):
            text1.append(input_data1[i][j])
            text2.append(input_data1[i][j])
    for i in range(0,len(input_data2)):
        for j in range(0, poetry_len):
            text1.append(input_data2[i][j])
    for i in range(0,len(input_data3)):
        for j in range(0, poetry_len):
            text1.append(input_data3[i][j])

    # the total character number
    print 'corpus length:', len(text1)

    # the total vocabulary size
    chars1 = set(text1)
    print 'total chars:', len(chars1)

    return input_data3, text2, chars1

#################
# dict function #
#################


# the dictionary function to build the one-hot dictionary
def build_dict():
    # build the dictionary
    char_indices = dict((c, i) for i, c in enumerate(chars))
    indices_char = dict((i, c) for i, c in enumerate(chars))
    return char_indices, indices_char

######################
# Sequence function #
######################


# the build_seq function build the sentence/next_sentence pair
def build_seq():

    # the input sentence/next sentence pair
    sentences = []
    next_sentences = []

    # put the content to the sentence/next_sentence pair
    for i in range(0, len(input_data)):
        for j in range(0, poetry_len - maxlen * 2 + 1, maxlen):
            sentences.append(input_data[i][j:j+maxlen])
            next_sentences.append(input_data[i][j + maxlen: j + maxlen * 2])
    # the number of the sentence/next sentence pair
    print 'nb sequences:', len(sentences)
    return sentences, next_sentences

###################
# vector function #
###################


# convert the character to the one-hot vector
# and then build the sentence matrix
def vector_function(char_indices):

    # read the sentences
    sentences, next_sentences = build_seq()
    # vectorization
    print 'Vectorization...'
    x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
    y = np.zeros((len(next_sentences), maxlen, len(chars)), dtype=np.bool)

    # build the one-hot matrix
    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            x[i, t, char_indices[char]] = 1

    for i, sentence in enumerate(next_sentences):
        for t, char in enumerate(sentence):
            y[i, t, char_indices[char]] = 1
    return x, y


###################
# model1 function #
###################


# the proposed model
def model1():

    # build the model
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
    model.add(Bidirectional(LSTMEncoder(output_dim=int(hidden_dim / 2), state_input=False,
                                        return_sequences=True, input_length=maxlen, input_dim=len(chars))))
    # the depth
    for i in range(0, depth1 - 1):
        model.add(Dropout(drop))
        model.add(Bidirectional(LSTMEncoder(output_dim=int(hidden_dim / 2), state_input=False,
                                            return_sequences=True, input_length=maxlen, input_dim=len(chars))))
    # dropout layer
    model.add(Dropout(drop))
    model.add(TimeDistributed(Dense(hidden_dim if depth2 > 1 else output_dim)))
    # decoder layer
    decoder = AttentionDecoder(hidden_dim=hidden_dim, output_length=output_length,
                               state_input=False, input_length=maxlen, input_dim=len(chars))
    # dropout layer
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

####################
# writing function #
####################


# select the seed sentence
def seed_function():

    # random select the first line
    start_index = random.randint(0, len(text) - maxlen - 1)
    start_index = start_index / poetry_len * poetry_len
    # selecting the first line
    sentence = text[start_index: start_index + maxlen]
    # write down the first line to the result file
    # out.write("".join(sentence))
    return sentence


#######################
# prediction function #
#######################


# predict the next line
def pre(char_indices, indices_char, diversity, model, sentence):

    # build the one hot model of the seed sentence
    x = np.zeros((1, maxlen, len(chars)))
    for t, char in enumerate(sentence):
        x[0, t, char_indices[char]] = 1.
    # predict the next line
    preds = model.predict(x, verbose=0)[0]
    # initial for the new sentence
    new_sentence = []

    for j in range(0, maxlen):
        next_index = sample(preds[j], diversity)
        next_char = indices_char[next_index]
        new_sentence.append(next_char)

    sentence = new_sentence
    # out.write("".join(sentence))
    print '----- Generate: "' + ''.join(new_sentence).encode('utf-8') + '"'
    return sentence

#################
# main function #
#################


def main():

    global input_data, text, chars, maxlen, poetry_len
    # the length of the line
    maxlen = 5
    # the length of the poem
    poetry_len = maxlen * 4

    # obtain the data
    input_data, text, chars = read_function()
    # obtain the dictionary
    char_indices, indices_char = build_dict()

    # obtain the input sentence and the target sentence
    x, y = vector_function(char_indices)
    # call the model
    model = model1()

    # compile layer
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop',metrics=['accuracy'])

    # print the model information
    print model.summary()
    # global out
    # out = codecs.open("../result/attention_1", 'w', 'utf-8')

    # train the model, output generated text after each iteration
    count = 1
    for iteration in range(1, 100):
        count += 1
        print '-' * 50
        print 'Iteration', iteration

        # fit the training data into the model
        model.fit(x, y, batch_size=128, nb_epoch=50)
        diversity = 1.0
        print '----- diversity:', diversity

        # call the seed function
        sentence = seed_function()
        print '----- Generating with seed: "' + ''.join(sentence).encode('utf-8') + '"'

        for i in range(0, 3):
            sentence = pre(char_indices, indices_char,diversity,model,sentence)
        # out.write("\n")
        # save the parameters
        if count == 10:
            model.save_weights("../weight/keras_w_10")
        elif count == 20:
            model.save_weights("../weight/keras_w_20")
        elif count == 30:
            model.save_weights("../weight/keras_w_30")
        elif count == 40:
            model.save_weights("../weight/keras_w_40")
        elif count == 50:
            model.save_weights("../weight/keras_w_50")
        elif count == 60:
            model.save_weights("../weight/keras_w_60")
        elif count == 70:
            model.save_weights("../weight/keras_w_70")
        elif count == 80:
            model.save_weights("../weight/keras_w_80")
        elif count == 90:
            model.save_weights("../weight/keras_w_90")
        elif count == 100:
            model.save_weights("../weight/keras_w_100")
    # out.flush()
    # out.close()
if __name__ == "__main__":
    main()
