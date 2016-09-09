from keras.layers import Activation, TimeDistributed, Dense, RepeatVector, recurrent
from keras.models import Sequential
import random
from ulti import *



# set the training dataset
set_name = 'train5'
qvalid_data = read_data(set_name)

# the length of a line
maxlen = 5
# the length of a poetry
poetry_len = maxlen * 4

# put the data into a list one by one
text = []
for i in range(0,len(qvalid_data)):
    for j in range(0, poetry_len):
        text.append(qvalid_data[i][j])

# the total word number
print 'corpus length:', len(text)
# the total vocabulary size
chars = set(text)
print 'total chars:', len(chars)

# build the dictionary
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))


# the input sentence/next sentence
sentences = []
next_sentences = []

# put the content to the sentence/next sentence
for i in range(0, len(qvalid_data)):
    for j in range(0, poetry_len - maxlen * 2 + 1, maxlen):
        sentences.append(qvalid_data[i][j:j+maxlen])
        next_sentences.append(qvalid_data[i][j + maxlen: j + maxlen * 2])
# the number of the sentence/next sentence pair
print 'nb sequences:', len(sentences)

# vectorization
print 'Vectorization...'
X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(next_sentences), maxlen, len(chars)), dtype=np.bool)

# build the one-hot model
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X[i, t, char_indices[char]] = 1

for i, sentence in enumerate(next_sentences):
    for t, char in enumerate(sentence):
        y[i, t, char_indices[char]] = 1

# build the model: simple sequence to sequence model
print 'Build model...'

# the hidden unit number
hidden_dim = 100
# the layer number
layer = 3
# the rnn element
RNN = recurrent.LSTM


model = Sequential()
# The encoder transfers the sentence to the hidden layer
model.add(RNN(hidden_dim, input_shape=(maxlen, len(chars))))
# For the decoder's input, we repeat the encoded input for each time step
model.add(RepeatVector(maxlen))
# The decoder RNN could be multiple layers stacked or a single layer
for _ in range(layer):
    model.add(RNN(hidden_dim, return_sequences=True))
# For each of step of the output sequence, decide which character should be chosen
model.add(TimeDistributed(Dense(len(chars))))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop',metrics=['accuracy'])

# print the model information
print model.summary()
out = codecs.open("../data/result_s2s", 'w', 'utf-8')
# train the model, output generated text after each iteration
for iteration in range(1, 100):
    print '-' * 50
    print 'Iteration', iteration

    # put the data into the model
    model.fit(X, y, batch_size=64, nb_epoch=50)

    # random select the first line
    start_index = random.randint(0, len(text) - maxlen - 1)
    start_index = start_index / poetry_len * poetry_len

    # add the diversity parameter in the sample function
    # for diversity in [0.2, 0.5, 1.0, 1.2]:
    diversity = 1.0
    print '----- diversity:', diversity
    # selecting the first line
    sentence = text[start_index: start_index + maxlen]
    # write down the first line to the result file
    out.write("".join(sentence))

    print '----- Generating with seed: "' + ''.join(sentence).encode('utf-8') + '"'

    # run 3 time to generate 3 lines
    for i in range(0, 3):

        # build the one hot model of the seed sentence
        x = np.zeros((1, maxlen, len(chars)))
        for t, char in enumerate(sentence):
            x[0, t, char_indices[char]] = 1.

        # predict with the next line
        preds = model.predict(x, verbose=0)[0]
        # initial for the new sentence
        new_sentence = []

        # print each character
        for j in range(0, maxlen):
            next_index = sample(preds[j], diversity)
            next_char = indices_char[next_index]
            new_sentence.append(next_char)

        sentence = new_sentence
        out.write("".join(sentence))
        print '----- Generate: "' + ''.join(new_sentence).encode('utf-8') + '"'
    out.write("\n")
out.flush()
out.close()


