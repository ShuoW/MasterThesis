from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
import random
from ulti import *


# the print function
def print_g(g):
    full = ''.join(g)
    print full[0:maxlen]
    print full[maxlen:maxlen*2]
    print full[maxlen*2:maxlen*3]
    print full[maxlen*3:maxlen*4]
global maxlen 
maxlen = 5
poetry_line = maxlen * 4
# set the training dataset
set_name = 'train5'
qvalid_data = read_data(set_name)

text = []
for i in range(0,len(qvalid_data)):
    for j in range(0, poetry_line):
        text.append(qvalid_data[i][j])
print('corpus length:', len(text))

# build the dictionary
chars = set(text)
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# build the sentence

step = 1
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])
print('nb sequences:', len(sentences))

# build the vector
print('Vectorization...')
X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1

# build the model: two LSTM layers
print('Build model...')
model = Sequential()
model.add(LSTM(32, return_sequences=True, input_shape=(maxlen, len(chars))))
model.add(Dropout(0.2))
model.add(LSTM(32, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(len(chars)))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')


# train the model, output generated text after each iteration with the different diversity
for iteration in range(1, 100):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    model.fit(X, y, batch_size=128, nb_epoch=50)

    # select the seed sentence
    start_index = random.randint(0, len(text) - maxlen - 1)
    start_index = start_index / poetry_line * poetry_line

    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print()
        print('----- diversity:', diversity)

        generated = []
        sentence = text[start_index: start_index + maxlen]
        generated += sentence
        print('----- Generating with seed: "' + ''.join(sentence) + '"')

        # predict the characters
        for i in range(maxlen*3):
            x = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(sentence):
                x[0, t, char_indices[char]] = 1.

            preds = model.predict(x, verbose=0)[0]
            # predict the next character
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]
            generated += next_char
            # compose the sentence
            sentence = sentence[1:] + [next_char]
        print_g(generated)


