import numpy
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
# fix random seed
numpy.random.seed(8)
# load the dataset with 5000 top words
top_words = 5000
max_words = 500
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words, maxlen = max_words)
# pad dataset to a maximum review length in words
X_train = sequence.pad_sequences(X_train, maxlen=max_words)
X_test = sequence.pad_sequences(X_test, maxlen=max_words)
# create the model
def create_model():
    model = Sequential()
    #use 32 dimension word-embedding using word2vec
    model.add(Embedding(top_words, 32, input_length=max_words))
    # add a layer od Conv and max pooling
    model.add(Conv1D(32, 3, border_mode= 'same' , activation= 'relu'))
    # reduce the space to half the size
    model.add(MaxPooling1D(pool_length=2))
    # Flatten the data to be fed into Feed-forward NN
    model.add(Flatten())
    # Input to Feed-forward RNN
    model.add(Dense(250, activation= 'relu' ))
    model.add(Dense(1, activation= 'sigmoid' ))
    model.compile(loss= 'binary_crossentropy' , optimizer= 'adam' , metrics=[ 'accuracy' ])
    print(model.summary())
    # Fitting the model
    model.fit(X_train, y_train, validation_data=(X_test, y_test), nb_epoch=2, batch_size=128, verbose=1)
    return model
#evaluation of the model
cnn_model = create_model()
scores = cnn_model.evaluate(X_test, y_test, verbose=0)
# print final accuracy and save the weights to be retrieved later
print("Accuracy: %.2f%%" % (scores[1]*100))
cnn_model.save_weights("Weights.best.hd5")
