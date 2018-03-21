from numpy import array
import numpy as np
import pickle
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional, Flatten
from keras.layers.embeddings import Embedding
from keras.utils import to_categorical
from keras import optimizers
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score

def load_pickle(filename):
  with open(filename, 'rb') as f:
    data = pickle.load(f)
  return data



#meta parameters
vocab_size = 10000 #only the 10000 most common words are used in the reviews
max_length = 200
units_LSTM = 64
dropout_rate = 0.8
lr = 0.001
decay_rate = 1e-6
epsilon = None
beta1 = 0.9
beta2 = 0.999
momentum = 0.9
number_classes = 5
number_epochs = 20


# define reviews
rev = load_pickle("py3_public_company_reviews_text.pickle")
rev = rev[:310100]
encoded_rev = [one_hot(d, vocab_size) for d in rev]
#splitting in train dev and test
encoded_rev_train = encoded_rev[:100000]
encoded_rev_dev = encoded_rev[200000:210000]
encoded_rev_test = encoded_rev[300000:310000]

# define class labels
labels = load_pickle("py3_public_company_scores.pickle")
labels = labels[:310100]
labels = [int(a) for a in labels]
labels = np.array(labels)
labels = to_categorical(labels)
labels = labels[:,1:] #not using the leading 0

#splitting in train dev and test
labels_train = labels[:100000]
labels_dev = labels[200000:210000]
labels_test = labels[300000:310000]

# pad documents to max length
encoded_rev_train = pad_sequences(encoded_rev_train, maxlen=max_length, padding='post')
encoded_rev_dev = pad_sequences(encoded_rev_dev, maxlen=max_length, padding='post')
encoded_rev_test = pad_sequences(encoded_rev_test, maxlen=max_length, padding='post')

# define the model
model = Sequential()
model.add(Embedding(vocab_size, max_length, input_length=max_length))
model.add(Bidirectional(LSTM(units_LSTM)))
model.add(Dropout(dropout_rate))
model.add(Dense(number_classes, activation='softmax'))
# compile the model

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# summarize the model
print(model.summary())

# fit the model
model.fit(encoded_rev_train, labels_train, epochs=number_epochs, verbose=1)



#save model
model.save("model_BiLSTM.h5")
print("Saved model to disk")


y_pred = model.predict(encoded_rev_dev)
y_pred = y_pred.argmax(axis=-1)
y_true = np.argmax(labels_dev, axis = 1)
cm = confusion_matrix(y_true, y_pred)
prf = precision_recall_fscore_support(y_true, y_pred, average='macro')
acc = accuracy_score(y_true, y_pred)


