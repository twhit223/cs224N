from numpy import array
from numpy import asarray
from numpy import zeros
import numpy as np
import pickle
from keras.preprocessing.text import one_hot
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional, Flatten
from keras.layers.embeddings import Embedding
from keras.utils import to_categorical
from keras import optimizers
from keras.models import model_from_json
from keras.models import load_model
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score

def load_pickle(filename):
  with open(filename, 'rb') as f:
    data = pickle.load(f)
  return data


#meta parameters
vocab_size = 10000 #only the 10000 most common words are used in the reviews
max_length = 200
glove_length = 100
units_LSTM = 64
dropout_rate = 0.8
lr = 0.001
decay_rate = 1e-6
epsilon = None
beta1 = 0.9
beta2 = 0.999
momentum = 0.9
number_classes = 5
number_epochs = 30


# define reviews
rev = load_pickle("py3_public_company_reviews_text.pickle")
rev = rev[:310100]


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

# prepare tokenizer
t = Tokenizer()
t.fit_on_texts(rev)
vocab_size = len(t.word_index) + 1


# integer encode the reviews
encoded_rev = t.texts_to_sequences(rev)

# pad reviews 
encoded_rev = pad_sequences(encoded_rev, maxlen=max_length, padding='post')
#splitting in train dev and test
encoded_rev_train = encoded_rev[:100000]
encoded_rev_dev = encoded_rev[200000:210000]
encoded_rev_test = encoded_rev[300000:310000]

# load the whole embedding into memory
embeddings_index = dict()
f = open('glove.6B.100d.txt', encoding="utf8")
for line in f:
	values = line.split()
	word = values[0]
	coefs = asarray(values[1:], dtype='float32')
	embeddings_index[word] = coefs
f.close()

# create a weight matrix for words in training revs
embedding_matrix = zeros((vocab_size, glove_length))
for word, i in t.word_index.items():
	embedding_vector = embeddings_index.get(word)
	if embedding_vector is not None:
		embedding_matrix[i] = embedding_vector

# define model
model = Sequential()
model.add(Embedding(vocab_size, glove_length, weights=[embedding_matrix], input_length=max_length, trainable=False))
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
model.save("model_BiLSTM_GloVe.h5")
print("Saved model to disk")


y_pred = model.predict(encoded_rev_dev)
y_pred = y_pred.argmax(axis=-1)
y_true = np.argmax(labels_dev, axis = 1)
cm = confusion_matrix(y_true, y_pred)	
prf = precision_recall_fscore_support(y_true, y_pred, average='macro')
acc = accuracy_score(y_true, y_pred)

y_pred_test = model.predict(encoded_rev_test)
y_pred_test = y_pred_test.argmax(axis=-1)
y_true_test = np.argmax(labels_test, axis = 1)
cm_test = confusion_matrix(y_true_test, y_pred_test)	
prf_test = precision_recall_fscore_support(y_true_test, y_pred_test, average='macro')
acc_test = accuracy_score(y_true_test, y_pred_test)