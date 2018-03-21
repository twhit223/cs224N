from numpy import array
from numpy import asarray
from numpy import zeros
import numpy as np
import pickle
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional, Flatten
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
from keras.models import model_from_json
from keras.models import load_model

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
number_epochs = 20

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

#x_dev = padded_docs[200000:210000]
#y_dev = labels[200000:210000]
# load json and create model
#json_file = open('model_test.json', 'r')
#loaded_model_json = json_file.read()
#json_file.close()
#loaded_model = model_from_json(loaded_model_json)
# load weights into new model_glove
model = load_model("model_glove_fulltrain_46accondev.h5")
print("Loaded model from disk")
 
# evaluate loaded model on test data
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
score = model.evaluate(encoded_rev_dev, labels_dev, verbose=1)
print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))

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
