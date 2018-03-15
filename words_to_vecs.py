import numpy as np
import nltk
import random
import pickle
import code
import string
import os
import re
import copy
import gensim

from keras.models import load_model

from collections import Counter

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

from sklearn.preprocessing import OneHotEncoder

# Set variables
num_features = 10000
len_revs = 100

### Import the reviews as a list
def import_reviews(filename):
  
  revs_raw = open("data/" + filename, "r").read().split('\n')
  print("Preprocessing reviews...")
  processed_revs = []
  scores = np.array((len(revs_raw), ))
  all_words = []
  print(len(revs_raw))
  count = 0
  max_length = 0 

  for rev in revs_raw:
    
    # Get the categories
    if rev == '':
      continue
    rev = rev.split('||')
    pro = rev[0]
    con = rev[1]
    # adv = rev[2]
    score = rev[3]
    # pre_post = rev[4]
    # rev_id = rev[5]
    # company_id = rev[6]
    # industry = rev[7]
    # comp_good_bad = rev[8]

    # Tokenize the words and cut each off at 100 words 
    pro_words = word_tokenize(pro.lower())[0:len_revs]
    con_words = word_tokenize(con.lower())[0:len_revs]
    rev_words = pro_words + con_words
    all_words.extend(rev_words)
    if len(rev_words) > max_length: max_length = len(rev_words)

    processed_revs.append(rev_words)
    np.append(scores, score)

  return all_words, processed_revs, scores



def flatten(seq,container=None):
    if container is None:
        container = []
    for s in seq:
        if hasattr(s,'__iter__'):
            flatten(s,container)
        else:
            container.append(s)
    return container

### Convert the reviews to one-hot bag of words vectors
# Takes an input that is a list of lists. Each list in the list contains a single review, with one element for each word.
def one_hot_encode(all_words, revs, num_words):

  # Get the 10k most common words as a dict with the form word:position
  print("Creating bag of words dictionary...")
  count = [['PAD', -1], ['UNK', -1]]
  count.extend(Counter(all_words).most_common(num_words - 2))
  dictionary = dict()
  for word, _ in count:
    dictionary[word] = len(dictionary)
  
  # Create a 200-length integer vector for each review
  print("Integer encoding reviews...")
  revs_int_encoded = [[dictionary[word] if word in dictionary else dictionary['UNK'] for word in rev] + [dictionary['PAD']]*(200-len(rev)) for rev in revs]
  
  # Create the one-hot vectors from the integer vectors
  print("One-hot encoding reviews...")
  enc = OneHotEncoder(n_values = 10000)
  enc.fit(revs_int_encoded)
  revs_one_hot_encoded = enc.transform(revs_int_encoded).toarray()
  print("All encoding complete...")

  # Get the counts for 'PAD' and 'UNK' and create the lookup table
  flat = flatten(revs_int_encoded)
  count[0][1] = flat.count(0)
  count[1][1] = flat.count(1)
  reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))

  # Convert revs_int_encoded to numpy array
  revs_int_encoded = np.array(revs_int_encoded)

  return revs_one_hot_encoded, revs_int_encoded, count, dictionary, reversed_dictionary


# ### Convert the reviews to word2vec bag of words vectors
# def word2vec_encode(revs):

#   model = gensim.models.Word2Vec.load_word2vec_format('data/GoogleNews-vectors-negative300.bin', binary=True)
#   zero = np.zeros(300)
#   revs_word2vec_encoded = [[model[word] if word in model.wv.vocab else zero for word in rev] for rev in revs]

#   return revs_word2vec_encoded


# ### Convert the reviews to GloVe vectors
# def load_glove_model(gloveFile):
#     print ("Loading Glove Model")
#     f = open(gloveFile,'r')
#     model = {}
#     for line in f:
#         splitLine = line.split()
#         word = splitLine[0]
#         embedding = np.array([float(val) for val in splitLine[1:]])
#         model[word] = embedding
#     print ("Done.",len(model)," words loaded!")
#     return model

# def glove_encode(revs):

#   model = load_glove_model('data/glove.42B.300d.txt')
#   zero = np.zeros(100)
#   revs_glove_encoded = [[model[word] if word in model else zero for word in rev] for rev in revs]

#   return revs_glove_encoded

# ### Convert the words to CoVe vectors
# def cove_encode(revs_glove_encoded):

#   cove_model = load_model('Keras_CoVe.h5')
#   # Pass each review into the glove model
#   revs_cove_encoded = np.array()
#   code.interact(local = locals())
#   for rev in revs_glove_encoded:
#     rev_reshaped = np.reshape(rev, (1, len(rev), 300))
#     revs_cove_encoded.append(cove_model.predict(rev))

#   return revs_cove_encoded


filename = 'reviews_control_private.txt'
all_words, revs, scores = import_reviews(filename)
revs_one_hot_encoded, revs_int_encoded, count, dictionary, reversed_dict = one_hot_encode(all_words, revs, num_features)
# # revs_word2vec_encoded = word2vec_encode(revs)
# revs_glove_encoded = glove_encode(revs)
# revs_cove_encoded = cove_encode(revs_glove_encoded)

code.interact(local = locals())