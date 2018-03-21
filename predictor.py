import pickle
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer


class Predictor(object):
  """
  Interface used for recurrent predictions after a one-time load of a model
  """
  model_path = 'fabian_model/model_BiLSTM_GloVe.h5'
  reviews_path = 'fabian_model/slice_public_companies_reviews_text.pickle'
  max_len = 200

  def __init__(self):
    """
    :param str model_path: Path to model to serve prediction with
    """
    print('Loading model from disk...')
    self.model = load_model(self.model_path)

    print('Compiling model...')
    self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    print('Unpickling reviews...')
    self.reviews = self.load_pkl(self.reviews_path)

    print('Creating tokenizer...')
    self.tokenizer = Tokenizer()
    self.tokenizer.fit_on_texts(self.reviews)

  def load_pkl(self, path):
    with open(path, 'rb') as f:
      return pickle.load(f)

  def prepro_review(self, review):
    """
    Preprocess the raw text review input with the following steps:

    (1) Convert text to sequence
    (2) Pad the sequence to max length

    :param input: Company review text (a pro or a con) inside a list
      :type: list(str) --> length 1
    :return: An encoded sequence of integers for this string
    :rtype: Sequence of integers
    """
    # Convert text --> sequence
    encoded_review = self.tokenizer.texts_to_sequences(review)

    # Pad the encoded input to a max length
    padded_review = pad_sequences(encoded_review, maxlen=self.max_len, padding='post')

    return padded_review

  def predict(self, pros='', cons=''):
    """
    Get a model prediction from raw input of pros/cons

    :param str pros: Company review pros
    :param str cons: Company review cons
    :return: Predicted review score
    :rtype: int
    """
    # Just combine pros and cons
    review = [' '.join((pros, cons))]

    # Preprocess the raw input
    preprocessed_review = self.prepro_review(review)

    # Make prediction
    result = self.model.predict(preprocessed_review)

    # Format prediction into integer
    prediction = int(result.argmax(axis=-1)[0])

    return prediction