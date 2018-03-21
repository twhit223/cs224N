import pickle
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer


class Predictor(object):
  """
  Interface used for recurrent predictions after a one-time load of a model
  """
  model_path = 'fabian_model/model_glove_fulltrain_46accondev.h5'
  reviews_path = 'fabian_model/py3_public_company_reviews_text.pickle'
  max_len = 200

  def __init__(self):
    """
    :param str model_path: Path to model to serve prediction with
    """
    # Load model from disk
    self.model = load_model(self.model_path)

    # Compile model
    self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Unpickle reviews
    self.reviews = self.load_pkl(self.reviews_path)

    self.tokenizer = Tokenizer()
    self.tokenizer.fit_on_texts(self.reviews)

  def load_pkl(self, path):
    with open(path, 'rb') as f:
      return pickle.load(f)

  def prepro_input(self, input):
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
    encoded_input = self.tokenizer.texts_to_sequences(input)

    # Pad the encoded input to a max length
    encoded_input = pad_sequences(encoded_input, maxlen=self.max_len, padding='post')

    return encoded_input

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
    preprocessed_review = self.prepro_input(review)

    # Make prediction
    result = self.model.predict(preprocessed_review)

    prediction = result.argmax(axis=-1)

    return prediction