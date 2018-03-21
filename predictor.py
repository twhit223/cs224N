from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer


class Predictor(object):
  """
  Interface used for recurrent predictions after a one-time load of a model
  """

  def __init__(self, model_path='', max_len=200):
    """
    :param str model_path: Path to model to serve prediction with
    """
    self.model_path = model_path
    self.max_len = max_len

    # Load model from disk
    self.model = load_model(self.model_path)

    # Create a tokenizer
    self.tokenizer = Tokenizer()

  def prepro_input(self, input):
    """
    Preprocess the raw text review input with the following steps:

    (1) Convert text to sequence
    (2) Pad the sequence to max length

    :param str input: Company review text (a pro or a con)
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
    :rtype: float
    """
    # Just combine pros and cons
    review = pros + cons

    # Preprocess the raw input
    # TODO: input = self.prepro_input(review)

    # Get a prediction from the model
    # TODO: score = self.model.predict([input])
    score = 5.0

    return score