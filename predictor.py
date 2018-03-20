import tensorflow as tf
from data_util import load_embeddings
from defs import LBLS
from Model_Glassdoor import GlassdoorModel, Config, ModelHelper
from util import read_conll


class Predictor(object):
  """
  Interface used for recurrent predictions after a one-time load of a model
  """
  def __init__(self, model_path=''):
    self.model_path = model_path

    # Create model helper from model_path
    self.helper = ModelHelper.load(model_path)

    # Configure word embeddings
    self.embedding_configs = EmbeddingConfigs()
    self.embeddings = load_embeddings(self.embedding_configs, self.helper)

    # Establish config info
    self.config = Config(model_path)
    self.config.embed_size = self.embeddings.shape[1]

    # Create our model
    self.model = GlassdoorModel(self.helper, self.config, self.embeddings)

    # Create a new session and initialize globals
    self.sess = tf.Session()
    self.sess.run(tf.global_variables_initializer())

    # Create our saver and restore the model
    self.saver = tf.train.Saver()
    self.saver.restore(self.sess, self.model.config.model_output)

  def predict(self, data):
    """
    Get a model prediction from raw input

    :param list data: List of raw inputs to make predictions for
    :return: Predictions and components for each
    :rtype: list(dict)
    """
    # TODO: Prolly don't wanna do this every time...
    data = read_conll(data)

    results = []

    for sentence, labels, predictions in self.model.output(self.sess, data):
      predictions = [LBLS[l] for l in predictions]

      results.append({
        'sentence': sentence,
        'labels': labels,
        'predictions': predictions
      })

    return results


class EmbeddingConfigs(object):
  vocab = ''
  vectors = ''