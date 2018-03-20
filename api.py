import os
import sys
from defs import auth_header_name, auth_header_env
from logging import StreamHandler, INFO
from flask import Flask, request
from flask_restplus import Api, Resource
from predictor import Predictor

# Create and configure the Flask app
app = Flask(__name__)

# Set up logging
app.logger.addHandler(StreamHandler(sys.stdout))
app.logger.setLevel(INFO)

# Create API object and namespace for it
api = Api(app=app, version='0.1', title='Glassdoor-NLP API')
namespace = api.namespace('api')

# Require SSL if desired
if os.environ.get('REQUIRE_SSL') == 'true':
  from flask_sslify import SSLify
  SSLify(app)

# Set up Predictor class to fetch predictions
predict_client = Predictor()


def request_authed():
  """Ensure request was authed"""
  return request.headers.get(auth_header_name) == os.environ.get(auth_header_env)


# Setup routes
@namespace.route('/predict')
class Predict(Resource):
  """POST to get a prediction from a model"""

  @namespace.doc('model_prediction')
  def post(self):
    # Validate request is authed
    if not request_authed():
      return {'error': 'Unauthorized'}, 401

    # Parse payload
    payload = api.payload or {}
    data = payload.get('data', [])

    # Ensure inputs were provided
    if not data:
      return {'error': 'Invalid payload'}, 400

    try:
      # Make predictions on each input
      predictions = predict_client.predict(data)
    except BaseException as e:
      err_msg = 'Unexpected error while making prediction: {}'.format(e)
      return {'error': err_msg}, 500

    return {'predictions': predictions}, 200


if __name__ == '__main__':
  port = int(os.environ.get('PORT', 5000))  # default to 5000
  app.run(host='0.0.0.0', port=port)