import os
from defs import auth_header_name, auth_header_env
from api_utils.abstract_api import AbstractApi


class ApiClient(AbstractApi):
  """
  Glassdoor-NLP API client class
  """

  def __init__(self, base_url=None):
    base_url = base_url or os.environ.get('API_URL', 'http://localhost:5000/api')

    super(ApiClient, self).__init__(base_url=base_url,
                                    auth_header_name=auth_header_name,
                                    auth_header_val=os.environ.get(auth_header_env))

  def predict(self, pros='', cons=''):
    payload = {
      'pros': pros,
      'cons': cons
    }

    try:
      resp = self.post('/predict', payload=payload)
      resp_data = resp.json
    except BaseException as e:
      print('Error while fetching prediction: {}'.format(e))
      exit(1)

    if not resp.ok:
      print('Got error response from API: {}, with code: {}.'.format(resp_data.get('error'), resp.status))
      exit(1)

    return resp_data.get('prediction')