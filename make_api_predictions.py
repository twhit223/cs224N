from api_client import api_client


if __name__ == '__main__':
  # Specify raw inputs to make predictions and create payload
  inputs = [{'key1': 'val'}]
  payload = {'data': inputs}

  # Fetch predictions from API
  try:
    resp = api_client.post('/predict', payload=payload)
    resp_data = resp.json
  except BaseException as e:
    print('Error while fetching predictions: {}'.format(e))
    exit(1)

  if not resp.ok:
    print('Got error response from API: {}, with code: {}.'.format(resp_data.get('error'), resp.status))
    exit(1)

  predictions = resp_data.get('predictions')

  # Print results
  print('Got Predictions: {}'.format(predictions))