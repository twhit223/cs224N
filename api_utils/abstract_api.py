import requests


class AbstractApi(object):
  """
  Provides a RESTful interface to the API url of choice.
  Wrapper around the 'requests' library.

  Allows for specification of:
    - Base url of API
    - Route to hit
    - Method to use (GET, POST, PUT, and DELETE supported)
    - Payload (auto-converted into query params for GET & DELETE requests)
    - Base headers (used on all requests)
    - Request-specific headers (can overwrite base headers)
    - Auth headers (can either be static or read a token from a provided function)

  Basic usage:
    api_client = AbstractApi(base_url='https://myurl.com/api',
                             base_headers={'My-Header': 'Always include this header'},
                             auth_header_name='My-Special-Token',
                             auth_header_val='super-secret-token')

    resp = api_client.get('/my-route', payload={'key': 'val'})

    resp.status         # 200
    resp.json           # => {'key': 'parsed json response'}
    resp.headers        # {'My-Response-Header': 'some value'}
    resp.response_obj   # 'requests.models.Response' class instance
  """

  def __init__(self, base_url=None, base_headers=None, auth_header_name=None,
               auth_header_val=None, auth_header_val_getter=None):
    """
    :param str base_url: Base url of the API you wish to hit
    :param dict base_headers: Headers to send with every request
    :param str auth_header_name: Name of authorization header
    :param str auth_header_val: Value of authorization header (overwritten by auth_header_val_getter)
    :param function auth_header_val_getter: Function that returns the value of the authorization header
    """
    self.base_url = base_url
    self.base_headers = base_headers or {}
    self.auth_header_name = auth_header_name
    self.auth_header_val = auth_header_val
    self.auth_header_val_getter = auth_header_val_getter

  def get(self, route, **kwargs):
    """
    Make a GET request to the given route

    :param str route: Route to hit on top of 'self.base_url'
    ** See 'self.make_request' for accepted kwargs params **

    :return: an API response object
    :rtype: AbstractApiResponse
    """
    return self.make_request('get', route, **kwargs)

  def post(self, route, **kwargs):
    """
    Make a POST request to the given route

    :param str route: Route to hit on top of 'self.base_url'
    ** See 'self.make_request' for accepted kwargs params **

    :return: an API response object
    :rtype: AbstractApiResponse
    """
    return self.make_request('post', route, **kwargs)

  def put(self, route, **kwargs):
    """
    Make a PUT request to the given route

    :param str route: Route to hit on top of 'self.base_url'
    ** See 'self.make_request' for accepted kwargs params **

    :return: an API response object
    :rtype: AbstractApiResponse
    """
    return self.make_request('put', route, **kwargs)

  def delete(self, route, **kwargs):
    """
    Make a DELETE request to the given route

    :param str route: Route to hit on top of 'self.base_url'
    ** See 'self.make_request' for accepted kwargs params **

    :return: an API response object
    :rtype: AbstractApiResponse
    """
    return self.make_request('delete', route, **kwargs)

  def make_request(self, method, route, payload=None, headers=None):
    """
    Actually perform the API call.

    :param str method:
      API method to perform. Valid options: 'get', 'post', 'put', 'delete'.
    :param str route:
      API route to hit on top of self.base_url
    :param dict payload:
      Payload to provide with request. For GET and DELETE requests, these are converted into query params.
    :param dict headers:
      Request-specific headers. Will overwrite any self.base_headers with the same key.
    :return: an API response object
    :rtype: AbstractApiResponse
    """
    # Get the proper method to call on the 'requests' object (get, post, put, or delete).
    request = getattr(requests, method)

    # Build up kwargs to pass to the requests method call.
    request_kwargs = {
      'headers': self.build_request_headers(headers=headers)
    }

    if method in ('get', 'delete'):
      # Set the payload as query params for GET and DELETE requests
      request_kwargs['params'] = payload or {}
    else:
      # Otherwise, just set the payload as json
      request_kwargs['json'] = payload or {}

    try:
      # Make the request
      response = request(self.base_url + route, **request_kwargs)
    except KeyboardInterrupt:
      # Allow the user to kill it if taking too long
      exit(0)

    # Create an abstract api response
    api_resp = AbstractApiResponse(response)

    return api_resp

  def build_request_headers(self, headers={}):
    """
    Build up headers for a request.

    Order in which headers are constructed:
      (1) Start as self.base_headers
      (2) Add key-val pairs from headers param
      (3) Add auth header

      **Note** Header keys can be overwritten if ^this order isn't noted

    :param dict headers: Request-specific headers
    :return: All headers for this request
    :rtype: dict
    """
    # Start with base headers
    all_headers = self.base_headers

    if headers:
      # Add request-specific headers if they exist
      for k, v in headers.items():
        all_headers[k] = headers[k]

    # Return early if no authorization header
    if not self.auth_header_name:
      return all_headers

    # auth_header_val_getter takes precedent over auth_header_val
    if self.auth_header_val_getter:
      all_headers[self.auth_header_name] = self.auth_header_val_getter()
    elif self.auth_header_val:
      all_headers[self.auth_header_name] = self.auth_header_val

    return all_headers


class AbstractApiResponse(object):
  """
  Interface to an API response from AbstractApi.
  """

  def __init__(self, response_obj):
    """
    :param response_obj:
      API response object returned by the 'requests' library
      :type: requests.Response
    """
    self.response_obj = response_obj
    self.headers = response_obj.headers
    self.status = response_obj.status_code
    self.ok = self.status in (200, 201)
    self.json = self.parse_json_resp()

  def parse_json_resp(self):
    """
    Attempt to parse a JSON response, defaulting to an empty dict.

    :return: Parsed JSON response
    :rtype: dict or list (dict if error)
    """
    try:
      return self.response_obj.json() or {}
    except:
      return {}