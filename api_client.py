"""
Glassdoor-NLP API client
"""
import os
from defs import auth_header_name, auth_header_env
from api_utils.abstract_api import AbstractApi

api_url = os.environ.get('API_URL', 'http://localhost:5000/api')

# Configure an API client
api_client = AbstractApi(base_url=api_url,
                         auth_header_name=auth_header_name,
                         auth_header_val=os.environ.get(auth_header_env))