from api_client import ApiClient
from argparse import ArgumentParser


def parse_args():
  parser = ArgumentParser()
  parser.add_argument('--pros', '-p', default='')
  parser.add_argument('--cons', '-c', default='')
  return parser.parse_args()


if __name__ == '__main__':
  # Get pros and cons from args
  args = parse_args()

  # Initialize api client
  client = ApiClient()

  # Make prediction
  score = client.predict(pros=args.pros, cons=args.cons)

  print('Score: {}'.format(score))
