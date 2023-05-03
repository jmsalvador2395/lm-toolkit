import argparse

def parse():
	"""
	builds and returns the program argument parser
	"""
	# base parser
	parser = argparse.ArgumentParser()

	# create subparser for procedures
	subparser = parser.add_subparsers(
		description='decides on which procedure to run',
		required=True,
		dest='procedure',
	)

	# add subparser for training procedure
	parser_train = subparser.add_parser('train')
	parser_train.add_argument(
		'-c',
		'--cfg',
		help='config path',
		default=None,
	)

	# parser for the evaluation procedure
	parser_eval = subparser.add_parser('eval')
	parser_eval.add_argument(
		'--checkpoint',
		help='checkpoint path'
	)

	# parser for unit testing
	parser_ut = subparser.add_parser('unit_test')

	return parser.parse_args()
