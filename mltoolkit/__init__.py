# external imports
import os

# local imports
from mltoolkit import (
	arguments, 
	unit_testing,
	evaluation,
	utils
)
from .train import train


def main():
	"""
	this is the entry point for the program
	look at the local functions imported from the top to see where the next steps are executed.
	to add more options, import the function, apply the same decorators that you see in the function
	definitions and add an 'add_argument()' entry below
	"""

	# read args
	args = arguments.parse()

	# choose execute the desired procedure
	match args.procedure:
		case 'train':
			train(args)
		case 'evaluate':
			evaluator.evaluate(args)
		case 'unit_test':
			unit_tester.unit_test(args)
		case _:
			raise NotImplementedError(utils.strings.clean_multiline(
				"""
				Procedure added to args but case not added to main function in <project root>/lm_toolkit/__init__.py.
				"""
			))
			
