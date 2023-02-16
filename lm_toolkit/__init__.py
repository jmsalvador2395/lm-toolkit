# external imports
import os
import click

# local imports
from .evaluator import evaluate
from .trainer import train
from .unit_tester import unit_test


@click.group
def main():
	"""
	this is the entry point for the program
	look at the local functions imported from the top to see where the next steps are executed.
	to add more options, import the function, apply the same decorators that you see in the function
	definitions and add an 'add_argument()' entry below
	"""
	pass

main.add_command(evaluate)
main.add_command(train)
main.add_command(unit_test)
