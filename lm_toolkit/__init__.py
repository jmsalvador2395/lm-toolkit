# external imports
import os
import click

# local imports
from .evaluator import evaluate
from .trainer import train
from .unit_tester import unit_test


@click.group
def main():
	print('okay i think it worked')
	print(os.getcwd())

main.add_command(evaluate)
main.add_command(train)
main.add_command(unit_test)
