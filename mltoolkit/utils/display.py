"""
this file contains functions for printing things in a certain format
"""
# external imports
import os

def title(title_str: str, fill_char: str='='):
	"""
	prints the given title centered in the terminal and surrounded by fill_char 

	:param title_str: the text to print in the middle
	:type title_str: str

	:param fill_char: the character to be used to fill out the rest of the line. (default '='
	:type fill_char: str
	"""

	# get terminal size
	col, lines = os.get_terminal_size()

	# print the title
	print('\n' + title_str.center(col, fill_char) + '\n')
