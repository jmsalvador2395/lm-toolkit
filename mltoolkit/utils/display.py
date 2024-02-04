"""
this file contains functions for printing things in a certain format
"""
# external imports
import os

# internal imports
from . import strings

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

def with_header(msg: str, header, color='green', end='\n'):
    """
    prints a message in the format '[<header>] <msg>' where the color <header> can be changed by the <color> argument

    Input:
        msg: str - the message to be displayed
        header: str - the header message. usually a single word
        color: str - the desired color for the header
    """
    colorizer = {
        'green': strings.green,
        'red': strings.red,
        'yellow': strings.yellow,
        'blue': strings.blue,
        'cyan': strings.cyan,
        'magenta': strings.magenta,
    }

    if color not in colorizer:
        raise ValueError(f'The color \'{color}\' is not supported')

    print('[' + colorizer[color](header) + '] ' + msg, end=end)

def warning(msg: str, end='\n'):
    with_header(msg, 'WARNING', color='yellow', end=end)

def debug(msg: str, end='\n'):
    with_header(msg, 'DEBUG', color='yellow', end=end)

def todo(msg: str, end='\n'):
    with_header(msg, 'TODO', color='yellow', end=end)

def error(msg: str, end='\n'):
    with_header(msg, 'ERROR', color='red', end=end)

def ok(msg: str, end='\n'):
    with_header(msg, 'OK', color='green', end=end)
    
def note(msg: str, end='\n'):
    with_header(msg, 'NOTE', color='magenta', end=end)

def in_progress(msg: str, end='\n'):
    with_header(msg, 'IN PROGRESS', color='blue', end=end)

def done(msg='', end='\n'):
    with_header(msg, 'DONE', color='green', end=end)
