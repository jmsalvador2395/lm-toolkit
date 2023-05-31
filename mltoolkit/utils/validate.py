"""
this file contains functions used for error checking
"""
# external imports
import os
from colorama import Fore, Style

# local imports
from . import display

def path_exists(path, extra_info=None):
    """
    checks if a given path exists and throws an error otherwise

    Input:
        path[str]: the path to check on
        extra_info[str]: any additional text to append to the basic message
    """
    # build message string
    message = f'The path \'{path}\' does not exist.'
    if message:
        message += f'\n{extra_info if extra_info is not None else ""}'

    if not os.path.exists(path):
        display.error(message)
        raise FileNotFoundError()

def in_dict(key, dictionary, dict_name):
    """
    checks if a given key is in a dictionary. providing the name of the dictionary helps me identify problems quicker

    Input:
        key[obj]: a dictionary key
        dictionary[dict]: the dictionary in question
        dict_name[str]: name of the dictionary
    """
    if key not in dictionary:
        display.error(
            f'the key \'{str(key)}\' was not found in the dictionary named \'{dict_name}\''
        )
        raise KeyError()


def is_assigned(var, name, extra_info=None):
    """
    checks if a variable is not None and throws an error otherwise

    Input:
        var[obj]: the variable to check
        name[str]: the variable name
        extra_info[str]: any additional text to append to the basic message
    """
    
    # define message
    msg = f'variable \'{name}\' not initialized.'
    if extra_info is not None:
        msg += f' {extra_info}'

    # check if variable is assigned
    if var is None:
        display.error(msg)
        raise ValueError()
