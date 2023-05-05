# external imports
from simple_term_menu import TerminalMenu
import os
import sys
from pathlib import Path

# local imports
from mltoolkit.utils import files

def file_explorer(prompt, start_path=None):
    """this displays a menu for finding and returning a desired file path

    :rtype str:
    """
    cwd = os.getcwd() if start_path is None else files.full_path(start_path)
    directory = ['../'] + sorted(os.listdir(cwd))
    
    while True:
        menu = TerminalMenu(
            directory,
            title=f'\n{prompt}: {cwd}',
            skip_empty_entries=True,
            clear_screen=True,
            show_shortcut_hints=True,
            show_shortcut_hints_in_status_bar=False,
        )
        choice = menu.show()

        # exits if the user quit the menu
        if choice is None:
            if binary_prompt('exit?'):
                sys.exit()

        fname = directory[choice]
        if fname == '../':
            target = str(Path(cwd).parent)
        else:
            target = f'{cwd}/{fname}'


        # check if the file is good
        if os.path.isfile(target):
            choice = binary_prompt(f'is {target} correct?')
            if choice:
                return target
        else:
            cwd = target
            directory = ['../'] + sorted(os.listdir(cwd))


def binary_prompt(message: str, default_no: bool=True) -> bool:
    """
    used to prompt the user for any yes or no options

    :param message: the message to be displayed before prompting
    :type message: str

    :param default_no: sets the default action. if default_no then prompt returns true unles 'n' or  'no' is specified

    :rtype bool: yes or no converted to true or false
    """

    choice = input(f'{message} (y/n): ').lower()
    if default_no:
        return not choice in ['n', 'no']
    else:
        return choice in ['y', 'yes']
