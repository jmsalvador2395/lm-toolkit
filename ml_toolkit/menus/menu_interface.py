# external imports
from simple_term_menu import TerminalMenu
import os
import sys
from pathlib import Path

# local imports
from ml_toolkit.utils.display import prompt_yes_no

def file_explorer(prompt):
	"""this displays a menu for finding and returning a desired file path

	:rtype str:
	"""
	cwd = os.getcwd()
	directory = ['../'] + os.listdir(cwd)
	
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
			if prompt_yes_no('exit?'):
				sys.exit()

		fname = directory[choice]
		if fname == '../':
			target = str(Path(cwd).parent)
		else:
			target = f'{cwd}/{fname}'


		# check if the file is good
		if os.path.isfile(target):
			choice = prompt_yes_no(f'is {target} correct?')
			if choice:
				return target
		else:
			cwd = target
			directory = ['../'] + os.listdir(cwd)
