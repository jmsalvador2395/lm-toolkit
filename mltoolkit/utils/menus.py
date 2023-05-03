# external imports
from simple_term_menu import TerminalMenu
import os
import sys
from pathlib import Path

# local imports
from mltoolkit.utils.display import binary_prompt
from mltoolkit.utils import files

def file_explorer(prompt, start_path=None):
	"""this displays a menu for finding and returning a desired file path

	:rtype str:
	"""
	cwd = os.getcwd() if start_path is None else files.get_full_path(start_path)
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
			directory = ['../'] + os.listdir(cwd)
