# external imports
import click

# local imports
from .menus import file_explorer
from .utils import validate

@click.command()
@click.option(
	'--config_path',
	default=None, 
	help=(
		'the path of the config file that defines the training procedure.\n' +
		'if none is provided a menu will open to allow you to find your file from the calling directory'
	)
)
def train(config_path):
	# open menu to find a config file if none is provided
	if config_path is None:
		config_path = file_explorer('choose a config file in')
	# check if the provided config path is valid
	else:
		validate.path_exists(
			config_path, 
			extra_info='the specified config file is invalid'
		)
			
	#TODO put config file reader here
