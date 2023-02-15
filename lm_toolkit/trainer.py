import click

@click.command()
@click.argument(
	'config_path',
	default=None, 
	help=(
		'the path of the config file that defines the training procedure.\n' +
		'if none is provided a menu will open to allow you to find your file from the calling directory'
	)
		
	)
def train(config_path=None):
	print('train')
	print(f'config path is: {config_path}')
