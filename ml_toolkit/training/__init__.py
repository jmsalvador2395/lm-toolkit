# external imports
import yaml

# local imports
from ml_toolkit.menus import file_explorer
from ml_toolkit import utils
from .trainer import Trainer

def train(args):
	# open menu to find a config file if none is provided
	if args['cfg'] is None:
		config_path = file_explorer('choose a config file in')

	# check if the provided config path is valid
	else:
		config_path = args['cfg']
		utils.validate.path_exists(
			config_path, 
			extra_info='the given path is invalid'
		)

	# load config
	cfg = utils.files.load_yaml(config_path)

	trainer = Trainer(cfg)
	trainer.train()
