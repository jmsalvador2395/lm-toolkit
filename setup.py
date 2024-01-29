from setuptools import setup
from pathlib import Path


# read in requirements. this gets used in the setup function
requirements_dir = f'{Path(__file__).parent}/requirements.txt'
with open(requirements_dir, 'r') as f:
	requirements = f.read().splitlines()

setup(
	name='mltoolkit',
	version='0.0.1',
	py_modules=['mltk'],
	install_requires=requirements,
	entry_points={
		'console_scripts' : [
			'mltk = mltoolkit.entry_points:main',
		],
	}
)
