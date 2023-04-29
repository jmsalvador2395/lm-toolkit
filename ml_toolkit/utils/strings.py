import re

"""
this file contains functions for generating strings
"""
def now():
	"""
	returns a string formatted as YYYYMMDD-HHmmSS (Year, Month, Day, Hour, Minute, Second in order)
	"""

	from datetime import datetime
	return datetime.now().strftime('%Y%m%d-%H%M%S')

def clean_multiline(text):
	"""
	removes unwanted tabs and returns from a multiline string
	"""
	text = re.sub('\n *', '\n', text)
	text = re.sub('\n\t*', '\n', text)
	return text.strip('\n')
