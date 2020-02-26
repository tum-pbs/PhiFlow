import os


with open(os.path.join(os.path.dirname(__file__), 'VERSION'), 'r') as version_file:
    __version__ = version_file.read()
