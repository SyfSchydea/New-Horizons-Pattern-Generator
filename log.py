#!/bin/python3

import sys

# Class for controlling logging and verbosity levels
class Log:
	# Verbosity Levels
	ERROR   = 0
	WARNING = 1
	INFO    = 2
	DEBUG   = 3

	def __init__(self, verbosity=INFO, file=sys.stdout):
		self.verbosity = verbosity
		self.file = file

	# Log information about something which prevents
	# the script from completing normally.
	def error(self, *msg):
		if self.verbosity >= Log.ERROR:
			print(*msg, file=sys.stderr)

	# Log information about something which may be undesirable,
	# but doesn't stop the script from continuing.
	def warn(self, *msg):
		if self.verbosity >= Log.WARNING:
			print("Warning:", *msg, file=self.file)

	# Log information about what the script is doing.
	def info(self, *msg):
		if self.verbosity >= Log.INFO:
			print(*msg, file=self.file)

	# Log information which may be useful for debugging,
	# but the end-user typically doesn't need to see.
	def debug(self, *msg):
		if self.verbosity >= Log.DEBUG:
			print(*msg, file=self.file)
