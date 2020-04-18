#!/bin/python3

import sys
import tty

ERROR_FORMAT   = tty.TextFormat(fg=tty.RED)
WARNING_FORMAT = tty.TextFormat(fg=tty.YELLOW)
INFO_FORMAT    = tty.TextFormat()
DEBUG_FORMAT   = tty.TextFormat(fg=tty.CYAN)

# Class for controlling logging and verbosity levels
class Log:
	# Verbosity Levels
	ERROR   = 0
	WARNING = 1
	INFO    = 2
	DEBUG   = 3

	def __init__(self, verbosity=INFO, file=sys.stdout):
		self.verbosity = verbosity
		self.file = tty.TTY(file)
		self.err_tty = tty.TTY(sys.stderr)

	def _print(self, file, msg, fmt):
		msg_text = " ".join(str(s) for s in msg) + "\n"
		file.write(tty.FormattedString(msg_text, fmt))
		if file is self.file:
			self.err_tty.refresh_fmt()
		else:
			self.file.refresh_fmt()

	# Log information about something which prevents
	# the script from completing normally.
	def error(self, *msg):
		if self.verbosity >= Log.ERROR:
			self._print(self.err_tty, msg, ERROR_FORMAT)

	# Log information about something which may be undesirable,
	# but doesn't stop the script from continuing.
	def warn(self, *msg):
		if self.verbosity >= Log.WARNING:
			self._print(self.file, ["Warning:", *msg], WARNING_FORMAT)

	# Log information about what the script is doing.
	def info(self, *msg):
		if self.verbosity >= Log.INFO:
			self._print(self.file, msg, INFO_FORMAT)

	# Log information which may be useful for debugging,
	# but the end-user typically doesn't need to see.
	def debug(self, *msg):
		if self.verbosity >= Log.DEBUG:
			self._print(self.file, msg, DEBUG_FORMAT)
