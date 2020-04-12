#!/bin/python3

# Classes to handle printing and formatting of text printed to a tty.

import sys
from copy import copy
import re

from ternary import Trit

# Control Sequence Introducer
CSI = "\x1b["

# Colour values which may be set to foreground or background in a tty
BLACK        = 0
RED          = 1
GREEN        = 2
BLUE         = 4
YELLOW       = RED  | GREEN
MAGENTA      = RED  | BLUE
CYAN         = BLUE | GREEN
WHITE        = RED  | GREEN | BLUE
RESET_COLOUR = 9

COLOUR_NAMES = {
	BLACK:   "black",
	RED:     "red",
	GREEN:   "green",
	BLUE:    "blue",
	YELLOW:  "yellow",
	MAGENTA: "magenta",
	CYAN:    "cyan",
	WHITE:   "white",
}

# Normal terminal settings
FG_DEFAULT = WHITE

# Ensure the colour is a normal colour code
def sanitise_fg(colour):
	colour = int(colour)
	colour %= 10

	if colour == RESET_COLOUR:
		colour = FG_DEFAULT

	return colour

# Represents a text formatting style
class TextFormat:
	def __init__(self, *, fg=FG_DEFAULT, bold=Trit.false, faint=Trit.false):
		self.fg = fg

		# whether or not the FG is set to the specified value.
		# If false, the FG is actually on the default value.
		self.fg_set = Trit.true

		self.bold  = Trit.of(bold)
		self.faint = Trit.of(faint)

	# Fetch the foreground colour
	#
	# return - Tuple of (fg_set, fg_colour)
	def get_fg(self):
		# fg_set being false, and fg being FG_DEFAULT are equivalent
		if self.fg == FG_DEFAULT or self.fg_set.definitely_false():
			self.fg = FG_DEFAULT
			self.fg_set = Trit.true

		return self.fg_set, self.fg

	def fg_matches(self, other_fg):
		fg_set, fg = self.get_fg()

		if not self.fg_set.known:
			return False

		return bool(fg_set) and self.fg == sanitise_fg(other_fg)

	def set_fg(self, fg):
		self.fg = sanitise_fg(fg)
		self.fg_set = Trit.true

	# Ambiguously reset formatting to default.
	def maybe_reset(self):
		self.bold = self.bold.maybe_set(False)
		self.faint = self.faint.maybe_set(False)
		self.fg_set = self.fg_set.maybe_set(False)
	
	# Return true if and only if `other` is an identical TextFormat
	def __eq__(self, other):
		if not isinstance(other, TextFormat):
			return False

		return (self.fg_matches(other.fg)
			and self.bold.definitely_equals(other.bold)
			and self.faint.definitely_equals(other.faint))

	def __str__(self):
		words = []

		if self.bold.maybe_true():
			bold = "bold"

			if not self.bold.known:
				bold = "maybe " + bold

			words.append(bold)

		if self.faint.maybe_true():
			faint = "faint"

			if not self.faint.known:
				faint = "maybe " + faint

			words.append(faint)

		fg_set, fg = self.get_fg()
		if fg_set.maybe_true() and fg != FG_DEFAULT:
			col = COLOUR_NAMES[fg]

			if not fg_set.known:
				col += " or default"

			words.append(col)

		if len(words) <= 0:
			words.append("normal")

		return ", ".join(words)

# Class for setting tty colour/formatting
class TTY:
	def __init__(self, file=sys.stdout, fmt=TextFormat(), *, use_colour=Trit.maybe):
		self.file = file
		self.fmt = copy(fmt)

		use_colour = Trit.of(use_colour)
		self.use_colour = bool(use_colour) if use_colour.known else file.isatty()

		# This property keeps track of formatting which it should be using,
		# However, it may not always be using this formatting if no non-
		# whitespace characters have been printed since setting the format.
		self.pending_fmt = copy(fmt)

	# Set foreground colour
	#
	# param colour - Colour to set foreground to.
	#                Should be one of the colour constants defined below.
	def set_fg(self, colour=7):
		if not self.use_colour:
			return

		# If the terminal is already outputting
		# in this colour, don't change that.
		if self.fmt.fg_matches(colour):
			return

		# Write control code
		self.fmt.set_fg(colour)
		self.file.write(CSI + "3" + str(self.fmt.fg) + "m")

	# Turn bold on
	def set_bold(self):
		if not self.use_colour:
			return

		if self.fmt.bold.definitely_true():
			return

		self.fmt.bold = Trit.true
		self.file.write(CSI + "1m")

	# Turn faint on
	def set_faint(self):
		if not self.use_colour:
			return

		if self.fmt.faint.definitely_true():
			return

		self.file.write(CSI + "2m")
		self.fmt.faint = Trit.true

	# Turn bold and faint off
	def reset_weight(self):
		if not self.use_colour:
			return

		if self.fmt.bold.definitely_false() and self.fmt.faint.definitely_false():
			return

		self.fmt.bold  = Trit.false
		self.fmt.faint = Trit.false
		self.file.write(CSI + "22m")

	# Reset all colours and formatting to default
	def reset_all(self):
		if not self.use_colour:
			return

		# Skip if already at default settings
		default_fmt = TextFormat()
		if self.fmt == TextFormat():
			return

		self.fmt = default_fmt
		self.file.write(CSI + "0m")


	# Set bold/faint properties
	def set_weight(self, bold=False, faint=False):
		bold  = bool(bold)
		faint = bool(faint)

		# If either bold or faint needs to be turned
		# off, print the reset_weight code
		if ((not bold and self.fmt.bold.maybe_true())
				or (not faint and self.fmt.faint.maybe_true())):
			self.reset_weight()

		# Turn bold and/or faint on individually
		if bold:
			self.set_bold()
		if faint:
			self.set_faint()

	# Set all format fields efficiently
	def set_format(self, fmt):
		# Exceptional case: set all fields to default
		if fmt == TextFormat():
			self.reset_all()
			return

		# Set fields individually.
		# Each fields' methods will do nothing if nothing needs to be done.
		self.set_fg(fmt.fg)
		self.set_weight(fmt.bold, fmt.faint)
	
	# Simulate the format maybe getting reset on newline.
	# When outputting directly to a tty, this does not happen,
	# But when outputting to `less -R`, this does, hence the ambiguity.
	def refresh_fmt(self):
		self.fmt.maybe_reset()


	# Write text to the tty.
	#
	# param string - May be a str or a FormattedString.
	def write(self, string):
		if isinstance(string, FormattedString):
			self.pending_fmt = string.fmt
			string = string.string


		lines = string.split("\n")

		for i, line in enumerate(lines):
			if i > 0:
				self.file.write("\n")
				self.refresh_fmt()

			# Avoid updating formatting for whitespace
			if (self.pending_fmt != self.fmt
					and not re.fullmatch(r"[ \n\t\r]*", line)):
				self.set_format(self.pending_fmt)

			self.file.write(line)


# Represents a string in a specific colour.
class FormattedString:
	def __init__(self, string, fmt=TextFormat()):
		self.string = string
		self.fmt = copy(fmt)

if __name__ == "__main__":
	# Main method to test output
	# Should probably remove this stuff before merging to master
	
	f = TTY(use_colour=True)
	f.write(FormattedString("blue text\n", TextFormat(fg=BLUE, bold=True)))
	f.write(FormattedString("faint text\n",  TextFormat(faint=True)))
	f.write(FormattedString("normal text\n", TextFormat()))
	f.write(FormattedString("red text\n",  TextFormat(fg=RED,  bold=True)))
	f.write(FormattedString("red text", TextFormat(fg=RED)))
	f.write(FormattedString(" and", TextFormat(fg=RED)))
	f.write(FormattedString(" more red text", TextFormat(fg=RED)))
