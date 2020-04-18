#!/bin/python3

# Functions for writing instructions files about how to create a pattern.

from ternary import Trit
import tty

# Default file path to save instructions files
DEFAULT_PATH = "nh-pattern-instructions.txt"

# Characters used to represent each colour in ascii output
DISPLAY_CHARS = "abcdefghijklmnopqrstuvwxyz"
PREV_FMT = tty.TextFormat(fg=tty.CYAN, faint=True)

EMPTY_CHAR = tty.FormattedString("·", tty.TextFormat(faint=True))

ACTIVE_CHAR = tty.FormattedString("#", tty.TextFormat(fg=tty.MAGENTA, bold=True))

# Format for heading text
HEADER_FMT = tty.TextFormat(fg=tty.YELLOW, bold=True)

# Format for colour HSV values in the above headers
COLOUR_VALUES_FMT = tty.TextFormat(fg=tty.RED)

# Write a single channel from the palette to the file.
#
# param file         - File to write to.
# param channel_idx  - Index of this channel in the colour space.
# param channel_name - Name of this channel in the colour space.
# param header_width - Characters to reserve at the start of
#                      each line for the names of channels.
# param palette      - Array of New Horizons colours.
def _write_palette_channel(file, channel_idx, channel_name, header_width, palette):
	file.write(tty.FormattedString(channel_name.rjust(header_width) + ": ", HEADER_FMT))
	file.write(tty.FormattedString("  ".join(str(col[channel_idx]).rjust(2) for col in palette)))
	file.write("\n")

# Write an ascii version of an indexed image to the file.
# Will include breaks every 16 pixels.
#
# param file          - File to write to.
# param indexed_img   - Array of colour indices for each pixel of the image.
# param display_chars - Indexable collection of characters to represent each colour.
def _write_indexed_img(file, indexed_img, display_chars):
	height, width = indexed_img.shape
	for y in range(height):
		for x in range(width):
			if x % 16 == 0:
				file.write(" ")

			idx = indexed_img[y, x]
			char = display_chars[idx]
			file.write(" ")
			file.write(char)

		file.write("\n")

		if y % 16 == 15:
			file.write("\n")

# Write instructions on how to draw this pattern to a file.
#
# param path_out    - Path to file to write to.
# param indexed_img - Array of colour indices for each pixel of the image.
# param palette     - Array of New Horizons colours
def write(path_out, indexed_img, palette, *, pattern_name="your pattern", use_colour=Trit.maybe):
	empty_pixel  = EMPTY_CHAR
	active_pixel = ACTIVE_CHAR

	height, width   = indexed_img.shape
	palette_size, _ =     palette.shape

	with open(path_out, "w") as file_raw:
		file = tty.TTY(file_raw, use_colour=use_colour)
		file.write(tty.FormattedString(f"How to draw {pattern_name}:\n\n", HEADER_FMT))

		file.write(tty.FormattedString("Colour palette:\n", HEADER_FMT))
		header_width = 11
		_write_palette_channel(file, 0, "Hue",        header_width, palette)
		_write_palette_channel(file, 1, "Vividness",  header_width, palette)
		_write_palette_channel(file, 2, "Brightness", header_width, palette)
		file.write("\n")

		# Print map for each colour in the palette.
		display_chars = [empty_pixel] * palette_size
		for i, col in enumerate(palette):
			file.write(tty.FormattedString(f"Colour {i} (",               HEADER_FMT))
			file.write(tty.FormattedString(" ".join(str(x) for x in col), COLOUR_VALUES_FMT))
			file.write(tty.FormattedString("):\n",                        HEADER_FMT))
			display_chars[i] = active_pixel
			_write_indexed_img(file, indexed_img, display_chars)
			prev_pixel = tty.FormattedString(DISPLAY_CHARS[i], PREV_FMT)
			display_chars[i] = prev_pixel
			file.write("\n")

		file.reset_all()
