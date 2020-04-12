#!/bin/python3

import sys
import argparse

import numpy as np
import cv2

from ternary import Trit
import tty
import colour
import palette
from log import Log

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
def write_instructions(path_out, indexed_img, palette, *, pattern_name="your pattern", use_colour=Trit.maybe):
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

def output_preview_image(path_out, indexed_img, bgr_palette):
	height, width = indexed_img.shape
	palette_size, depth = bgr_palette.shape

	bgr_out_palette = (bgr_palette * 255).astype(np.uint8)

	bgr_img = np.zeros((height, width, depth), dtype=np.uint8)
	for x in range(width):
		for y in range(height):
			colour_idx = indexed_img[y, x]
			bgr_colour = bgr_out_palette[colour_idx]
			bgr_img[y, x] = bgr_colour

	cv2.imwrite(path_out, bgr_img)

def get_rng_seed():
	np.random.seed()
	return np.random.randint(2 ** 32 - 1)

# Pseudo main-method
# Load image, convert to NH pattern, output preview and instructions
def analyse_img(path, weight_map_path, img_out, instr_out, *,
		seed=None, retry_on_dupe=False, use_dithering=True, use_colour=Trit.maybe,
		verbosity=Log.INFO):
	# NH allows 15 colours + transparent
	PALETTE_SIZE = 15

	log = Log(verbosity)

	if seed is None:
		seed = get_rng_seed()
		log.info("Using seed:", seed)

	# Input image as BGR(255)
	log.info("Opening image...")
	img_raw = cv2.imread(path)
	if img_raw is None:
		raise FileNotFoundError()

	log.info("Converting to Lab...")
	img_lab = cv2.cvtColor(img_raw.astype(np.float32) / 255, cv2.COLOR_BGR2Lab)

	weight_map = None
	if weight_map_path is not None:
		weight_map = cv2.imread(weight_map_path, 0).astype(np.float32)

	nh_palette, lab_palette = palette.pick_palette(img_lab, PALETTE_SIZE, seed, weight_map,
		retry_on_dupe=retry_on_dupe, log=log)

	# Convert the NH colours to BGR(1) to Lab
	hsv_approximated = colour.palette_nh_to_hsv(nh_palette)
	log.debug("NH-HSV Palette:\n", hsv_approximated)
	bgr_approx = colour.convert_palette(hsv_approximated, cv2.COLOR_HSV2BGR)
	log.debug("NH-BGR Palette:\n", bgr_approx)

	# Create indexed image using clusters
	log.info("Creating indexed image...")
	if use_dithering:
		lab_approx = colour.convert_palette(bgr_approx, cv2.COLOR_BGR2Lab)
		indexed_img = palette.create_indexed_img_dithered(img_lab, lab_approx)
	else:
		indexed_img = palette.create_indexed_img_threshold(img_lab, lab_palette)

	# Print drawing instructions
	write_instructions(instr_out, indexed_img, nh_palette, use_colour=use_colour)

	# Generate BGR image using colour map and the BGR version of the approximated colour space
	# Export approximated image
	log.info("Exporting image...")
	output_preview_image(img_out, indexed_img, bgr_approx)

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description=
		"Convert an image to a New Horizons custom pattern")

	parser.add_argument("input-file", help="Path to input image")
	parser.add_argument("-o", "--out", default="nh-pattern.png",
		help="Path to save output preview image")
	parser.add_argument("-i", "--instructions-out", default="nh-pattern-instructions.txt",
		help="Path to save pattern instructions")

	parser.add_argument("-d", "--dithering", action="store_true",
		help="Use Floyd-Steinberg dithering when creating the pattern")

	parser.add_argument("-w", "--weight-map",
		help="Map of pixel weights for palette selection")

	parser.add_argument("-r", "--retry-duplicate", action="store_true",
		help="Retry palette generation until there are no duplicate colours")

	parser.add_argument("-s", "--seed", type=int, default=None,
		help="RNG seed for K-Means initialisation")

	tty_colour_group = parser.add_mutually_exclusive_group()
	tty_colour_group.add_argument("-c", "--tty-colours", action="store_true",
		help="Always use tty colours when printing the instructions file")
	tty_colour_group.add_argument("--no-tty-colours", action="store_true",
		help="Never use tty colours when printing the instructions file")

	verbosity_group = parser.add_mutually_exclusive_group()
	verbosity_group.add_argument("-q", "--quiet", action="count",
		help="Print less information to stdout. May be stacked "
			+ "up to three times to hide warnings and errors")
	verbosity_group.add_argument("-v", "--verbose", action="store_true",
		help="Print more debug info to stdout");

	args = parser.parse_args()

	input_file = getattr(args, "input-file")

	use_colour = Trit.maybe
	if args.tty_colours:
		use_colour = Trit.true
	elif args.no_tty_colours:
		use_colour = Trit.false

	verbosity = Log.INFO
	if args.quiet is not None:
		verbosity -= args.quiet
	if args.verbose:
		verbosity += 1

	try:
		analyse_img(input_file, weight_map_path=args.weight_map,
			img_out=args.out, instr_out=args.instructions_out,
			seed=args.seed, retry_on_dupe=args.retry_duplicate,
			use_dithering=args.dithering, use_colour=use_colour,
			verbosity=verbosity);
	except FileNotFoundError:
		sys.stderr.write("File does not exist or is not a valid image\n")
		sys.exit(1)

