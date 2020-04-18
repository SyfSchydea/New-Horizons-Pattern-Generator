#!/bin/python3

import sys
import argparse

import numpy as np
import cv2

from ternary import Trit
import colour
import palette
from log import Log
import instructions
import qr

def get_rng_seed():
	np.random.seed()
	return np.random.randint(2 ** 32 - 1)

# Pseudo main-method
# Load image, convert to pattern, output preview and instructions
def analyse_img(path, weight_map_path, img_out, instr_out, *,
		seed=None, retry_on_dupe=False, use_dithering=True, use_colour=Trit.maybe, verbosity=Log.INFO, new_leaf=False):
	# NH/NL allow 15 colours + transparent
	PALETTE_SIZE = 15

	log = Log(verbosity)

	if seed is None:
		seed = get_rng_seed()
		log.info("Using seed:", seed)
	
	if new_leaf:
		log.info("Generating a New Leaf pattern")

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

	game_palette, lab_palette = palette.pick_palette(img_lab, PALETTE_SIZE, seed, weight_map,
		retry_on_dupe=retry_on_dupe, new_leaf=new_leaf, log=log)

	# Convert the NH colours to BGR in the range 0 to 1
	if new_leaf:
		bgr_approx = palette.nl_to_bgr(game_palette)
		log.debug("NL-BGR Palette:\n", bgr_approx)
	else:
		hsv_approximated = colour.palette_nh_to_hsv(game_palette)
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

	log.debug("Indexed img:\n", indexed_img)

	# Print drawing instructions
	if new_leaf:
		log.info("Generating QR code...")
		# TODO: Allow user to enter their own Player and Town identifier data.
		qr_code = qr.generate("Generated pattern", game_palette, indexed_img)
		# TODO: Allow user to enter QR code filename
		qr.export_img("pattern-qr.png", qr_code)
	else:
		log.info("Writing instructions file...")
		instructions.write(instr_out, indexed_img, game_palette, use_colour=use_colour)

	# Generate BGR image using colour map and the BGR version of the approximated colour space
	# Export approximated image
	log.info("Exporting preview image...")
	palette.output_preview_image(img_out, indexed_img, bgr_approx)


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description=
		"Convert an image to a New Horizons custom pattern")

	parser.add_argument("input-file", help="Path to input image")
	parser.add_argument("-o", "--out", default=None,
		help="Path to save output preview image")
	parser.add_argument("-i", "--instructions-out", default=None,
		help="Path to save pattern instructions")

	parser.add_argument("-d", "--dithering", action="store_true",
		help="Use Floyd-Steinberg dithering when creating the pattern")

	parser.add_argument("-w", "--weight-map",
		help="Map of pixel weights for palette selection")

	parser.add_argument("-L", "--new-leaf", action="store_true",
		help="Generate a pattern using the New Leaf colour palette instead of the New Horizons"
			+ "palette. This option also allows a QR code to be generated for the pattern")

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

	# Fetch preview image filename
	if args.out is not None:
		preview_file = args.out
	elif args.new_leaf:
		preview_file = palette.DEFAULT_PREVIEW_FILENAME_NL
	else:
		preview_file = palette.DEFAULT_PREVIEW_FILENAME_NH

	# Validate --instructions-out <path>
	if args.instructions_out == "":
		sys.stderr.write("Instructions file output path cannot be empty\n")
		sys.exit(1)
	if args.instructions_out and args.new_leaf:
		sys.stderr.write("Cannot produce an instructions file for a New Leaf texture. Import using the QR code instead\n");
		sys.exit(1)

	instructions_file = args.instructions_out or instructions.DEFAULT_PATH

	# Fetch tty colour options
	use_colour = Trit.maybe
	if args.tty_colours:
		use_colour = Trit.true
	elif args.no_tty_colours:
		use_colour = Trit.false

	# Fetch verbosity options
	verbosity = Log.INFO
	if args.quiet is not None:
		verbosity -= args.quiet
	if args.verbose:
		verbosity += 1

	try:
		analyse_img(input_file, weight_map_path=args.weight_map, img_out=preview_file, instr_out=instructions_file,
			seed=args.seed, retry_on_dupe=args.retry_duplicate, use_dithering=args.dithering, use_colour=use_colour,
			new_leaf=args.new_leaf, verbosity=verbosity);
	except FileNotFoundError:
		sys.stderr.write("File does not exist or is not a valid image\n")
		sys.exit(1)

