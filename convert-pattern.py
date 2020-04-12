#!/bin/python3

import sys
import argparse

import numpy as np
import cv2

from ternary import Trit
import tty
import colour

# Characters used to represent each colour in ascii output
DISPLAY_CHARS = "abcdefghijklmnopqrstuvwxyz"
PREV_FMT = tty.TextFormat(fg=tty.CYAN, faint=True)

EMPTY_CHAR = tty.FormattedString("·", tty.TextFormat(faint=True))

ACTIVE_CHAR = tty.FormattedString("#", tty.TextFormat(fg=tty.MAGENTA, bold=True))

# Format for heading text
HEADER_FMT = tty.TextFormat(fg=tty.YELLOW, bold=True)

# Format for colour HSV values in the above headers
COLOUR_VALUES_FMT = tty.TextFormat(fg=tty.RED)

# Find which center a given data point is closest to.
#
# @param item    - m-length array data point.
# @param centers - k*m array of cluster centers.
#
# @return        - Index of the closest center.
def find_closest(item, centers):
	best_idx = -1
	best_distance = float("inf")

	for i, center in enumerate(centers):
		distance = colour.lab_distance(item, center)

		if distance < best_distance:
			best_distance = distance
			best_idx = i

	return best_idx

# K-means++ initialisation
# Pick the first point randomly, subsequent points are
# the furthest point from their nearest center.
#
# param items - n*m array of data points. Where n = number of data
#               points, and m = number of dimensions per data point.
# param k     - Number of clusters to find.
# param rng   - RandomState object to draw random numbers from.
#
# return      - Array of initial cluster centers.
def k_means_pp_init(items, k, rng):
	n, dimensions = items.shape

	# Create empty array of centers
	centers = np.zeros((k, dimensions), dtype=np.float32)

	# Create array of item distances to their nearest centers
	item_distances = np.zeros(n, dtype=np.float32) + np.inf

	# Generate first center using uniform weights
	first_idx = rng.choice(range(n))
	centers[0] = items[first_idx]

	# For each remaining center
	for center_idx in range(1, k):
		prev_center = centers[center_idx - 1]
		for item_idx in range(n):
			# Update item distance
			item = items[item_idx]
			dist = colour.lab_distance(item, prev_center)
			if dist < item_distances[item_idx]:
				item_distances[item_idx] = dist

		# Choose next center using weights equal to the
		# square of the distance to the nearest point.
		weights = item_distances ** 2
		weights = weights / sum(weights)
		idx = rng.choice(range(n), p=weights)
		centers[center_idx] = items[idx]

	# Return all centers.
	return centers

# K means clustering algorithm.
#
# param items - n*m array of data points. Where n = number of data
#               points, and m = number of dimensions per data point.
# param k     - Number of clusters to find.
#
# return      - k*m array of cluster centers.
def k_means(items, k, weight_map, *, seed=None):
	rng = np.random.RandomState(seed)

	n, dimensions = items.shape
	dtype = items.dtype

	# Initialise centers
	centers = k_means_pp_init(items, k, rng)

	while True:
		# Sum of all points matched to each center
		center_totals = np.zeros((k, dimensions), dtype=dtype)

		# Number of points matched to each center
		center_matches = np.zeros(k, dtype=np.float32)

		for item, weight in zip(items, weight_map):
			# print(item, weight)
			close_idx = find_closest(item, centers)

			center_matches[close_idx] += weight
			center_totals[close_idx] += item * weight

		# Calculate each center's mean value
		center_means = np.copy(center_totals)
		reset_center = False
		for i in range(k):
			if center_matches[i] == 0:
				# If a center has no matches, leave it at its existing value
				center_means[i] = centers[i]
				reset_center = True
				continue

			center_means[i] /= center_matches[i]

		# If the new centers match the previous centers, return
		if not reset_center and np.array_equal(centers, center_means):
			return centers;

		# Otherwise, use the means as the centers for the next iteration
		centers = center_means

# Create an indexed image using a given image and a palette of colours.
# No dithering. Take a closest colour to each pixel value.
#
# param img     - Image as a NumPy array of size (height, width, depth)
# param palette - Array of colours to use, in the same colour space as img.
#                 Array dimensions: (n, depth) where n is
#                 the number of colours in the palette.
#
# return        - Indexed image as (height, width) size array of ints
def create_indexed_img_threshold(img, palette):
	height, width, _ = img.shape
	indexed = np.zeros((height, width), dtype=np.uint8)

	for x in range(width):
		for y in range(height):
			indexed[y, x] = find_closest(img[y, x], palette)

	return indexed

# Create an indexed image from an image and colour
# palette, using Floyd-Steinberg dithering.
#
# param img     - Image as a NumPy array of size (height, width, depth)
# param palette - Array of colours to use, in the same colour space as img.
#                 Array dimensions: (n, depth) where n is
#                 the number of colours in the palette.
#
# return        - Indexed image as (height, width) size array of ints
def create_indexed_img_dithered(img, palette):
	# Weighting of quantisation error diffused to neighbouring pixels.
	# In order of (y, x): (+0, +1), (+1, -1), (+1, +0), (+1, +1)
	# Traditional Floyd-Steinberg uses [7, 3, 5, 1].
	DIFFUSION_COEFFS = [7, 3, 5, 1]

	# Smallest unit by which error is diffused.
	# Error will only be diffused in integer multiples of
	# this (as long as the coefficients are integers).
	DIFFUSION_UNIT = sum(DIFFUSION_COEFFS)

	height, width, _ = img.shape

	indexed = np.zeros((height, width), dtype=np.uint8)
	img_target = img.copy()

	for y in range(height):
		not_last_row = y != height - 1

		range_x = range(width)
		ahead = 1
		first_col = 0
		last_col = width - 1

		# This is python, so use serpentine scanning.
		if y % 2 == 1:
			range_x = reversed(range_x)
			ahead *= -1
			last_col, first_col = first_col, last_col

		behind = -ahead

		for x in range_x:
			not_first_col = x != first_col
			not_last_col  = x != last_col

			target_colour = img_target[y, x]
			idx = find_closest(target_colour, palette)
			indexed[y, x] = idx

			colour = palette[idx]
			quant_error = (target_colour - colour) / DIFFUSION_UNIT

			if not_last_col:
				img_target[y, x + ahead] += quant_error * DIFFUSION_COEFFS[0]
			if not_last_row:
				if not_first_col:
					img_target[y + 1, x + behind] += quant_error * DIFFUSION_COEFFS[1]
				img_target[y + 1, x] += quant_error * DIFFUSION_COEFFS[2]
				if not_last_col:
					img_target[y + 1, x + ahead] += quant_error * DIFFUSION_COEFFS[3]

	return indexed

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
	file.write(tty.FormattedString("  ".join(str(colour[channel_idx]).rjust(2) for colour in palette)))
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

class Log:
	# Verbosity Levels
	ERROR   = 0
	WARNING = 1
	INFO    = 2
	DEBUG   = 3

	def __init__(self, verbosity, file=sys.stdout):
		self.verbosity = verbosity
		self.file = file

	def error(self, *msg):
		if self.verbosity >= Log.ERROR:
			print(*msg, file=sys.stderr)

	def warn(self, *msg):
		if self.verbosity >= Log.WARNING:
			print("Warning:", *msg, file=self.file)

	def info(self, *msg):
		if self.verbosity >= Log.INFO:
			print(*msg, file=self.file)

	def debug(self, *msg):
		if self.verbosity >= Log.DEBUG:
			print(*msg, file=self.file)

def get_rng_seed():
	np.random.seed()
	return np.random.randint(2 ** 32 - 1)

# Take an image in Lab colour space, and output a colour palette in NH colours
#
# param img           - Input Lab image.
# param palette_size  - Number of colours to use in the palette.
# param seed          - Starting RNG seed.
# param retry_on_dupe - If truthy, when a palette containing a duplicate colour
#                       is generated, the RNG seed will be incremented, and the
#                       process will repeat until a unique palette is returned.
# param log           - Log to use to output info.
#
# return              - Tuple of (NH Palette as array, Lab palette as array)
def pick_palette(img, palette_size, seed, weight_map, *, retry_on_dupe=False, log=Log(Log.ERROR)):
	# Generate colour palette using K-Means
	log.info("Finding suitable colour palette...")
	height, width, depth = img.shape
	colours = img.reshape((height * width, depth))

	if weight_map is not None:
		weight_map = weight_map.reshape(height * width)
	else:
		weight_map = np.ones(height * width)

	while True:
		colour_palette = k_means(colours, palette_size, weight_map, seed=seed)
		log.debug("Lab Palette:", colour_palette)

		# Convert the palette into HSV
		log.info("Converting palette to HSV...")
		hsv_palette = colour.convert_palette(colour_palette, cv2.COLOR_Lab2RGB, cv2.COLOR_RGB2HSV)
		log.debug("HSV Palette:\n", hsv_palette)

		# Round to ACNH's colour space
		log.info("Converting to NH colours...")
		nh_palette = colour.palette_hsv_to_nh(hsv_palette)
		log.debug("NH Palette:\n", nh_palette)

		# Check if any colours are identical after rounding
		duplicate_colours = colour.get_duplicate_colours(nh_palette)
		if len(duplicate_colours) <= 0:
			return nh_palette, colour_palette

		if not retry_on_dupe:
			log.warn("Repeated colours in palette:", *duplicate_colours)
			return nh_palette, colour_palette

		seed += 1
		log.info("Palette has duplicate colours. Retrying with seed:", seed)

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

	nh_palette, lab_palette = pick_palette(img_lab, PALETTE_SIZE, seed, weight_map,
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
		indexed_img = create_indexed_img_dithered(img_lab, lab_approx)
	else:
		indexed_img = create_indexed_img_threshold(img_lab, lab_palette)

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

