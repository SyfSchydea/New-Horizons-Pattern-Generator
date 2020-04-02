#!/bin/python3

import sys
import argparse

import numpy as np
import cv2

# Number of values per dimension in NH's colour space
NH_DEPTH_H = 30
NH_DEPTH_S = 15
NH_DEPTH_V = 15

# Max values in OpenCV's HSV colour space
HSV_H_MAX = 360
HSV_S_MAX =   1
HSV_V_MAX =   1

# Characters used to represent each colour in ascii output
DISPLAY_CHARS = "abcdefghijklmnopqrstuvwxyz"
EMPTY_COLOUR = "·"
ACTIVE_COLOUR = "#"

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
		distance = np.linalg.norm(center - item)

		if distance < best_distance:
			best_distance = distance
			best_idx = i

	return best_idx

# K means clustering algorithm.
#
# param items - n*m array of data points. Where n = number of data
#               points, and m = number of dimensions per data point.
# param k     - Number of clusters to find.
#
# return      - k*m array of cluster centers.
def k_means(items, k, *, seed=None):
	rng = np.random.RandomState(seed)

	n, dimensions = items.shape
	dtype = items.dtype

	# Initialise centers randomly
	# TODO: K-Means++ initialisation
	indices = rng.choice(range(n), k, False)
	centers = items[indices]

	while True:
		# Sum of all points matched to each center
		center_totals = np.zeros((k, dimensions), dtype=dtype)

		# Number of points matched to each center
		center_matches = np.zeros(k, dtype=np.uint32)

		for item in items:
			close_idx = find_closest(item, centers)

			center_matches[close_idx] += 1
			center_totals[close_idx] += item

		# Calculate each center's mean value
		center_means = np.copy(center_totals)
		reset_center = False
		for i in range(k):
			if center_matches[i] == 0:
				# raise ZeroDivisionError("A cluster had no matches")
				# If a center has no matches, set it to a random point in the dataset
				centers_means[i] = items[rng.choice(range(n), 1)]
				reset_center = True
				continue

			center_means[i] /= center_matches[i]

		# If the new centers match the previous centers, return
		if not reset_center and np.array_equal(centers, center_means):
			return centers;

		# TODO: Count how many times centers have been reset. If it happens a lot, exit with an exception

		# Otherwise, use the means as the centers for the next iteration
		centers = center_means

# Create an indexed image using a given image and a palette of colours.
# No dithering. Take a closest colour to each pixel value.
#
# param img     - Image as a numpy array of size (height, width, depth)
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
# param img     - Image as a numpy array of size (height, width, depth)
# param palette - Array of colours to use, in the same colour space as img.
#                 Array dimensions: (n, depth) where n is
#                 the number of colours in the palette.
#
# return        - Indexed image as (height, width) size array of ints
def create_indexed_img_dithered(img, palette):
	# Weighting of quantisation error difused to neighbouring pixels.
	# In order of (y, x): (+0, +1), (+1, -1), (+1, +0), (+1, +1)
	# Traditional Floyd-Steinberg uses [7, 3, 5, 1].
	DIFFUSION_COEFFS = [5, 2, 3, 1]

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

# Convert a colour palette to a different colour space.
#
# param palette     - n*depth array of colours, where n is the
#                     number of colours in the palette.
# param conversions - Values which may be passed to cv2.cvtColor to
#                     define the input and output colour spaces.
#
# return            - Colour palette in the new colour space,
#                     in a similar format as the input.
def convert_palette(palette, *conversions):
	# cvtColour requires a 3d array, so reshape this into n*1*depth
	n, depth = palette.shape
	palette_img = np.reshape(palette, (n, 1, depth))
	for conversion in conversions:
		palette_img = cv2.cvtColor(palette_img, conversion)

	return np.reshape(palette_img, (n, depth))

# Convert an HSV colour palette to New Horizon's subset of the HSV colour space.
#
# param palette - Array of colours in HSV.
# return        - Array of colours in New Horizon's colour space.
def palette_hsv_to_nh(palette):
	palette_size, depth = palette.shape
	nh_palette = np.zeros((palette_size, 3), dtype=np.int8)

	for i in range(palette_size):
		nh_colour = nh_palette[i]
		h, s, v = palette[i]

		nh_colour[0] = round(h / HSV_H_MAX *  NH_DEPTH_H) % NH_DEPTH_H
		nh_colour[1] = round(s / HSV_S_MAX * (NH_DEPTH_S - 1))
		nh_colour[2] = round(v / HSV_V_MAX * (NH_DEPTH_V - 1))

	# TODO: Handle duplicate colours better here.
		# Try to check what exact colours they came from and round up/down to distinguish them
		# Remember that modifying some colours may result in new duplicates, maybe just repeating the process will fix this eventually?

	return nh_palette

# Convert a palette of New Horizon colours to the HSV range used by OpenCV2
def palette_nh_to_hsv(palette):
	palette_size, depth = palette.shape
	hsv_palette = np.zeros((palette_size, 3), dtype=np.float32)

	for i in range(palette_size):
		nh_colour  =     palette[i]
		hsv_colour = hsv_palette[i]

		hsv_colour[0] = nh_colour[0] /  NH_DEPTH_H      * HSV_H_MAX
		hsv_colour[1] = nh_colour[1] / (NH_DEPTH_S - 1) * HSV_S_MAX
		hsv_colour[2] = nh_colour[2] / (NH_DEPTH_V - 1) * HSV_V_MAX

	return hsv_palette

# Find duplicate colours in a palette.
#
# param palette - Colour palette as array
# return        - List of duplicate colours
def get_duplicate_colours(palette):
	# Sort colours
	palette_record = np.core.records.fromarrays(
		palette.transpose(), names="h, s, v")
	palette_record.sort()

	# Look for consecutive, identical colours
	palette_size, = palette_record.shape
	duplicate_colours = []
	for i in range(0, palette_size - 1):
		this_col = palette_record[i]
		# Log this colour is a duplicate if it matches the next one.
		# Skip adding it if it's already in the list of duplicates
		if (np.array_equal(this_col, palette_record[i + 1])
				and not (len(duplicate_colours) > 0
					and np.array_equal(this_col, duplicate_colours[-1]))):
			duplicate_colours.append(this_col)

	return duplicate_colours

# Write a single channel from the palette to the file.
#
# param file         - File to write to.
# param channel_idx  - Index of this channel in the colour space.
# param channel_name - Name of this channel in the colour space.
# param header_width - Characters to reserve at the start of
#                      each line for the names of channels.
# param palette      - Array of New Horizons coloursd.
def _write_palette_channel(file, channel_idx, channel_name, header_width, palette):
	file.write(channel_name.rjust(header_width) + ": ")

	for colour in palette:
		file.write(str(colour[channel_idx]).rjust(2) + "  ")

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
			file.write(char.rjust(2))

		file.write("\n")

		if y % 16 == 15:
			file.write("\n")

# Write instructions on how to draw this pattern to a file.
#
# param path_out    - Path to file to write to.
# param indexed_img - Array of colour indices for each pixel of the image.
# param palette     - Array of New Horizons colours
def write_instructions(path_out, indexed_img, palette, *, pattern_name="your pattern"):
	height, width   = indexed_img.shape
	palette_size, _ =     palette.shape

	with open(path_out, "w") as file:
		file.write(f"How to draw {pattern_name}:\n\n")

		file.write("Colour palette:\n")
		header_width = 11
		_write_palette_channel(file, 0, "Hue",        header_width, palette)
		_write_palette_channel(file, 1, "Vividness",  header_width, palette)
		_write_palette_channel(file, 2, "Brightness", header_width, palette)
		file.write("\n")

		# Print map for each colour in the palette.
		display_chars = [EMPTY_COLOUR] * palette_size
		for i, col in enumerate(palette):
			file.write(f"Colour {i} {col}:\n")
			display_chars[i] = ACTIVE_COLOUR
			_write_indexed_img(file, indexed_img, display_chars)
			display_chars[i] = DISPLAY_CHARS[i]
			file.write("\n")

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

def analyse_img(path, img_out, instr_out, *, verbosity=Log.INFO, seed=None):
	# NH allows 15 colours + transparent
	PALETTE_SIZE = 15

	# TODO: parameter for this
	USE_DITHERING = True

	# TODO: Check for and handle transparency properly
	log = Log(verbosity)

	if seed is None:
		seed = np.random.randint(2 ** 32 - 1)
		log.info("Using seed:", seed)

	# Input image as BGR(255)
	log.info("Opening image...")
	img_raw = cv2.imread(path)
	if img_raw is None:
		raise FileNotFoundError()

	# TODO: resize image
	log.info("Converting to Lab...")
	img_lab = cv2.cvtColor(img_raw.astype(np.float32) / 255, cv2.COLOR_BGR2Lab)

	# Generate colour palette using K-Means
	log.info("Finding suitable colour palette...")
	height, width, depth = img_lab.shape
	colours = img_lab.reshape((height * width, depth))
	colour_palette = k_means(colours, PALETTE_SIZE, seed=seed)
	log.debug("Lab Palette:", colour_palette)

	# Convert the palette into HSV
	log.info("Converting palette to HSV...")
	hsv_palette = convert_palette(colour_palette, cv2.COLOR_Lab2RGB, cv2.COLOR_RGB2HSV)
	log.debug("HSV Palette:", hsv_palette)

	# Round to ACNH's colour space
	log.info("Converting to NH colours...")
	nh_palette = palette_hsv_to_nh(hsv_palette)

	# Check if any colours are identical after rounding
	# TODO: Option to automatically advance seed and retry if there is >=1 duplicate
	duplicate_colours = get_duplicate_colours(nh_palette)
	if len(duplicate_colours) > 0:
		# TODO: Allow stacking of --quiet to hide warnings too
		log.warn("Repeated colours in palette:", *duplicate_colours)

	# Convert the NH colours to BGR(1) to Lab
	hsv_approximated = palette_nh_to_hsv(nh_palette)
	log.debug("NH-HSV Palette:", hsv_approximated)
	bgr_approx = convert_palette(hsv_approximated, cv2.COLOR_HSV2BGR)
	log.debug("NH-BGR Palette:", bgr_approx)

	# Create indexed image using clusters
	log.info("Creating indexed image...")
	if USE_DITHERING:
		lab_approx = convert_palette(bgr_approx, cv2.COLOR_BGR2Lab)
		indexed_img = create_indexed_img_dithered(img_lab, lab_approx)
	else:
		indexed_img = create_indexed_img_threshold(img_lab, colour_palette)

	# Print drawing instructions
	write_instructions(instr_out, indexed_img, nh_palette)

	# Generate BGR image using colour map and the BGR version of the approximated colour space
	# Export approximated image
	log.info("Exporting image...")
	output_preview_image(img_out, indexed_img, bgr_approx)

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description=
		"Convert an image to a New Horizons custom pattern")

	# TODO: Option to pass a weight map, to allow some areas to be given higher priority when performing k-means to choose a palette

	parser.add_argument("input-file", help="Path to input image")
	parser.add_argument("-o", "--out", default="nh-pattern.png",
		help="Path to save output preview image")
	parser.add_argument("-i", "--instructions-out", default="nh-pattern-instructions.txt",
		help="Path to save pattern instructions")
	parser.add_argument("-s", "--seed", type=int, default=None,
		help="RNG seed for K-Means initialisation")

	verbosity_group = parser.add_mutually_exclusive_group();
	verbosity_group.add_argument("-q", "--quiet", action="store_true",
		help="Don't print any info messages to stdout")
	verbosity_group.add_argument("-v", "--verbose", action="store_true",
		help="Print more debug info to stdout");
	# TODO: Option to output palette as an image
	# TODO: Option to force no repeat colours in the palette (increment seed until no repeats)

	args = parser.parse_args()

	input_file = getattr(args, "input-file")

	verbosity = Log.INFO
	if args.quiet:
		verbosity -= 1
	if args.verbose:
		verbosity += 1

	try:
		analyse_img(input_file, img_out=args.out, instr_out=args.instructions_out,
			verbosity=verbosity, seed=args.seed);
	except FileNotFoundError:
		sys.stderr.write("File does not exist or is not a valid image\n")
		sys.exit(1)
