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

# Control Sequence Introducer
CSI = "\x1b["

# Class for setting tty colour/formatting
class TTY:
	def __init__(self, file=sys.stdout):
		self.file = file
		self.fg = 7

	# Set foreground colour
	#
	# param colour - Colour to set foreground to.
	#                Should be one of the colour constants defined below.
	def set_fg(self, colour=7):
		# Only allow colour values 0 to 9.
		colour %= 10

		# Avoid outputting a control code when switching
		# from "default" to "white" or vice-versa.
		if colour == TTY.RESET_COLOUR:
			colour = TTY.FG_DEFAULT

		# If the terminal is already outputting
		# in this colour, don't change that.
		if colour == self.fg:
			return

		# Write control code
		self.file.write(CSI + "3" + str(colour)[-1] + "m")
		self.fg = colour
	
	# Write text to the tty.
	#
	# param string - May be a str or a FormattedString.
	def write(self, string):
		if isinstance(string, FormattedString):
			self.set_fg(string.fg)
			string = string.string

		self.file.write(string)
	
	# Reset all colours and formatting to default
	def reset_all(self):
		self.set_fg(TTY.RESET_COLOUR)

# Colour values which may be set to foreground or background
TTY.BLACK        = 0
TTY.RED          = 1
TTY.GREEN        = 2
TTY.BLUE         = 4
TTY.YELLOW       = TTY.RED  | TTY.GREEN
TTY.MAGENTA      = TTY.RED  | TTY.BLUE
TTY.CYAN         = TTY.BLUE | TTY.GREEN
TTY.WHITE        = TTY.RED  | TTY.GREEN | TTY.BLUE
TTY.RESET_COLOUR = 9

# Normal terminal settings
TTY.FG_DEFAULT = TTY.WHITE

# Represents a string in a specific colour.
class FormattedString:
	def __init__(self, string, fg=TTY.RESET_COLOUR):
		self.string = string
		self.fg = fg
	
	# Set this string's foreground colour.
	# Returns itself to allow chaining.
	def set_fg(self, fg):
		self.fg = fg
		return self

# Characters used to represent each colour in ascii output
DISPLAY_CHARS = "abcdefghijklmnopqrstuvwxyz"
PREV_COLOUR = TTY.BLUE

EMPTY_CHAR = FormattedString("·").set_fg(TTY.FG_DEFAULT)

ACTIVE_CHAR = FormattedString("#").set_fg(TTY.MAGENTA)

# Distance between two Lab colours.
# Currently just euclidean distance.
#
# param colour_a - First colour to compare.
# param colour_b - Second colour to compare.
#
# return         - Distance between the two colours.
def lab_distance(colour_a, colour_b):
	return np.linalg.norm(colour_a - colour_b)

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
		distance = lab_distance(item, center)

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

	# Generate first center randomly
	first_idx = rng.choice(range(n))
	centers[0] = items[first_idx]

	# For each remaining center
	for center_idx in range(1, k):
		prev_center = centers[center_idx - 1]
		furthest_distance = -1
		furthest_center = None
		for item_idx in range(n):
			# Update item distance
			item = items[item_idx]
			dist = lab_distance(item, prev_center)
			if dist < item_distances[item_idx]:
				item_distances[item_idx] = dist

			# Find item furthest from a center
			dist = item_distances[item_idx]
			if dist > furthest_distance:
				furthest_distance = dist
				furthest_center = item

		# Use as next center
		centers[center_idx] = furthest_center

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

	return nh_palette

# Convert a palette of New Horizon colours to the HSV range used by OpenCV
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
# param palette      - Array of New Horizons colours.
def _write_palette_channel(file, channel_idx, channel_name, header_width, palette):
	file.write(channel_name.rjust(header_width) + ": ")
	file.write("  ".join(str(colour[channel_idx]).rjust(2) for colour in palette))
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
def write_instructions(path_out, indexed_img, palette, *, pattern_name="your pattern"):
	empty_pixel  = EMPTY_CHAR
	active_pixel = ACTIVE_CHAR

	height, width   = indexed_img.shape
	palette_size, _ =     palette.shape

	with open(path_out, "w") as file:
		tty = TTY(file)
		tty.write(f"How to draw {pattern_name}:\n\n")

		tty.write("Colour palette:\n")
		header_width = 11
		_write_palette_channel(tty, 0, "Hue",        header_width, palette)
		_write_palette_channel(tty, 1, "Vividness",  header_width, palette)
		_write_palette_channel(tty, 2, "Brightness", header_width, palette)
		tty.write("\n")

		# Print map for each colour in the palette.
		display_chars = [empty_pixel] * palette_size
		for i, col in enumerate(palette):
			colour_str = "(" + " ".join(str(x) for x in col) + ")"
			tty.write(f"Colour {i} {colour_str}:\n")
			display_chars[i] = active_pixel
			_write_indexed_img(tty, indexed_img, display_chars)
			prev_pixel = FormattedString(DISPLAY_CHARS[i]).set_fg(PREV_COLOUR)
			display_chars[i] = prev_pixel
			tty.write("\n")

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
		hsv_palette = convert_palette(colour_palette, cv2.COLOR_Lab2RGB, cv2.COLOR_RGB2HSV)
		log.debug("HSV Palette:", hsv_palette)

		# Round to ACNH's colour space
		log.info("Converting to NH colours...")
		nh_palette = palette_hsv_to_nh(hsv_palette)

		# Check if any colours are identical after rounding
		duplicate_colours = get_duplicate_colours(nh_palette)
		if len(duplicate_colours) <= 0:
			return nh_palette, colour_palette

		if not retry_on_dupe:
			log.warn("Repeated colours in palette:", *duplicate_colours)
			return nh_palette, colour_palette

		seed += 1
		log.info("Palette has duplicate colours. Retrying with seed:", seed)

def analyse_img(path, weight_map_path, img_out, instr_out, *,
		seed=None, retry_on_dupe=False, use_dithering=True,
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
	hsv_approximated = palette_nh_to_hsv(nh_palette)
	log.debug("NH-HSV Palette:", hsv_approximated)
	bgr_approx = convert_palette(hsv_approximated, cv2.COLOR_HSV2BGR)
	log.debug("NH-BGR Palette:", bgr_approx)

	# Create indexed image using clusters
	log.info("Creating indexed image...")
	if use_dithering:
		lab_approx = convert_palette(bgr_approx, cv2.COLOR_BGR2Lab)
		indexed_img = create_indexed_img_dithered(img_lab, lab_approx)
	else:
		indexed_img = create_indexed_img_threshold(img_lab, lab_palette)

	# Print drawing instructions
	write_instructions(instr_out, indexed_img, nh_palette)

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

	verbosity_group = parser.add_mutually_exclusive_group();
	verbosity_group.add_argument("-q", "--quiet", action="count",
		help="Print less information to stdout. May be stacked "
			+ "up to three times to hide warnings and errors")
	verbosity_group.add_argument("-v", "--verbose", action="store_true",
		help="Print more debug info to stdout");

	args = parser.parse_args()

	input_file = getattr(args, "input-file")

	verbosity = Log.INFO
	if args.quiet is not None:
		verbosity -= args.quiet
	if args.verbose:
		verbosity += 1

	try:
		analyse_img(input_file, weight_map_path=args.weight_map,
			img_out=args.out, instr_out=args.instructions_out,
			seed=args.seed, retry_on_dupe=args.retry_duplicate,
			use_dithering=args.dithering,
			verbosity=verbosity);
	except FileNotFoundError:
		sys.stderr.write("File does not exist or is not a valid image\n")
		sys.exit(1)

