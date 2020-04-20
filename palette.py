#!/bin/python3

# Functions related to choosing, and using colour palettes.

import numpy as np
import cv2

import colour
from log import Log

# File names used to output preview images
DEFAULT_PREVIEW_FILENAME_NH = "nh-pattern.png"
DEFAULT_PREVIEW_FILENAME_NL = "nl-pattern.png"

_new_leaf_rgb = None
_new_leaf_lab = None

# Convert and return the New Leaf global palette in RGB in the range 0 to 255
#
# return - Tuple of:
#          - array of colour ids
#          - array of colours in RGB
def get_nl_rgb():
	global _new_leaf_rgb

	if _new_leaf_rgb is not None:
		return _new_leaf_rgb

	from newleafpalette import NEW_LEAF_GLOBAL_RGB

	n = len(NEW_LEAF_GLOBAL_RGB)

	colour_ids = np.zeros(n, dtype=np.uint8)
	colours_rgb = np.zeros((n, 3), dtype=np.uint8)

	for i in range(n):
		id, colour_hex = NEW_LEAF_GLOBAL_RGB[i]
		colour_ids[i] = id

		colours_rgb[i, 0] = (colour_hex & 0xff0000) >> 16
		colours_rgb[i, 1] = (colour_hex & 0x00ff00) >>  8
		colours_rgb[i, 2] =  colour_hex & 0x0000ff

	_new_leaf_rgb = (colour_ids, colours_rgb)
	return colour_ids, colours_rgb

# Convert and return the New Leaf global palette in the Lab colour space.
#
# return - Tuple of:
#          - array of colour ids
#          - array of colours in Lab
def get_nl_lab():
	global _new_leaf_lab

	if _new_leaf_lab is not None:
		return _new_leaf_lab

	colour_ids, colours_rgb = get_nl_rgb()
	colours_rgb = colours_rgb.astype(np.float32) / 255
	colours_lab = colour.convert_palette(colours_rgb, cv2.COLOR_RGB2Lab)

	_new_leaf_lab = (colour_ids, colours_lab)
	return colour_ids, colours_lab

# Convert a palette of Lab colours to New Leaf palette IDs
def palette_to_nl(lab_palette):
	palette_size, _ = lab_palette.shape

	nl_global_ids, nl_global_colours = get_nl_lab()

	nl_palette = np.zeros(palette_size, dtype=np.uint8)
	for i in range(palette_size):
		# Find closest NL colour
		idx = find_closest(lab_palette[i], nl_global_colours)
		nl_palette[i] = nl_global_ids[idx]
	
	return nl_palette

# Takes a palette of New Leaf colour indices and returns the palette in BGR format in the range 0 to 1.
def nl_to_bgr(nl_palette):
	nl_global_ids, nl_global_rgb = get_nl_rgb()

	palette_size, = nl_palette.shape
	palette_bgr = np.zeros((palette_size, 3), np.float32)

	for i in range(palette_size):
		# Look up colour in list of New Leaf colours
		matches = np.where(nl_global_ids == nl_palette[i])[0]
		if len(matches) == 0:
			raise ValueError(f"Could not find {hex(nl_palette[i])} in list of New Leaf colours")

		idx = matches[0]
		r, g, b = nl_global_rgb[idx]
		palette_bgr[i] = (b, g, r)

	return palette_bgr / 255

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

# Attempt to find an exact palette for the image. This will only succeed if the list of colours has `max_size`
# unique colours or fewer. If there are more unique colours, this operation is impossible, and the function will
# return None.
#
# param colours  - NumPy array of colours
# param max_size - Maximum number of colours allowed in the palette
#
# return         - NumPy array of palette colours if possible, or None if not.
def find_small_palette(colours, max_size=15):
	_, depth = colours.shape
	palette = []

	# Iterate through each colour in the image
	for col in colours:
		found_colour = False

		for palette_colour in palette:
			if np.array_equal(col, palette_colour):
				found_colour = True
				break

		if found_colour:
			continue

		# If this colour is not yet in the palette, add it
		palette.append(col)

		# If this makes the palette too big, this image has too many
		# colours to produce an exact palette, so return None here.
		if len(palette) > max_size:
			return None

	return np.array(palette)

# Take an image in Lab colour space, and output a colour palette in NH colours
#
# param img           - Input Lab image.
# param palette_size  - Number of colours to use in the palette.
# param seed          - Starting RNG seed.
# param retry_on_dupe - If truthy, when a palette containing a duplicate colour is generated, the RNG seed will
#                       be incremented, and the process will repeat until a unique palette is returned.
# new_leaf              If truthy, a New Leaf style palette will be generated instead of a New Horizons one.
# param log           - Log to use to output info.
#
# return              - Tuple of (NH Palette as array, Lab palette as array)
def pick_palette(img, palette_size, seed, weight_map=None, *, retry_on_dupe=False, new_leaf=False, log=Log(Log.ERROR)):
	# Generate colour palette using K-Means
	log.info("Finding suitable colour palette...")
	height, width, depth = img.shape
	colours = img.reshape((height * width, depth))

	small_palette = find_small_palette(colours, 15)
	if small_palette is not None:
		log.info("Found exact palette")
		log.debug("Palette: \n", small_palette)

		hsv_palette = colour.convert_palette(small_palette, cv2.COLOR_Lab2RGB, cv2.COLOR_RGB2HSV)
		nh_palette = colour.palette_hsv_to_nh(hsv_palette)
		log.debug("NH Palette:\n", nh_palette)

		duplicate_colours = colour.get_duplicate_colours(nh_palette)
		if len(duplicate_colours) > 0:
			log.warn("Repeated colours in palette:", *duplicate_colours)

		return nh_palette, small_palette

	if weight_map is not None:
		weight_map = weight_map.reshape(height * width)
	else:
		weight_map = np.ones(height * width)

	while True:
		colour_palette = k_means(colours, palette_size, weight_map, seed=seed)
		log.debug("Lab Palette:", colour_palette)

		if new_leaf:
			# Round to ACNL's palette.
			log.info("Converting to New Leaf colours...")
			game_palette = palette_to_nl(colour_palette)
			log.debug("NL Palette:", ", ".join(np.vectorize(hex)(game_palette)))
		else:
			# Convert the palette into HSV
			log.info("Converting palette to HSV...")
			hsv_palette = colour.convert_palette(colour_palette, cv2.COLOR_Lab2RGB, cv2.COLOR_RGB2HSV)
			log.debug("HSV Palette:\n", hsv_palette)

			# Round to ACNH's colour space 
			log.info("Converting to NH colours...")
			game_palette = colour.palette_hsv_to_nh(hsv_palette)
			log.debug("NH Palette:\n", game_palette)

		# Check if any colours are identical after rounding
		duplicate_colours = colour.get_duplicate_colours(game_palette)
		if len(duplicate_colours) <= 0:
			return game_palette, colour_palette

		if not retry_on_dupe:
			if new_leaf:
				log.warn("Repeated colours in palette:", *[hex(c[0]) for c in duplicate_colours])
			else:
				log.warn("Repeated colours in palette:", *duplicate_colours)
			return game_palette, colour_palette

		seed += 1
		log.info("Palette has duplicate colours. Retrying with seed:", seed)

# Write a preview of the pattern to a file.
# Expects the palette to already be converted to BGR from 0 to 1.
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
