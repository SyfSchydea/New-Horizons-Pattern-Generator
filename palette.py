#!/bin/python3

# Functions related to choosing, and using colour palettes.

import numpy as np
import cv2

import colour
from log import Log

# New Leaf's global colour palette.
NEW_LEAF_GLOBAL_RGB = [
	# (id, rgb_code)

	# Pink square
	(0x00, 0xffefff),
	(0x01, 0xff9aad),
	(0x02, 0xef559c),
	(0x03, 0xff65ad),
	(0x04, 0xff0063),
	(0x05, 0xbd4573),
	(0x06, 0xce0052),
	(0x07, 0x9c0031),
	(0x08, 0x522031),

	# Red square
	(0x10, 0xffbace),
	(0x11, 0xff7573),
	(0x12, 0xde3010),
	(0x13, 0x445542),
	(0x14, 0xff0000),
	(0x15, 0xce6563),
	(0x16, 0xbd4542),
	(0x17, 0xbd0000),
	(0x18, 0x8c2021),

	# Orange square
	(0x20, 0xdecfbd),
	(0x21, 0xffcf63),
	(0x22, 0xde6521),
	(0x23, 0xffaa21),
	(0x24, 0xff6500),
	(0x25, 0xbd8a52),
	(0x26, 0xde4500),
	(0x27, 0xbd4500),
	(0x28, 0x633010),

	# Beige square
	(0x30, 0xffefde),
	(0x31, 0xffdfce),
	(0x32, 0xffcfad),
	(0x33, 0xffba8c),
	(0x34, 0xffaa8c),
	(0x35, 0xde8a63),
	(0x36, 0xbd6542),
	(0x37, 0x9c5531),
	(0x38, 0x8c4521),

	# Purple square
	(0x40, 0xffcfff),
	(0x41, 0xef8aff),
	(0x42, 0xce65de),
	(0x43, 0xbd8ace),
	(0x44, 0xce00ff),
	(0x45, 0x9c659c),
	(0x46, 0x8c00ad),
	(0x47, 0x520073),
	(0x48, 0x310042),

	# Magenta square
	(0x50, 0xffbaff),
	(0x51, 0xff9aff),
	(0x52, 0xde20bd),
	(0x53, 0xff55ef),
	(0x54, 0xff00ce),
	(0x55, 0x8c5573),
	(0x56, 0xbd009c),
	(0x57, 0x8c0063),
	(0x58, 0x520042),

	# Brown square
	(0x60, 0xdeba9c),
	(0x61, 0xceaa73),
	(0x62, 0x734561),
	(0x63, 0xad7542),
	(0x64, 0x9c3000),
	(0x65, 0x733021),
	(0x66, 0x522000),
	(0x67, 0x311000),
	(0x68, 0x211000),

	# Yellow square
	(0x70, 0xffffce),
	(0x71, 0xffff73),
	(0x72, 0xdedf21),
	(0x73, 0xffff00),
	(0x74, 0xffdf00),
	(0x75, 0xceaa00),
	(0x76, 0x9c9a00),
	(0x77, 0x8c7500),
	(0x78, 0x525500),

	# Indigo square
	(0x80, 0xdebaff),
	(0x81, 0xbd9aef),
	(0x82, 0x6330ce),
	(0x83, 0x9c55ff),
	(0x84, 0x6300ff),
	(0x85, 0x52458c),
	(0x86, 0x42009c),
	(0x87, 0x210063),
	(0x88, 0x211031),

	# Blue square
	(0x90, 0xbdbaff),
	(0x91, 0x8c9aff),
	(0x92, 0x3130ad),
	(0x93, 0x3155ef),
	(0x94, 0x0000ff),
	(0x95, 0x31308c),
	(0x96, 0x0000ad),
	(0x97, 0x101063),
	(0x98, 0x000021),

	# Dark green square
	(0xa0, 0x9cefbd),
	(0xa1, 0x63cf73),
	(0xa2, 0x216510),
	(0xa3, 0x42aa31),
	(0xa4, 0x008a31),
	(0xa5, 0x527552),
	(0xa6, 0x215500),
	(0xa7, 0x103021),
	(0xa8, 0x002010),

	# Lime square
	(0xb0, 0xdeffbd),
	(0xb1, 0xceff8c),
	(0xb2, 0x8caa52),
	(0xb3, 0xaddf8c),
	(0xb4, 0x8cff00),
	(0xb5, 0xadba9c),
	(0xb6, 0x63ba00),
	(0xb7, 0x529a00),
	(0xb8, 0x316500),

	# Sky blue square
	(0xc0, 0xbddfff),
	(0xc1, 0x73cfff),
	(0xc2, 0x31559c),
	(0xc3, 0x639aff),
	(0xc4, 0x1075ff),
	(0xc5, 0x4275ad),
	(0xc6, 0x214573),
	(0xc7, 0x002073),
	(0xc8, 0x001042),

	# Turquoise square
	(0xd0, 0xadffff),
	(0xd1, 0x52ffff),
	(0xd2, 0x008abd),
	(0xd3, 0x52bace),
	(0xd4, 0x00cfff),
	(0xd5, 0x429aad),
	(0xd6, 0x00658c),
	(0xd7, 0x004552),
	(0xd8, 0x002031),

	# Cyan square
	(0xe0, 0xceffef),
	(0xe1, 0xadefde),
	(0xe2, 0x31cfad),
	(0xe3, 0x52efbd),
	(0xe4, 0x00ffce),
	(0xe5, 0x73aaad),
	(0xe6, 0x00aa9c),
	(0xe7, 0x008a73),
	(0xe8, 0x004531),

	# Light green square
	(0xf0, 0xadffad),
	(0xf1, 0x73ff73),
	(0xf2, 0x63df42),
	(0xf3, 0x00ff00),
	(0xf4, 0x21df21),
	(0xf5, 0x52ba52),
	(0xf6, 0x00ba00),
	(0xf7, 0x008a00),
	(0xf8, 0x214521),

	# Greyscale
	(0x0f, 0xffffff),
	(0x1f, 0xefefef),
	(0x2f, 0xdedfde),
	(0x3f, 0xcdcfce),
	(0x4f, 0xbdbabd),
	(0x5f, 0xadaaad),
	(0x6f, 0x9c9a9c),
	(0x7f, 0x8c8a8c),
	(0x8f, 0x737573),
	(0x9f, 0x636563),
	(0xaf, 0x525552),
	(0xbf, 0x424542),
	(0xcf, 0x313031),
	(0xdf, 0x212021),
	(0xef, 0x000000),
]

# Convert and return the New Leaf global palette in the Lab colour space.
#
# return - Tuple of:
#          - array of colour ids
#          - array of colours in Lab
def get_nl_lab():
	n = len(NEW_LEAF_GLOBAL_RGB)

	colour_ids = np.zeros(n, dtype=np.uint8)
	colours_rgb = np.zeros((n, 3), dtype=np.float32)

	for i in range(n):
		id, colour_hex = NEW_LEAF_GLOBAL_RGB[i]
		colour_ids[i] = id

		colours_rgb[i, 0] = (colour_hex & 0xff0000) >> 16
		colours_rgb[i, 1] = (colour_hex & 0x00ff00) >>  8
		colours_rgb[i, 2] =  colour_hex & 0x0000ff

	colours_lab = colour.convert_palette(colours_rgb / 255, cv2.COLOR_RGB2Lab)

	return colour_ids, colours_lab

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
